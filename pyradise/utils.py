import os
from re import sub
from typing import (Sequence, Tuple, Sized, Iterable)

import numpy as np
from pydicom import (dcmread, Dataset)
from pydicom.tag import Tag


def is_dir_and_exists(path: str) -> str:
    """Check if the provided path specifies a directory and if it exists.

    Args:
        path (str): The path to test.

    Returns:
        str: The normalized path.
    """
    if not os.path.exists(path):
        raise NotADirectoryError(f'The path {path} is not existing!')

    if not os.path.isdir(path):
        raise NotADirectoryError(f'The path {path} is not a directory!')

    return os.path.normpath(path)


def is_file_and_exists(path: str) -> str:
    """Check if the provided path specifies a file and if it exists.

    Args:
        path (str): The path to test.

    Returns:
        str: The normalized path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'The path {path} is not existing!')

    if not os.path.isfile(path):
        raise FileNotFoundError(f'The path {path} is not a file!')

    return os.path.normpath(path)


def is_dicom_file(path: str) -> bool:
    """Check if a path is specifying a DICOM file.

    Args:
        path (str): The path to test.

    Returns:
        bool: True if the path is specifying a DICOM file, False otherwise.
    """
    with open(path, 'rb') as fp:
        fp.seek(128)
        return fp.read(4).decode('utf-8') == 'DICM'


def remove_illegal_folder_chars(name: str) -> str:
    """Removes illegal characters from a folder name.

    Args:
        name (str): The folder name with potential illegal characters.

    Returns:
        str: The folder name without illegal characters.
    """
    illegal_characters = r"""[<>:/\\|?*\']|[\0-\31]"""
    return sub(illegal_characters, "", name)


def load_dataset(path: str,
                 stop_before_pixels: bool = True
                 ) -> Dataset:
    """Loads a DICOM dataset from a path.

    Args:
        path (str): The path to the DICOM file.
        stop_before_pixels (bool): If True the loading process will not be performed for the image data.

    Returns:
        Dataset: The :class:`Dataset` loaded.
    """
    dataset = dcmread(path, stop_before_pixels=stop_before_pixels)
    dataset.decode()
    return dataset


def load_datasets(paths: Sequence[str], stop_before_pixels: bool = True) -> Tuple[Dataset, ...]:
    """Load multiple DICOM datasets from multiple paths.

    Args:
        paths (Tuple[str]): The paths to the DICOM files.
        stop_before_pixels (bool): Indicates if the loading process should stop before the pixel data.

    Returns:
        Tuple[Dataset, ...]: The loaded :class:`Dataset` s.
    """
    datasets = []
    for path in paths:
        dataset = load_dataset(path, stop_before_pixels)
        datasets.append(dataset)

    return tuple(datasets)


def load_dataset_tag(path: str,
                     tags: Sequence[Tag],
                     stop_before_pixels: bool = True
                     ) -> Dataset:
    """Loads a DICOM dataset from a file with specific tags only.

    Args:
        path (str): The path to the DICOM file.
        tags (Sequence[Tag]): One or multiple tags to load from the file.
        stop_before_pixels (bool): If True the loading process will not be performed for the image data.

    Returns:
        Dataset: The :class:`Dataset` loaded with specific tags.
    """
    dataset = dcmread(path, stop_before_pixels=stop_before_pixels, specific_tags=list(tags))
    dataset.decode()
    return dataset


def chunkify(iterable: Sized,
             size: int
             ) -> Iterable:
    """Separate data from an iterable into chunks of a certain size.

    Args:
        iterable (Sized): The iterable to separate.
        size (int): The size of the chunks.

    Returns:
        Iterable: A chunk.
    """
    assert size > 1, 'The size for chunking the data can not be smaller than 1!'

    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def get_slice_direction(image_dataset: Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the slice direction from the :class:`Dataset`.

    Args:
        image_dataset (Dataset): The :class:`Dataset` from which the slice direction should be determined.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The directions in all three dimensions.
    """
    orientation = image_dataset.get('ImageOrientationPatient')

    row_direction = np.array(orientation[:3])
    column_direction = np.array(orientation[3:])
    slice_direction = np.cross(row_direction, column_direction)

    validate_directions = (np.allclose(np.dot(row_direction, column_direction), 0.0, atol=1e-3),
                           np.allclose(np.linalg.norm(slice_direction), 1.0, atol=1e-3))

    if not all(validate_directions):
        raise Exception(f'Invalid ImageOrientationPatient attribute in {image_dataset.get("PatientID")}!')

    return row_direction, column_direction, slice_direction


def get_slice_position(image_dataset: Dataset) -> np.ndarray:
    """Get the slice position from a :class:`Dataset`.

    Args:
        image_dataset (Dataset): The :class:`Dataset` from which the slice position should be determined.

    Returns:
        np.ndarray: The position of the slice in space.
    """
    orientation = image_dataset.get('ImagePositionPatient')

    _, _, slice_direction = get_slice_direction(image_dataset)

    return np.dot(slice_direction, orientation)


def get_spacing_between_slices(image_datasets: Tuple[Dataset, ...]) -> float:
    """Get the spacing between the slices based on the first and last slice positions.

    Args:
        image_datasets (Tuple[Dataset, ...]): The :class:`Dataset` s to get the spacing from

    Returns:
        float: The spacing between the slices.
    """
    if len(image_datasets) > 1:
        first = get_slice_position(image_datasets[0])
        last = get_slice_position(image_datasets[-1])
        return (last - first) / (len(image_datasets) - 1)

    return 1.0