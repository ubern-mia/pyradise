import os
from typing import (Sequence, Tuple, Sized, Iterable)

from pydicom import (dcmread, Dataset)
from pydicom.tag import Tag


def check_is_dir_and_existing(path: str) -> None:
    """Checks if the provided path is specifying a directory and if that directory is existing.

    Args:
        path (str): The path to test.

    Returns:
        None
    """
    if not os.path.exists(path):
        raise NotADirectoryError(f'The path {path} is not existing!')

    if not os.path.isdir(path):
        raise NotADirectoryError(f'The path {path} is not a directory!')


def check_is_file_and_existing(path: str) -> None:
    """Checks if the provided path is specifying a file and if that file is existing.

    Args:
        path (str): The path to test.

    Returns:
        None
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'The path {path} is not existing!')

    if not os.path.isfile(path):
        raise FileNotFoundError(f'The path {path} is not a file!')


def load_dataset(path: str,
                 stop_before_pixels: bool = True
                 ) -> Dataset:
    """Loads a DICOM dataset from a path.

    Args:
        path (str): The path to the DICOM file.
        stop_before_pixels (bool): If True the loading process will not be performed for the image data.

    Returns:
        Dataset: The Dataset loaded.
    """
    dataset = dcmread(path, stop_before_pixels=stop_before_pixels)
    dataset.decode()
    return dataset


def load_datasets(paths: Sequence[str], stop_before_pixels: bool = True) -> Tuple[Dataset, ...]:
    """Load multiple DICOM datasets from multiple paths.

    Args:
        paths (Tuple[str]): The paths of the Datasets to load.
        stop_before_pixels (bool): Indicates if the loading process should stop before the pixel data.

    Returns:
        Tuple[Dataset, ...]: The loaded Datasets.
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
        Dataset: The Dataset loaded with specific tags.
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
