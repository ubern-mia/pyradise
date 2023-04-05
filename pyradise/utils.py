import os
from re import sub
from typing import Iterable, Sequence, Sized, Tuple

import itk
import numpy as np
import SimpleITK as sitk
from pydicom import Dataset, dcmread
from pydicom.tag import Tag


def is_dir_and_exists(path: str) -> str:
    """Check if the provided path specifies a directory and if it exists.

    Args:
        path (str): The path to test.

    Returns:
        str: The normalized path.
    """
    if not os.path.exists(path):
        raise NotADirectoryError(f"The path {path} is not existing!")

    if not os.path.isdir(path):
        raise NotADirectoryError(f"The path {path} is not a directory!")

    return os.path.normpath(path)


def is_file_and_exists(path: str) -> str:
    """Check if the provided path specifies a file and if it exists.

    Args:
        path (str): The path to test.

    Returns:
        str: The normalized path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} is not existing!")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"The path {path} is not a file!")

    return os.path.normpath(path)


def is_dicom_file(path: str) -> bool:
    """Check if a path is specifying a DICOM file.

    Args:
        path (str): The path to test.

    Returns:
        bool: True if the path is specifying a DICOM file, False otherwise.
    """

    file_name = os.path.basename(path)
    if "." in file_name:
        file_extension = file_name.split(".")[-1]
        return file_extension.lower() == "dcm"

    with open(path, "rb") as fp:
        fp.seek(128)
        return fp.read(4).decode("utf-8") == "DICM"


def assume_is_segmentation(path: str) -> bool:
    """Assume if the image is a segmentation image based on the pixel data type or the SOPClassUID for DICOM files.

    Notes:
        Assume that a segmentation image has the pixel data type unsigned char.

    Args:
        path (str): The path to the image file.

    Returns:
        bool: True if the image is assumed to be a segmentation image, False otherwise.
    """
    if any([entry in path for entry in (".nii", ".nrrd", ".mha")]):
        reader = sitk.ImageFileReader()
        reader.LoadPrivateTagsOn()
        reader.SetFileName(path)
        reader.ReadImageInformation()
        if reader.GetPixelIDValue() == 1:
            return True
        return False

    elif is_dicom_file(path):
        dataset = load_dataset_tag(path, (Tag(0x0008, 0x0016),))
        sop_class_uid = str(dataset.SOPClassUID)
        if sop_class_uid == "1.2.840.10008.5.1.4.1.1.66.4":
            return True
        return False

    else:
        raise ValueError(f"The path {path} specifies a not supported file type!")


def convert_to_sitk_image(image: itk.Image) -> sitk.Image:
    """Convert an :class:`itk.Image` to a :class:`SimpleITK.Image`.

    Args:
        image (itk.Image): The :class:`itk.Image` to be converted.

    Returns:
        sitk.Image: The converted :class:`SimpleITK.Image`.
    """
    if image.GetImageDimension() > 3:
        raise NotImplementedError(f"Conversion of {image.GetDimension()}D images is not supported!")

    is_vector_image = image.GetNumberOfComponentsPerPixel() > 1
    image_sitk = sitk.GetImageFromArray(itk.GetArrayFromImage(image), isVector=is_vector_image)
    image_sitk.SetOrigin(tuple(image.GetOrigin()))
    image_sitk.SetSpacing(tuple(image.GetSpacing()))
    image_sitk.SetDirection(itk.GetArrayFromMatrix(image.GetDirection()).flatten())
    return image_sitk


def convert_to_itk_image(image: sitk.Image) -> itk.Image:
    """Convert a :class:`SimpleITK.Image` to an :class:`itk.Image`.

    Args:
        image (sitk.Image): The :class:`SimpleITK.Image` to be converted.

    Returns:
        itk.Image: The converted :class:`itk.Image`.
    """
    if image.GetDimension() > 3:
        raise NotImplementedError(f"Conversion of {image.GetDimension()}D images is not supported!")

    is_vector_image = image.GetNumberOfComponentsPerPixel() > 1
    image_itk = itk.GetImageFromArray(sitk.GetArrayFromImage(image), is_vector=is_vector_image)
    image_itk.SetOrigin(image.GetOrigin())
    image_itk.SetSpacing(image.GetSpacing())
    image_itk.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(image.GetDirection()), [3] * 2)))
    return image_itk


def assume_is_intensity_image(path: str) -> bool:
    """Assume if the image is an intensity image based on the pixel data type or the SOPClassUID for DICOM files.

    Notes:
        Assume that an intensity image has a pixel data type which is different from unsigned char.

    Args:
        path (str): The path to the image file.

    Returns:
        bool: True if the image is assumed to be an intensity image, False otherwise.
    """
    if any([entry in path for entry in (".nii", ".nrrd", ".mha")]):
        reader = sitk.ImageFileReader()
        reader.LoadPrivateTagsOn()
        reader.SetFileName(path)
        reader.ReadImageInformation()
        if reader.GetPixelIDValue() != 1:
            return True
        return False

    elif is_dicom_file(path):
        dataset = load_dataset_tag(path, (Tag(0x0008, 0x0016),))
        sop_class_uid = str(dataset.SOPClassUID)
        sop_class_last_uid = int(sop_class_uid.split(".")[9])
        if "1.2.840.10008.5.1.4.1.1" in sop_class_uid and sop_class_last_uid <= 20:
            return True
        return False

    else:
        raise ValueError(f"The path {path} specifies a not supported file type!")


def remove_illegal_folder_chars(name: str) -> str:
    """Removes illegal characters from a folder name.

    Args:
        name (str): The folder name with potential illegal characters.

    Returns:
        str: The folder name without illegal characters.
    """
    illegal_characters = r"""[<>:/\\|?*\']|[\0-\31]"""
    return sub(illegal_characters, "", name)


def load_dataset(path: str, stop_before_pixels: bool = True) -> Dataset:
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


def load_dataset_tag(path: str, tags: Sequence[Tag], stop_before_pixels: bool = True) -> Dataset:
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


def chunkify(iterable: Sized, size: int) -> Iterable:
    """Separate data from an iterable into chunks of a certain size.

    Args:
        iterable (Sized): The iterable to separate.
        size (int): The size of the chunks.

    Returns:
        Iterable: A chunk.
    """
    assert size > 1, "The size for chunking the data can not be smaller than 1!"

    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def get_slice_direction(image_dataset: Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the slice direction from the :class:`Dataset`.

    Args:
        image_dataset (Dataset): The :class:`Dataset` from which the slice direction should be determined.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The directions in all three dimensions.
    """
    orientation = image_dataset.get("ImageOrientationPatient")

    row_direction = np.array(orientation[:3])
    column_direction = np.array(orientation[3:])
    slice_direction = np.cross(row_direction, column_direction)

    validate_directions = (
        np.allclose(np.dot(row_direction, column_direction), 0.0, atol=1e-3),
        np.allclose(np.linalg.norm(slice_direction), 1.0, atol=1e-3),
    )

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
    orientation = image_dataset.get("ImagePositionPatient")

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
