import pytest
import os

import SimpleITK as sitk

from pyradise.utils import (
    is_dir_and_exists,
    is_file_and_exists,
    is_dicom_file,
    assume_is_segmentation,
    convert_to_sitk_image,
    convert_to_itk_image,
    assume_is_intensity_image,
    remove_illegal_folder_chars,
    load_dataset,
    load_datasets,
    load_dataset_tag,
    chunkify,
    get_slice_direction,
    get_slice_position,
    get_spacing_between_slices,
)

from tests.unit.helpers.image_helpers import (
    get_sitk_intensity_image,
    get_sitk_segmentation_image,
)

sitk_img_1 = get_sitk_intensity_image(1)
sitk_seg_1 = get_sitk_segmentation_image(2)


def test_is_dir_and_exists_1():
    with pytest.raises(NotADirectoryError):
        is_dir_and_exists("not_a_file")


def test_is_dir_and_exists_2():
    this_file_path = os.path.realpath(__file__)
    with pytest.raises(NotADirectoryError):
        is_dir_and_exists(this_file_path)


def test_is_dir_and_exists_3():
    this_file_dir = os.path.dirname(os.path.realpath(__file__))
    assert is_dir_and_exists(this_file_dir) == this_file_dir


def test_is_file_and_exists_1():
    with pytest.raises(FileNotFoundError):
        is_file_and_exists("not_a_file")


def test_is_file_and_exists_2():
    this_file_dir = os.path.dirname(os.path.realpath(__file__))
    with pytest.raises(FileNotFoundError):
        is_file_and_exists(this_file_dir)


def test_is_file_and_exists_3():
    this_file_path = os.path.realpath(__file__)
    assert is_file_and_exists(this_file_path) == this_file_path


def test_is_dicom_file_1(image_file):
    assert is_dicom_file(str(image_file)) is False


def test_is_dicom_file_2(dcm_file):
    assert is_dicom_file(str(dcm_file)) is True


def test_is_dicom_file_3(dicm_folder):
    # todo: do test 63-66
    pass


def test_assume_is_segmentation_1(segmentation_file_1):
    assert assume_is_segmentation(str(segmentation_file_1)) is True


def test_assume_is_segmentation_2(image_file_2):
    assert assume_is_segmentation(str(image_file_2)) is False
