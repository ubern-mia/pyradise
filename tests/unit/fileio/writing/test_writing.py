from pyradise.data import IntensityImage, SegmentationImage, Subject
from pyradise.fileio.writing import (
    ImageFileFormat,
    default_intensity_file_name_fn,
    default_segmentation_file_name_fn,
)
from tests.conftest import get_sitk_image

sitk_img_1 = get_sitk_image(seed=0, low=0, high=101, meta="nii")
sitk_seg_1 = get_sitk_image(seed=0, low=0, high=2, meta="nii")


def test_default_intensity_file_name_fn():
    sub = Subject("test_name")
    img = IntensityImage(sitk_img_1, "modality")
    file_name = default_intensity_file_name_fn(sub, img)
    assert file_name == "img_test_name_modality"
    assert isinstance(file_name, str)


def test_default_segmentation_file_name_fn():
    sub = Subject("test_name")
    seg = SegmentationImage(sitk_seg_1, "organ", "annotation")
    file_name = default_segmentation_file_name_fn(sub, seg)
    assert file_name == "seg_test_name_annotation_organ"
    assert isinstance(file_name, str)


def test_class_format():
    assert ImageFileFormat.NIFTI.value == ".nii"
    assert ImageFileFormat.NIFTI_GZ.value == ".nii.gz"
    assert ImageFileFormat.NRRD.value == ".nrrd"
    assert ImageFileFormat.MHA.value == ".mha"
