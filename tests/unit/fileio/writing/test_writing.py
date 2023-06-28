import SimpleITK as sitk

from pyradise.data import IntensityImage, SegmentationImage, Subject
from pyradise.fileio.writing import (
    default_intensity_file_name_fn,
    default_segmentation_file_name_fn,
)


def test_default_intensity_file_name_fn(img_file_nii):
    sub = Subject("test_name")
    img = IntensityImage(sitk.ReadImage(img_file_nii), "modality")
    file_name = default_intensity_file_name_fn(sub, img)
    assert file_name == "img_test_name_modality"
    assert isinstance(file_name, str)


def test_default_segmentation_file_name_fn(seg_file_nii):
    sub = Subject("test_name")
    seg = SegmentationImage(sitk.ReadImage(seg_file_nii), "organ", "annotation")
    file_name = default_segmentation_file_name_fn(sub, seg)
    assert file_name == "seg_test_name_annotation_organ"
    assert isinstance(file_name, str)
