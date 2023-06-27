from pyradise.process.base import (
    Filter,
)
from pyradise.data import (Image, ImageProperties, IntensityImage,
                           SegmentationImage, Subject, TransformInfo)

import SimpleITK as sitk

class TestFilter(Filter):

    def __init__(self):
        super().__init__()

    def is_invertible(self):
        """Return True if the filter is invertible, False otherwise."""
        return False

    def execute(self, subject, params) -> Subject:
        return subject

    def execute_inverse(self, subject, transform_info, target_image):
        return subject


def test_filter_set_verbose():
    filter = TestFilter()
    assert filter.verbose is False
    assert filter.set_verbose(True) is None
    assert filter.verbose is True
    assert isinstance(filter.verbose, bool)


def test_filter_set_warning_on_non_invertible():
    filter = TestFilter()
    filter.set_warning_on_non_invertible(True)
    assert filter.warn_on_non_invertible is True


def test__register_tracked_data(img_file_nii):
    filter = TestFilter()
    sitk_image_1 = sitk.ReadImage(img_file_nii)
    img_1 = IntensityImage(sitk_image_1, "modality")
    filter._register_tracked_data(img_1, sitk_image_1, sitk_image_1)
    # TODO:


