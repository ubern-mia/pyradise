from pyradise.process.base import (
    Filter,
)
from pyradise.data import (Image, ImageProperties, IntensityImage,
                           SegmentationImage, Subject, TransformInfo)

from tests.unit.helpers.image_helpers import get_sitk_image

sitk_image_1 = get_sitk_image(1, 1, 101, 'nii')
img_1 = IntensityImage(sitk_image_1, 'modality_1')


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


def test__register_tracked_data():
    filter = TestFilter()
    filter._register_tracked_data(img_1, sitk_image_1, sitk_image_1)
    # TODO:


