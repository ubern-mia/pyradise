import numpy as np
import pytest
import SimpleITK as sitk

from pyradise.data import IntensityImage, Subject
from pyradise.process.base import Filter, LoopEntryFilter, LoopEntryFilterParams


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


def test__register_tracked_data_1(img_file_nii):
    filter = TestFilter()
    sitk_image_1 = sitk.ReadImage(img_file_nii)
    img_1 = IntensityImage(sitk_image_1, "modality")
    filter._register_tracked_data(img_1, sitk_image_1, sitk_image_1)
    assert filter.tracking_data == {}


class TestLoopEntryFilter(LoopEntryFilter):
    def __init__(self):
        super().__init__()

    def is_invertible(self):
        return False

    def execute(self, subject, params) -> Subject:
        return subject

    def execute_inverse(self, subject, transform_info, target_image):
        return subject


def test_loop_entry_filter_params_1():
    params = LoopEntryFilterParams(1)
    assert params.loop_axis == 1


def test_loop_entry_filter_params_2():
    with pytest.raises(AssertionError):
        LoopEntryFilterParams(-1)


def test_loop_entry_filter_params_3():
    with pytest.raises(AssertionError):
        LoopEntryFilterParams(3)


def test_loop_entries():
    def dummy_filter_fn(data, params):
        return data, params

    array = np.ndarray([1, 0, 1, 0])
    filter_ = TestLoopEntryFilter()
    assert filter_.loop_entries(array, 0, dummy_filter_fn, None) == (array, 0)
