import numpy as np
import SimpleITK as sitk

from pyradise.data import IntensityImage, Modality, Subject
from pyradise.process.intensity import ZScoreNormFilter, ZScoreNormFilterParams


class HelperTransformInfo:
    def __init__(self):
        self.params = {
            "mean_0": 0.5,
            "std_0": 0.5,
        }

    def get_data(self, key):
        return self.params.get(key)


def test__init__1():
    params = ZScoreNormFilterParams(0, ("modality",))
    assert params.modalities == (Modality("modality"),)


def test__init__2():
    params = ZScoreNormFilterParams(None)
    assert params.modalities is None


def test_is_invertible():
    filter = ZScoreNormFilter()
    assert filter.is_invertible() is True


def test_modify_array():
    filter = ZScoreNormFilter()
    array = np.array([1, 0, 1, 0])
    assert np.array_equal(filter._modify_array(array, None), [1.0, -1.0, 1.0, -1.0])


def test_modify_array_inverse():
    filter = ZScoreNormFilter()
    post_array = np.array([1, -1, 1, -1])
    params = HelperTransformInfo()
    assert np.array_equal(
        filter._modify_array_inverse(post_array, params), [1.0, 0.0, 1.0, 0.0]
    )


def test_execute():
    image = IntensityImage(
        sitk.GetImageFromArray(np.array([[1, 0], [1, 0]])), "modality"
    )
    s = Subject("test_name", image)
    filter = ZScoreNormFilter()
    filter_params = ZScoreNormFilterParams(0, ("modality",))
    result = filter.execute(s, filter_params)
    assert np.array_equal(
        result.get_image_by_modality("modality").get_image_data_as_np(),
        [[1.0, -1.0], [1.0, -1.0]],
    )


def test_execute_inverse():
    image = IntensityImage(
        sitk.GetImageFromArray(np.array([[1, -1], [1, -1]])), "modality"
    )
    s = Subject("test_name", image)
    filter = ZScoreNormFilter()
    params = HelperTransformInfo()
    # todo: inverse
