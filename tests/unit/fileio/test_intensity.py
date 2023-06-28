import numpy as np
import pytest
import SimpleITK as sitk

from pyradise.data import IntensityImage, Modality, Subject, TransformInfo
from pyradise.process.intensity import (
    ClipIntensityFilter,
    ClipIntensityFilterParams,
    GaussianFilter,
    GaussianFilterParams,
    IntensityFilter,
    IntensityFilterParams,
    IntensityLoopFilter,
    IntensityLoopFilterParams,
    LaplacianFilterParams,
    MedianFilter,
    MedianFilterParams,
    RescaleIntensityFilter,
    RescaleIntensityFilterParams,
    ZeroOneNormFilter,
    ZeroOneNormFilterParams,
    ZScoreNormFilter,
    ZScoreNormFilterParams,
)


class HelperTransformInfo:
    def __init__(self):
        self.params = {
            "mean_0": 0.5,
            "std_0": 0.5,
        }

    def get_data(self, key):
        return self.params.get(key)


def test_intensity_filter_params_1():
    params = IntensityFilterParams(("modality",))
    assert params.modalities == (Modality("modality"),)


def test_intensity_filter_params_2():
    params = IntensityFilterParams(None)
    assert params.modalities is None


def test_intensity_filter_1():
    pass


def test_intensity_loop_filter_params_1():
    params = IntensityLoopFilterParams(0, ("modality",))
    assert params.modalities == (Modality("modality"),)


def test_intensity_loop_filter_params_2():
    params = IntensityLoopFilterParams(None)
    assert params.modalities is None


def test_intensity_loop_filter_1():
    pass


def test_zscore_norm_filter_params_1():
    params = ZScoreNormFilterParams(0, ("modality",))
    assert params.modalities == (Modality("modality"),)


def test_zscore_norm_filter_params_2():
    params = ZScoreNormFilterParams(None)
    assert params.modalities is None


def test_zscore_norm_filter_1():
    filter = ZScoreNormFilter()
    assert filter.is_invertible() is True


def test_zscore_norm_filter_2():
    filter = ZScoreNormFilter()
    array = np.array([1, 0, 1, 0])
    assert np.array_equal(filter._modify_array(array, None), [1.0, -1.0, 1.0, -1.0])


def test_zscore_norm_filter_3():
    filter = ZScoreNormFilter()
    post_array = np.array([1, -1, 1, -1])
    params = HelperTransformInfo()
    assert np.array_equal(
        filter._modify_array_inverse(post_array, params), [1.0, 0.0, 1.0, 0.0]
    )


def test_zscore_norm_filter_4():
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


def test_zscore_norm_filter_5():
    image = IntensityImage(
        sitk.GetImageFromArray(np.array([[1, -1], [1, -1]])), "modality"
    )
    s = Subject("test_name", image)
    filter = ZScoreNormFilter()
    params = HelperTransformInfo()
    # todo: inverse


def test_zero_one_norm_filter_params_1():
    params = ZeroOneNormFilterParams(0, ("modality",))
    assert params.modalities == (Modality("modality"),)


def test_zero_one_norm_filter_params_2():
    params = ZeroOneNormFilterParams(None)
    assert params.modalities is None


def test_rescale_intensity_filter_params_1():
    params = RescaleIntensityFilterParams(0, 1, ("modality",))
    assert params.modalities == (Modality("modality"),)
    assert params.min_out == 0
    assert params.max_out == 1


def test_rescale_intensity_filter_params_2():
    params = RescaleIntensityFilterParams(0, 1, None)
    assert params.modalities is None
    assert params.min_out == 0
    assert params.max_out == 1


def test_rescale_intensity_filter_params_3():
    params = RescaleIntensityFilterParams(1, 0, ("modality",))
    assert params.modalities == (Modality("modality"),)
    assert params.min_out == 0
    assert params.max_out == 1


def test_rescale_intensity_filter_params_4():
    with pytest.raises(ValueError):
        RescaleIntensityFilterParams(1, 1, None)


def test_clip_intensity_filter_params_1():
    params = ClipIntensityFilterParams(0, 1, ("modality",))
    assert params.modalities == (Modality("modality"),)
    assert params.min_value == 0
    assert params.max_value == 1


def test_clip_intensity_filter_params_2():
    params = ClipIntensityFilterParams(0, 1, None)
    assert params.modalities is None
    assert params.min_value == 0
    assert params.max_value == 1


def test_clip_intensity_filter_params_3():
    params = ClipIntensityFilterParams(1, 0, ("modality",))
    assert params.modalities == (Modality("modality"),)
    assert params.min_value == 0
    assert params.max_value == 1


def test_clip_intensity_filter_params_4():
    with pytest.raises(ValueError):
        ClipIntensityFilterParams(1, 1, None)


def test_gaussian_filter_params_1():
    params = GaussianFilterParams(1, 2, ("modality",))
    assert params.modalities == (Modality("modality"),)
    assert params.variance == 1
    assert params.kernel_size == 2


def test_gaussian_filter_params_2():
    params = GaussianFilterParams(1, 2, None)
    assert params.modalities is None
    assert params.variance == 1
    assert params.kernel_size == 2


def test_gaussian_filter_params_3():
    with pytest.raises(ValueError):
        GaussianFilterParams(0, 1, None)


def test_gaussian_filter_params_4():
    with pytest.raises(ValueError):
        GaussianFilterParams(1, 0, None)


def test_median_filter_params_1():
    params = MedianFilterParams(1, ("modality",))
    assert params.modalities == (Modality("modality"),)
    assert params.radius == 1


def test_median_filter_params_2():
    params = MedianFilterParams(1, None)
    assert params.modalities is None
    assert params.radius == 1


def test_laplace_filter_params_1():
    params = LaplacianFilterParams(("modality",))
    assert params.modalities == (Modality("modality"),)


def test_laplace_filter_params_2():
    params = LaplacianFilterParams(None)
    assert params.modalities is None
