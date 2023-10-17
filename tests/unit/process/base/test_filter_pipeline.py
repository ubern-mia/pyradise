import logging

import pytest
import SimpleITK as sitk

from pyradise.data import IntensityImage, Subject
from pyradise.process.base import FilterPipeline
from pyradise.process.intensity import ZScoreNormFilter, ZScoreNormFilterParams


def test__init__1():
    filter_ = ZScoreNormFilter()
    params = ZScoreNormFilterParams(1, ("modality",))
    filter_pipeline = FilterPipeline((filter_,), (params,), False)
    assert filter_pipeline.warn_on_non_invertible is False
    assert filter_pipeline.filters == [filter_]
    assert filter_pipeline.params == [params]
    assert filter_pipeline.logger is None


def test__init__2():
    filter_ = ZScoreNormFilter()
    filter_pipeline = FilterPipeline((filter_,), None, False)
    assert filter_pipeline.warn_on_non_invertible is False
    assert filter_pipeline.filters == [filter_]
    assert filter_pipeline.params == [None]


def test__init__3():
    filter_ = ZScoreNormFilter()
    params = ZScoreNormFilterParams(1, ("modality",))
    filter_pipeline = FilterPipeline((filter_,), (params,), True)
    assert filter_pipeline.warn_on_non_invertible is True


def test__init__4():
    filter_ = ZScoreNormFilter()
    params = ZScoreNormFilterParams(1, ("modality",))
    with pytest.raises(ValueError):
        FilterPipeline(
            (
                filter_,
                filter_,
            ),
            (params,),
            False,
        )


def test_set_verbose_all_1():
    filter_ = ZScoreNormFilter()
    params = ZScoreNormFilterParams(1, ("modality",))
    filter_pipeline = FilterPipeline((filter_,), (params,), False)
    filter_pipeline.set_verbose_all(True)
    assert filter_pipeline.filters[0].verbose is True


def test_set_verbose_all_2():
    filter_ = ZScoreNormFilter()
    params = ZScoreNormFilterParams(1, ("modality",))
    filter_pipeline = FilterPipeline((filter_,), (params,), False)
    filter_pipeline.set_verbose_all(False)
    assert filter_pipeline.filters[0].verbose is False


def test_add_filter():
    filter_ = ZScoreNormFilter()
    params = ZScoreNormFilterParams(1, ("modality",))
    filter_pipeline = FilterPipeline((filter_,), (params,), False)
    assert len(filter_pipeline.filters) == 1
    assert len(filter_pipeline.params) == 1
    filter_pipeline.add_filter(filter_, params)
    assert len(filter_pipeline.filters) == 2
    assert len(filter_pipeline.params) == 2


def test_set_params_1():
    filter_ = ZScoreNormFilter()
    params = ZScoreNormFilterParams(1, ("modality",))
    filter_pipeline = FilterPipeline(
        (
            filter_,
            filter_,
        ),
        (
            params,
            params,
        ),
        False,
    )
    filter_pipeline.set_param(params, 0)
    filter_pipeline.set_param(params, 1)
    assert len(filter_pipeline.filters) == 2
    assert len(filter_pipeline.params) == 2


def test_set_params_2():
    filter_ = ZScoreNormFilter()
    params_0 = ZScoreNormFilterParams(1, ("modality_0",))
    params_1 = ZScoreNormFilterParams(1, ("modality_1",))
    filter_pipeline = FilterPipeline(
        (
            filter_,
            filter_,
        ),
        (
            params_0,
            params_1,
        ),
        False,
    )
    filter_pipeline.set_param(params_0, -1)
    assert filter_pipeline.params[1] == params_0


def test_add_logger():
    filter_ = ZScoreNormFilter()
    params = ZScoreNormFilterParams(1, ("modality",))
    filter_pipeline = FilterPipeline((filter_,), (params,), False)
    assert filter_pipeline.logger is None
    filter_pipeline.add_logger(1)
    assert filter_pipeline.logger == 1


def test_execute_iteratively_1(img_file_nii):
    filter_ = ZScoreNormFilter()
    params_0 = ZScoreNormFilterParams(1, ("modality_0",))
    params_1 = ZScoreNormFilterParams(1, ("modality_0",))
    image = IntensityImage(sitk.ReadImage(img_file_nii), "modality_0")
    input_subject = Subject("test_name", [image])
    filter_pipeline = FilterPipeline((filter_, filter_), (params_0, params_1), False)
    for index, (subject, filter_name) in enumerate(filter_pipeline.execute_iteratively(input_subject)):
        assert index < 2
        assert isinstance(subject, Subject)
        assert filter_name == "ZScoreNormFilter"


def test_execute_iteratively_2(img_file_nii):
    filter_ = ZScoreNormFilter()
    params_0 = ZScoreNormFilterParams(1, ("modality_0",))
    params_1 = ZScoreNormFilterParams(1, ("modality_0",))
    image = IntensityImage(sitk.ReadImage(img_file_nii), "modality_0")
    input_subject = Subject("test_name", [image])
    filter_pipeline = FilterPipeline((filter_, filter_), (params_0, params_1), True)
    logger = logging.getLogger("name1")
    filter_pipeline.add_logger(logger)
    for index, (subject, filter_name) in enumerate(filter_pipeline.execute_iteratively(input_subject)):
        assert index < 2
        assert isinstance(subject, Subject)
        assert filter_name == "ZScoreNormFilter"


def test_execute_iteratively_3(img_file_nii):
    filter_ = ZScoreNormFilter()
    params = ZScoreNormFilterParams(1, ("modality_0",))
    image = IntensityImage(sitk.ReadImage(img_file_nii), "modality_0")
    input_subject = Subject("test_name", [image])
    filter_pipeline = FilterPipeline(
        (filter_, filter_),
        (
            params,
            params,
        ),
        True,
    )
    filter_pipeline.params = (params,)
    with pytest.raises(ValueError):
        filter_pipeline.execute(input_subject)


def test_execute(img_file_nii):
    filter_ = ZScoreNormFilter()
    params_0 = ZScoreNormFilterParams(1, ("modality_0",))
    params_1 = ZScoreNormFilterParams(1, ("modality_0",))
    image = IntensityImage(sitk.ReadImage(img_file_nii), "modality_0")
    input_subject = Subject("test_name", [image])
    filter_pipeline = FilterPipeline((filter_, filter_), (params_0, params_1), True)
    assert isinstance(filter_pipeline.execute(input_subject), Subject)
