from typing import TypeVar

import SimpleITK as sitk

from pyradise.data import ImageProperties, TransformInfo
from pyradise.process import FilterParams, ZScoreNormFilter


class TestFilterParams(FilterParams):
    def __init__(self) -> None:
        super().__init__()
        self.a = 1


filter_params = TestFilterParams()


class TestImageProperties(ImageProperties):
    def __init__(self, image) -> None:
        super().__init__(image=image, data={})


image_1 = sitk.Image(1, 1, 1, sitk.sitkUInt8)
image_2 = sitk.Image(1, 1, 1, sitk.sitkUInt8)
image_2.SetOrigin((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

image_properties_1 = TestImageProperties(image_1)
image_properties_2 = TestImageProperties(image_2)

IntensityImage = TypeVar("IntensityImage")
SegmentationImage = TypeVar("SegmentationImage")
FilterParameters = TypeVar("FilterParameters")


def test_get_subclasses():
    tra_info = TransformInfo("", filter_params, ImageProperties, ImageProperties)
    found_classes = tra_info._get_subclasses(FilterParams)
    assert isinstance(found_classes, dict)
    assert isinstance(found_classes["TestFilterParams"](), TestFilterParams)
    assert found_classes["TestFilterParams"]().a == 1


def test_get_filter():
    tra_info = TransformInfo(
        "ZScoreNormFilter", FilterParams, ImageProperties, ImageProperties
    )
    assert isinstance(tra_info.get_filter(), ZScoreNormFilter)


def test_get_params():
    tra_info = TransformInfo("", filter_params, ImageProperties, ImageProperties)
    assert isinstance(tra_info.get_params(), TestFilterParams)
    assert tra_info.get_params().a == 1


def test_get_image_properties():
    tra_info = TransformInfo("", filter_params, 1, 2)
    assert tra_info.get_image_properties(pre_transform=True) == 1
    assert tra_info.get_image_properties(pre_transform=False) == 2


def test_add_data():
    tra_info = TransformInfo("", filter_params, ImageProperties, ImageProperties)
    tra_info.add_data("a", 1)
    assert tra_info.additional_data["a"] == 1
    tra_info.add_data("b", 2)
    assert tra_info.additional_data["b"] == 2


def test_get_data():
    tra_info = TransformInfo(
        "", filter_params, ImageProperties, ImageProperties, None, {"a": 1, "b": 2}
    )
    assert tra_info.get_data("a") == 1
    assert tra_info.get_data("b") == 2


def test_get_transform_1():
    transform_params = (10, 20, 0)
    transform = sitk.TranslationTransform(3, transform_params)
    inverse_transform = transform.GetInverse()
    tra_info = TransformInfo(
        "", filter_params, ImageProperties, ImageProperties, None, None, transform
    )
    assert tra_info.get_transform(inverse=False) == transform
    assert (
        tra_info.get_transform(inverse=True).GetParameters()
        == inverse_transform.GetParameters()
    )


def test_get_transform_2():
    tra_info = TransformInfo("", filter_params, image_properties_1, image_properties_1)
    assert isinstance(tra_info.get_transform(inverse=False), sitk.Transform)
    assert isinstance(tra_info.get_transform(inverse=True), sitk.Transform)
    assert tra_info.get_transform(inverse=False).GetParameters() == (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    assert tra_info.get_transform(inverse=True).GetParameters() == (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )


def test_get_transform_3():
    tra_info = TransformInfo("", filter_params, image_properties_1, image_properties_2)
    assert isinstance(tra_info.get_transform(inverse=False), sitk.Transform)
    assert isinstance(tra_info.get_transform(inverse=True), sitk.Transform)
    assert tra_info.get_transform(inverse=False).GetParameters() == (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
    )
    assert tra_info.get_transform(inverse=True).GetParameters() == (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        -1.0,
        0.0,
        0.0,
    )
