import itk
import numpy as np
import pytest
import SimpleITK as sitk

from pyradise.data import Image, TransformTape, TransformInfo
from tests.unit.helpers.image_helpers import get_itk_image, get_sitk_image


itk_img_1 = get_itk_image(seed=0, low=0, high=101, meta="nii")
itk_img_2 = get_itk_image(seed=1, low=0, high=101, meta="nii")
sitk_img_1 = get_sitk_image(seed=2, low=0, high=101, meta="nii")
sitk_img_2 = get_sitk_image(seed=3, low=0, high=101, meta="nii")


class NewImage(Image):
    def __init__(self, image, data) -> None:
        super().__init__(image, data)

    def copy_info(self, source: "Image", include_transforms: bool = False) -> tuple:
        pass

    def is_intensity_image(self) -> bool:
        pass

    def __eq__(self, other) -> bool:
        pass


def test__init__1():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    assert i.image == sitk_img_1
    assert i.data == data


def test__init__2():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(itk_img_1, data)
    assert i.image == sitk_img_1
    assert i.data == data


def test__init__3():
    data = {"a": 1, "b": 2, "c": 3}
    with pytest.raises(TypeError):
        i = NewImage(object, data)


def test__init__4():
    with pytest.raises(TypeError):
        i = NewImage(sitk_img_1, object)


def test__init__5():
    with pytest.raises(TypeError):
        i = NewImage(sitk_img_1, {1: 1, "b": 2, "c": 3, "d": 4})


def test_return_image_as():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    assert isinstance(i._return_image_as(sitk_img_1, as_sitk=True), sitk.Image)
    assert isinstance(i._return_image_as(sitk_img_1, as_sitk=False), itk.Image)
    assert isinstance(i._return_image_as(itk_img_1, as_sitk=True), sitk.Image)
    assert isinstance(i._return_image_as(itk_img_1, as_sitk=False), itk.Image)


def test_add_data():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    i.add_data({"d": 4})
    assert i.data == {"a": 1, "b": 2, "c": 3, "d": 4}
    assert i.add_data({"e": 5}) is None
    assert isinstance(i.data, dict)
    assert i.data == data


def test_add_data_by_key():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    i.add_data_by_key("c", 4)
    assert i.data == {"a": 1, "b": 2, "c": 4}


def test_get_data():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    assert i.get_data() == data


def test_get_data_by_key():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    assert i.get_data_by_key("a") == 1
    assert i.get_data_by_key("b") == 2
    assert i.get_data_by_key("c") == 3


def test_replace_data_1():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    i.replace_data(key="c", new_data=5, add_if_missing=False)
    assert i.data == {"a": 1, "b": 2, "c": 5}
    i.replace_data(key="c", new_data=1, add_if_missing=True)
    assert i.data == {"a": 1, "b": 2, "c": 1}


def test_replace_data_2():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    i.replace_data(key="d", new_data=5, add_if_missing=True)
    assert i.data == {"a": 1, "b": 2, "c": 3, "d": 5}
    assert i.replace_data(key="x", new_data=1, add_if_missing=False) is False
    assert i.data == {"a": 1, "b": 2, "c": 3, "d": 5}


def test_remove_additional_data():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    i.remove_additional_data()
    assert i.data == {}


def test_remove_additional_data_by_key():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    assert i.remove_additional_data_by_key("a") is True
    assert i.data == {"b": 2, "c": 3}


def test_cast():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    assert isinstance(i.cast(image=sitk_img_1, pixel_type=1, as_sitk=True), sitk.Image)
    assert isinstance(i.cast(image=sitk_img_1, pixel_type=1, as_sitk=False), itk.Image)
    assert isinstance(
        i.cast(image=itk_img_1, pixel_type=itk.UC, as_sitk=True), sitk.Image
    )
    assert isinstance(
        i.cast(image=itk_img_1, pixel_type=itk.UC, as_sitk=False), itk.Image
    )


def test_get_image_data():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    assert isinstance(i.get_image_data(as_sitk=True), sitk.Image)
    assert isinstance(i.get_image_data(as_sitk=False), itk.Image)
    i = NewImage(itk_img_1, data)
    assert isinstance(i.get_image_data(as_sitk=True), sitk.Image)
    assert isinstance(i.get_image_data(as_sitk=False), itk.Image)


def test_set_image_data():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    i.set_image_data(sitk_img_2)
    assert i.image == sitk_img_2
    i.set_image_data(itk_img_1)
    assert i.image == sitk_img_1
    assert i.set_image_data(sitk_img_1) is None


def test_get_image_data_as_np():
    data = {"a": 1, "b": 2, "c": 3}
    sitk_img = sitk.Image(5, 10, 15, sitk.sitkInt8)
    i = NewImage(sitk_img, data)
    assert isinstance(i.get_image_data_as_np(adjust_axes=False), np.ndarray)
    assert isinstance(i.get_image_data_as_np(adjust_axes=True), np.ndarray)
    assert i.get_image_data_as_np(adjust_axes=False).shape == (15, 10, 5)
    assert i.get_image_data_as_np(adjust_axes=True).shape == (5, 10, 15)


def test_get_image_itk_type():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    assert i.get_image_itk_type() == itk.Image[itk.template(itk_img_1)[1]]


def test_get_origin():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    assert i.get_origin() == (0.0, 0.0, 0.0)
    assert isinstance(i.get_origin(), tuple)


def test_get_direction():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    assert i.get_direction().shape == (3, 3)
    assert isinstance(i.get_direction(), np.ndarray)


def test_get_spacing():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    assert i.get_spacing() == (1.0, 1.0, 1.0)
    assert isinstance(i.get_spacing(), tuple)


def test_get_size():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    assert i.get_size() == (182, 218, 182)
    assert isinstance(i.get_size(), tuple)


def test_get_dimensions():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    assert i.get_dimensions() == 3
    assert isinstance(i.get_dimensions(), int)


def test_get_orientation():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    assert i.get_orientation() == "LPS"
    assert isinstance(i.get_orientation(), str)


def test_get_transform_tape():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    assert isinstance(i.get_transform_tape(), TransformTape)


def test_set_transform_tape():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    i.set_transform_tape(TransformTape())
    assert isinstance(i.get_transform_tape(), TransformTape)
    assert i.set_transform_tape(TransformTape()) is None


def test_add_transform_info():
    data = {"a": 1, "b": 2, "c": 3}
    i = NewImage(sitk_img_1, data)
    t = TransformInfo(
        name="test",
        params="",
        pre_transform_image_properties=sitk_img_1,
        post_transform_image_properties=sitk_img_2,
    )
    i.add_transform_info(t)
    assert isinstance(i.get_transform_tape(), TransformTape)
    assert i.add_transform_info(t) is None


def test_copy_info():  # not meaningful to test, only for abstract class implementation
    with pytest.raises(TypeError):
        i = Image()


def test_is_intensity_image():  # not meaningful to test, only for abstract class implementation
    with pytest.raises(TypeError):
        i = Image()


def test__eq__():  # not meaningful to test, only for abstract class implementation
    with pytest.raises(TypeError):
        i = Image()
