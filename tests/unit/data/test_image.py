import pytest
from pyradise.data.image import Image
from tests.unit.helpers.image_helpers import (
    get_sitk_intensity_image,
    get_itk_intensity_image,
)
import SimpleITK as sitk
import itk
import numpy as np
from pyradise.data.taping import TransformTape, TransformInfo

sitk_img_1 = get_sitk_intensity_image(1)
sitk_img_2 = get_sitk_intensity_image(2)
itk_img_1 = get_itk_intensity_image(3)
itk_img_2 = get_itk_intensity_image(4)
data = {'a': 1, 'b': 2, 'c': 3}


class TestImage(Image):

    def __init__(self, image, data) -> None:
        super().__init__(image, data)

    def copy_info(self, source: "Image", include_transforms: bool = False) -> tuple:
        return source, include_transforms

    def is_intensity_image(self) -> bool:
        return True

    def __eq__(self, other) -> bool:
        return True


def test__init__1():
    i = TestImage(sitk_img_1, data)
    assert i.image == sitk_img_1
    assert i.data == data


def test__init__2():
    i = TestImage(itk_img_1, data)
    assert i.image == sitk_img_1
    assert i.data == data


def test__init__3():
    with pytest.raises(TypeError):
        i = TestImage(object, data)


def test__init__4():
    with pytest.raises(TypeError):
        i = TestImage(sitk_img_1, object)


def test__init__5():
    with pytest.raises(TypeError):
        i = TestImage(sitk_img_1, {1: 1, 'b': 2, 'c': 3, 'd': 4})


def test_return_image_as():
    i = TestImage(sitk_img_1, data)
    assert isinstance(i._return_image_as(sitk_img_1, as_sitk=True), sitk.Image)
    assert isinstance(i._return_image_as(sitk_img_1, as_sitk=False), itk.Image)
    assert isinstance(i._return_image_as(itk_img_1, as_sitk=True), sitk.Image)
    assert isinstance(i._return_image_as(itk_img_1, as_sitk=False), itk.Image)


@pytest.fixture
def test_add_data():
    i = TestImage(sitk_img_1, data)
    i.add_data({'d': 4})
    assert i.data == {'a': 1, 'b': 2, 'c': 3, 'd': 4}


@pytest.fixture
def test_add_data_by_key():
    i = TestImage(sitk_img_1, data)
    i.add_data_by_key('c', 4)
    assert i.data == {'a': 1, 'b': 2, 'c': 4}


def test_get_data():
    i = TestImage(sitk_img_1, data)
    assert i.get_data() == data


def test_get_data_by_key():
    i = TestImage(sitk_img_1, data)
    assert i.get_data_by_key('a') == 1
    assert i.get_data_by_key('b') == 2
    assert i.get_data_by_key('c') == 3


@pytest.fixture
def test_replace_data_1():
    i = TestImage(sitk_img_1, data)
    i.replace_data(key='c', new_data=5, add_if_missing=False)
    assert i.data == {'a': 1, 'b': 2, 'c': 5}
    i.replace_data(key='c', new_data=1, add_if_missing=True)
    assert i.data == {'a': 1, 'b': 2, 'c': 1}


@pytest.fixture
def test_replace_data_2():
    i = TestImage(sitk_img_1, data)
    i.replace_data(key='d', new_data=5, add_if_missing=True)
    assert i.data == {'a': 1, 'b': 2, 'c': 3, 'd': 5}
    assert i.replace_data(key='x', new_data=1, add_if_missing=False) is False
    assert i.data == {'a': 1, 'b': 2, 'c': 3, 'd': 5}


@pytest.fixture
def test_remove_additional_data():
    i = TestImage(sitk_img_1, data)
    i.remove_additional_data()
    assert i.data == {}


@pytest.fixture
def test_remove_additional_data_by_key():
    i = TestImage(sitk_img_1, data)
    assert i.remove_additional_data_by_key('a') is True
    assert i.data == {'b': 2, 'c': 3}


def test_cast():
    i = TestImage(sitk_img_1, data)
    assert isinstance(i.cast(image=sitk_img_1, pixel_type=1, as_sitk=True), sitk.Image)
    assert isinstance(i.cast(image=sitk_img_1, pixel_type=1, as_sitk=False), itk.Image)
    # TODO:  Add more tests for cast itk to sitk and itk to itk


def test_get_image_data():
    i = TestImage(sitk_img_1, data)
    assert isinstance(i.get_image_data(as_sitk=True), sitk.Image)
    assert isinstance(i.get_image_data(as_sitk=False), itk.Image)
    i = TestImage(itk_img_1, data)
    assert isinstance(i.get_image_data(as_sitk=True), sitk.Image)
    assert isinstance(i.get_image_data(as_sitk=False), itk.Image)


def test_set_image_data():
    i = TestImage(sitk_img_1, data)
    i.set_image_data(sitk_img_2)
    assert i.image == sitk_img_2
    i.set_image_data(itk_img_1)
    assert i.image == sitk_img_1
    assert i.set_image_data(sitk_img_1) is None


def test_get_image_data_as_np():
    sitk_img = sitk.Image(5, 10, 15, sitk.sitkInt8)
    i = TestImage(sitk_img, data)
    assert isinstance(i.get_image_data_as_np(adjust_axes=False), np.ndarray)
    assert isinstance(i.get_image_data_as_np(adjust_axes=True), np.ndarray)
    assert i.get_image_data_as_np(adjust_axes=False).shape == (15, 10, 5)
    assert i.get_image_data_as_np(adjust_axes=True).shape == (5, 10, 15)


def test_get_image_itk_type():
    i = TestImage(sitk_img_1, data)
    assert i.get_image_itk_type() == itk.Image[itk.template(itk_img_1)[1]]


def test_get_origin():
    i = TestImage(sitk_img_1, data)
    assert i.get_origin() == (0.0, 0.0, 0.0)
    assert isinstance(i.get_origin(), tuple)


def test_get_direction():
    i = TestImage(sitk_img_1, data)
    assert i.get_direction().shape == (3, 3)
    assert isinstance(i.get_direction(), np.ndarray)


def test_get_spacing():
    i = TestImage(sitk_img_1, data)
    assert i.get_spacing() == (1.0, 1.0, 1.0)
    assert isinstance(i.get_spacing(), tuple)


def test_get_size():
    i = TestImage(sitk_img_1, data)
    assert i.get_size() == (182, 218, 182)
    assert isinstance(i.get_size(), tuple)


def test_get_dimensions():
    i = TestImage(sitk_img_1, data)
    assert i.get_dimensions() == 3
    assert isinstance(i.get_dimensions(), int)


def test_get_orientation():
    i = TestImage(sitk_img_1, data)
    assert i.get_orientation() == 'LPS'
    assert isinstance(i.get_orientation(), str)


def test_get_transform_tape():
    i = TestImage(sitk_img_1, data)
    assert isinstance(i.get_transform_tape(), TransformTape)


def test_set_transform_tape():
    i = TestImage(sitk_img_1, data)
    i.set_transform_tape(TransformTape())
    assert isinstance(i.get_transform_tape(), TransformTape)
    assert i.set_transform_tape(TransformTape()) is None


def test_add_transform_info():
    i = TestImage(sitk_img_1, data)
    t = TransformInfo(
        name='test',
        params='',
        pre_transform_image_properties=sitk_img_1,
        post_transform_image_properties=sitk_img_2,
    )
    i.add_transform_info(t)
    assert isinstance(i.get_transform_tape(), TransformTape)
    assert i.add_transform_info(t) is None


def test_copy_info():  # not meaningful to test, only for abstract class implementation
    i_1 = TestImage(sitk_img_1, data)
    i_2 = TestImage(sitk_img_2, data)
    i_1.copy_info(i_2)
    assert i_1 == i_2


def test_is_intensity_image():   # not meaningful to test, only for abstract class implementation
    i = TestImage(sitk_img_1, data)
    assert i.is_intensity_image() is True


def test__eq__():  # not meaningful to test, only for abstract class implementation
    i_1 = TestImage(sitk_img_1, data)
    assert i_1 == i_1
