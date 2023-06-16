import pytest
import itk
from pyradise.data.image import ImageProperties
from tests.unit.helpers.image_helpers import (
    get_sitk_intensity_image,
    get_itk_intensity_image,
)

sitk_img_1 = get_sitk_intensity_image(1)
itk_img_1 = get_itk_intensity_image(2)


def test__init__1():
    i = ImageProperties(sitk_img_1)
    assert i._spacing == sitk_img_1.GetSpacing()
    assert i._origin == sitk_img_1.GetOrigin()
    assert i._direction == sitk_img_1.GetDirection()
    assert i._size == sitk_img_1.GetSize()


def test__init__2():
    i = ImageProperties(itk_img_1)
    assert i._spacing == itk_img_1.GetSpacing()
    assert i._origin == itk_img_1.GetOrigin()
    assert i._direction == tuple(itk.GetArrayFromMatrix(itk_img_1.GetDirection()).flatten())


def test__init__3():
    with pytest.raises(TypeError):
        i = ImageProperties(object)


def test_get_entry():
    i = ImageProperties(sitk_img_1, resize=(1, 1, 1), shape=None)
    assert i.get_entry('resize') == (1, 1, 1)


def test_set_entry_1():
    i = ImageProperties(sitk_img_1, resize=(1, 1, 1), shape=None)
    assert i.set_entry('new_resize', (2, 2, 2)) is None
    assert i.get_entry('new_resize') == (2, 2, 2)


def test_set_entry_2():
    i = ImageProperties(sitk_img_1, resize=(1, 1, 1), shape=None)
    with pytest.raises(ValueError):
        i.set_entry('resize', (2, 2))


def test_origin():
    i = ImageProperties(sitk_img_1)
    assert i.origin == sitk_img_1.GetOrigin()
    assert isinstance(i.origin, tuple)


def test_spacing():
    i = ImageProperties(sitk_img_1)
    assert i.spacing == sitk_img_1.GetSpacing()
    assert isinstance(i.spacing, tuple)


def test_direction():
    i = ImageProperties(sitk_img_1)
    assert i.direction == sitk_img_1.GetDirection()
    assert isinstance(i.direction, tuple)


def test_size():
    i = ImageProperties(sitk_img_1)
    assert i.size == sitk_img_1.GetSize()
    assert isinstance(i.size, tuple)


def test_has_equal_origin_direction():
    i_1 = ImageProperties(sitk_img_1)
    assert i_1.has_equal_origin_direction(i_1) is True
    assert isinstance(i_1.has_equal_origin_direction(i_1), bool)


def test__eq__1():
    i_1 = ImageProperties(sitk_img_1)
    assert i_1.__eq__(i_1) is True


def test__eq__2():
    i_1 = ImageProperties(sitk_img_1)
    assert i_1.__eq__(object) is False
