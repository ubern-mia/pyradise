import pytest

from pyradise.data.image import IntensityImage
from pyradise.data.modality import Modality
from tests.conftest import get_itk_image, get_sitk_image

itk_img_1 = get_itk_image(seed=0, low=0, high=101, meta="nii")
sitk_img_1 = get_sitk_image(seed=1, low=0, high=101, meta="nii")
sitk_img_2 = get_sitk_image(seed=2, low=0, high=101, meta="nii")


def test__init__1():
    i = IntensityImage(sitk_img_1, "modality")
    assert isinstance(i.modality, Modality)
    assert i.modality.get_name() == "modality"


def test__init__2():
    i = IntensityImage(itk_img_1, "modality")
    assert isinstance(i.modality, Modality)
    assert i.modality.get_name() == "modality"


def test_get_modality():
    i = IntensityImage(sitk_img_1, "modality")
    assert isinstance(i.get_modality(), Modality)
    assert i.get_modality(as_str=True) == "modality"
    assert i.get_modality(as_str=False).get_name() == "modality"


def test_set_modality():
    i = IntensityImage(sitk_img_1, "modality_1")
    i.set_modality(Modality("modality_2"))
    assert isinstance(i.get_modality(), Modality)
    assert i.get_modality(as_str=True) == "modality_2"
    assert i.get_modality(as_str=False).get_name() == "modality_2"


def test_copy_info_1():
    i_1 = IntensityImage(sitk_img_1, "modality_1")
    i_2 = IntensityImage(sitk_img_2, "modality_2")
    assert i_1.copy_info(i_2, include_transforms=False) is None
    assert isinstance(i_2, IntensityImage)
    assert i_2.get_modality(as_str=True) == "modality_2"
    assert i_2.get_modality(as_str=False).get_name() == "modality_2"


def test_copy_info_2():
    i_1 = IntensityImage(sitk_img_1, "modality_1")
    i_2 = IntensityImage(sitk_img_2, "modality_2")
    assert i_1.copy_info(i_2, include_transforms=True) is None
    assert isinstance(i_2, IntensityImage)
    assert i_2.get_modality(as_str=True) == "modality_2"
    assert i_2.get_modality(as_str=False).get_name() == "modality_2"


def test_copy_info_3():
    with pytest.raises(TypeError):
        i_1 = IntensityImage(sitk_img_1, "modality_1")
        i_1.copy_info(object, include_transforms=False)


def test_is_intensity_image():
    i = IntensityImage(sitk_img_1, "modality")
    assert i.is_intensity_image() is True


def test__eq__1():
    i_1 = IntensityImage(sitk_img_1, "modality_1")
    i_2 = IntensityImage(sitk_img_2, "modality_2")
    assert i_1.__eq__(i_2) is False


def test__eq__2():
    i_1 = IntensityImage(sitk_img_1, "modality_1")
    assert i_1.__eq__(object) is False
    assert isinstance(i_1.__eq__(object), bool)


def test__str__():
    i = IntensityImage(sitk_img_1, "modality")
    assert isinstance(i.__str__(), str)
    assert i.__str__() == "Intensity image: modality"
