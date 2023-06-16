import pytest
from pyradise.data.image import Image
from tests.unit.helpers.image_helpers import (
    get_sitk_intensity_image,
    get_sitk_segmentation_image,
    get_itk_intensity_image,
)

sitk_img_1 = get_sitk_intensity_image(1)
itk_img_1 = get_itk_intensity_image(2)
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
    assert i.image == itk_img_1
    assert i.data == data


def test__init__3():
    with pytest.raises(TypeError):
        i = TestImage(object, data)

    # assert i._image is None
    # assert i._properties is None
    # assert i._modality is N

