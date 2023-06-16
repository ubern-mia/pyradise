from pyradise.data.image import SegmentationImage
from pyradise.data import Organ, Annotator, OrganAnnotatorCombination

from tests.unit.helpers.image_helpers import get_sitk_segmentation_image


sitk_seg_1 = get_sitk_segmentation_image(1)
sitk_seg_2 = get_sitk_segmentation_image(2)


def test__init__1():
    s = SegmentationImage(sitk_seg_1, 'organ', 'annotator')
    assert isinstance(s.organ, Organ)
    assert isinstance(s.annotator, Annotator)
    assert s.organ.get_name() == 'organ'
    assert s.annotator.get_name() == 'annotator'


def test__init__2():
    s = SegmentationImage(sitk_seg_1, 'organ', None)
    assert isinstance(s.organ, Organ)
    assert s.annotator is None
    assert s.organ.get_name() == 'organ'


def test_get_organ():
    s = SegmentationImage(sitk_seg_1, 'organ', 'annotator')
    assert s.get_organ(as_str=True) == 'organ'
    assert isinstance(s.get_organ(as_str=False), Organ)
    assert s.get_organ(as_str=False).get_name() == 'organ'


def test_set_organ():
    s = SegmentationImage(sitk_seg_1, 'organ_1', 'annotator')
    s.set_organ(Organ('organ_2'))
    assert s.get_organ(as_str=True) == 'organ_2'
    assert isinstance(s.get_organ(as_str=False), Organ)
    assert s.get_organ(as_str=False).get_name() == 'organ_2'


def test_get_annotator():
    s = SegmentationImage(sitk_seg_1, 'organ', 'annotator')
    assert s.get_annotator(as_str=True) == 'annotator'
    assert isinstance(s.get_annotator(as_str=False), Annotator)
    assert s.get_annotator(as_str=False).get_name() == 'annotator'


def test_set_annotator():
    s = SegmentationImage(sitk_seg_1, 'organ', 'annotator_1')
    s.set_annotator(Annotator('annotator_2'))
    assert s.get_annotator(as_str=True) == 'annotator_2'
    assert isinstance(s.get_annotator(as_str=False), Annotator)
    assert s.get_annotator(as_str=False).get_name() == 'annotator_2'


def test_organ_annotator_combination():
    s = SegmentationImage(sitk_seg_1, 'organ', 'annotator')
    assert isinstance(s.get_organ_annotator_combination(), OrganAnnotatorCombination)


def test_copy_info_1():
    s_1 = SegmentationImage(sitk_seg_1, 'organ_1', 'annotator_1')
    s_2 = SegmentationImage(sitk_seg_2, 'organ_2', 'annotator_2')
    s_1.copy_info(source=s_2, include_transforms=True)
    assert s_1.get_spacing() == sitk_seg_2.GetSpacing()
    assert s_1.get_origin() == sitk_seg_2.GetOrigin()
    assert s_1.get_size() == sitk_seg_2.GetSize()


def test_copy_info_2():
    s_1 = SegmentationImage(sitk_seg_1, 'organ_1', 'annotator_1')
    s_2 = SegmentationImage(sitk_seg_2, 'organ_2', 'annotator_2')
    s_1.copy_info(source=s_2, include_transforms=False)
    assert s_1.get_spacing() == sitk_seg_2.GetSpacing()
    assert s_1.get_origin() == sitk_seg_2.GetOrigin()
    assert s_1.get_size() == sitk_seg_2.GetSize()


def test_is_intensity_image():
    s = SegmentationImage(sitk_seg_1, 'organ', 'annotator')
    assert s.is_intensity_image() is False


def test_is_binary():
    s = SegmentationImage(sitk_seg_1, 'organ', 'annotator')
    assert s.is_binary() is True
    assert isinstance(s.is_binary(), bool)


def test__eq__1():
    s_1 = SegmentationImage(sitk_seg_1, 'organ_1', 'annotator_1')
    s_2 = SegmentationImage(sitk_seg_1, 'organ_1', 'annotator_1')
    assert s_1.__eq__(s_2) is True


def test__eq__2():
    s_1 = SegmentationImage(sitk_seg_1, 'organ_1', 'annotator_1')
    s_2 = SegmentationImage(sitk_seg_2, 'organ_2', 'annotator_2')
    assert s_1.__eq__(s_2) is False


def test__eq__3():
    s_1 = SegmentationImage(sitk_seg_1, 'organ_1', 'annotator_1')
    assert s_1.__eq__(object) is False


def test__str__():
    s = SegmentationImage(sitk_seg_1, 'organ', 'annotator')
    assert isinstance(s.__str__(), str)
    assert s.__str__() == 'SegmentationImage: organ / annotator'
