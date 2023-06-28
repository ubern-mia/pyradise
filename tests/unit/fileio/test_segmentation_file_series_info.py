import pytest

from pyradise.data import Annotator, Organ
from pyradise.fileio.series_info import SegmentationFileSeriesInfo


def test__init__1(img_file_nii):
    sfsi = SegmentationFileSeriesInfo(img_file_nii, "test_name", "organ", "annotator")
    assert isinstance(sfsi.annotator, Annotator)
    assert isinstance(sfsi.organ, Organ)


def test__init__2(img_file_nii):
    with pytest.raises(TypeError):
        SegmentationFileSeriesInfo(0, "test_name", "organ", "annotator")


def test_get_organ(img_file_nii):
    sfsi = SegmentationFileSeriesInfo(img_file_nii, "test_name", "organ", "annotator")
    assert sfsi.get_organ() == Organ("organ")


def test_set_organ(img_file_nii):
    sfsi = SegmentationFileSeriesInfo(img_file_nii, "test_name", "organ", "annotator")
    sfsi.set_organ(Organ("new_organ"))
    assert sfsi.get_organ() == Organ("new_organ")


def test_get_annotator(img_file_nii):
    sfsi = SegmentationFileSeriesInfo(img_file_nii, "test_name", "organ", "annotator")
    assert sfsi.get_annotator() == Annotator("annotator")


def test_set_annotator(img_file_nii):
    sfsi = SegmentationFileSeriesInfo(img_file_nii, "test_name", "organ", "annotator")
    sfsi.set_annotator(Annotator("new_annotator"))
    assert sfsi.get_annotator() == Annotator("new_annotator")


def test_update(img_file_nii):
    sfsi = SegmentationFileSeriesInfo(img_file_nii, "test_name", "organ", "annotator")
    assert sfsi.is_updated() is True
