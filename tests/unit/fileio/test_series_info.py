import os.path

from pyradise.fileio.series_info import SeriesInfo


class TestSeriesInfo(SeriesInfo):

    def __init__(self, path):
        super().__init__(path)

    def update(self) -> None:
        pass


def test_check__init__1(img_file_nii):
    tsi = TestSeriesInfo(img_file_nii)
    assert tsi.path == (img_file_nii,)


def test_check__init__2(img_file_nii):
    tsi = TestSeriesInfo((img_file_nii, img_file_nii))
    assert tsi.path == (img_file_nii, img_file_nii)


def test_check_paths_1(img_file_nii):
    tsi = TestSeriesInfo(img_file_nii)
    tsi._check_paths(img_file_nii, should_be_dir=False)
    assert tsi.path == (img_file_nii,)


def test_check_paths_2(img_file_nii):
    tsi = TestSeriesInfo((img_file_nii, img_file_nii))
    tsi._check_paths(img_file_nii, should_be_dir=False)
    assert tsi.path == (img_file_nii, img_file_nii)


def test_check_paths_3(img_file_nii):
    tsi = TestSeriesInfo(img_file_nii)
    tsi._check_paths(os.path.dirname(img_file_nii), should_be_dir=True)


def test_check_paths_4(img_file_nii):
    tsi = TestSeriesInfo((img_file_nii, img_file_nii))
    tsi._check_paths((os.path.dirname(img_file_nii), os.path.dirname(img_file_nii)), should_be_dir=True)
    assert tsi.path == (img_file_nii, img_file_nii)


def test_get_path(img_file_nii):
    tsi = TestSeriesInfo(img_file_nii)
    assert isinstance(tsi.get_path(), tuple)
    assert tsi.get_path() == (img_file_nii,)


def test_get_patient_name(img_file_nii):
    tsi = TestSeriesInfo(img_file_nii)
    assert tsi.get_patient_name() == ""


def test_get_patient_id(img_file_nii):
    tsi = TestSeriesInfo(img_file_nii)
    assert tsi.get_patient_id() == ""


def test_is_update(img_file_nii):
    tsi = TestSeriesInfo(img_file_nii)
    assert tsi.is_updated() is False

