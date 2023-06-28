import pytest

from pyradise.data import Modality
from pyradise.fileio.series_info import IntensityFileSeriesInfo


def test__init__1(img_file_nii):
    ifsi = IntensityFileSeriesInfo(img_file_nii, "test_name", "modality")
    assert ifsi.path == (img_file_nii,)
    assert ifsi.modality == Modality("modality")


def test__init__2(img_file_nii):
    with pytest.raises(TypeError):
        IntensityFileSeriesInfo(0, "test_name", "modality")


def test_get_modality(img_file_nii):
    ifsi = IntensityFileSeriesInfo(img_file_nii, "test_name", "modality")
    assert ifsi.get_modality() == Modality("modality")


def test_set_modality(img_file_nii):
    ifsi = IntensityFileSeriesInfo(img_file_nii, "test_name", "modality")
    ifsi.set_modality(Modality("test_modality"))
    assert ifsi.modality == Modality("test_modality")


def test_update(img_file_nii):
    ifsi = IntensityFileSeriesInfo(img_file_nii, "test_name", "modality")
    ifsi.update()
    assert ifsi.is_updated() is True
