import os

from pyradise.fileio.series_info import DicomSeriesImageInfo
from pyradise.data.modality import Modality


def test__init__(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)
    assert dsii.modality == Modality("UNKNOWN")


def test_get_modality(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)
    assert dsii.get_modality() == Modality("UNKNOWN")


def test_set_modality(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)
    dsii.set_modality(Modality("CT"))
    assert dsii.get_modality() == Modality("CT")


def test_update(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)
    assert dsii._is_updated is False
    dsii.update()
    assert dsii._is_updated is True