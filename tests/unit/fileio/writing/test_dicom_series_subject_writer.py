import os
from zipfile import ZipFile

import pydicom
import pytest

from pyradise.fileio import DicomSeriesImageInfo, DicomSeriesSubjectWriter


def helper_get_ds(img_series_dcm) -> (pydicom.Dataset, str):
    series = os.listdir(img_series_dcm)
    dicom_path = os.path.join(img_series_dcm, series[0])
    with open(dicom_path, "rb") as file:
        ds = pydicom.dcmread(file)
    return ds, dicom_path


def test__init__():
    subject_writer = DicomSeriesSubjectWriter()
    assert subject_writer.as_zip is False


def test__write_to_folder_1(img_series_dcm, empty_folder):
    ds, dicom_path = helper_get_ds(img_series_dcm)
    info = DicomSeriesImageInfo((dicom_path,))
    subject_writer = DicomSeriesSubjectWriter()
    subject_writer._write_to_folder((info,), (("dataset", ds),), empty_folder, "folder_name")
    assert "dcm" not in os.listdir(empty_folder)


def test__write_to_folder_2(img_series_dcm, empty_folder):
    ds, dicom_path = helper_get_ds(img_series_dcm)
    info = DicomSeriesImageInfo((dicom_path,))
    subject_writer = DicomSeriesSubjectWriter()
    subject_writer._write_to_folder((info,), (("dataset", ds),), empty_folder, "folder_name")
    assert os.path.exists(os.path.join(empty_folder, "folder_name"))
    assert "dcm" not in os.listdir(os.path.join(empty_folder, "folder_name"))


def test__write_to_zip_1(img_series_dcm, empty_folder):
    ds, dicom_path = helper_get_ds(img_series_dcm)
    info = DicomSeriesImageInfo((dicom_path,))
    subject_writer = DicomSeriesSubjectWriter()
    subject_writer._write_to_zip((info,), (("dataset", ds),), empty_folder, "folder_name")
    assert os.path.exists(os.path.join(empty_folder, "folder_name.zip"))


def test__write_to_zip_2(img_series_dcm, empty_folder):
    ds, dicom_path = helper_get_ds(img_series_dcm)
    info = DicomSeriesImageInfo((dicom_path,))
    subject_writer = DicomSeriesSubjectWriter()
    subject_writer._write_to_zip((info,), (("dataset", ds),), empty_folder, "folder_name.zip")
    assert os.path.exists(os.path.join(empty_folder, "folder_name.zip"))


def test__write_to_zip_3(img_series_dcm, empty_folder):
    ds, dicom_path = helper_get_ds(img_series_dcm)
    info = DicomSeriesImageInfo((dicom_path,))
    subject_writer = DicomSeriesSubjectWriter()
    os.makedirs(os.path.join(empty_folder, "folder_name.zip"))
    with pytest.raises(Exception):
        subject_writer._write_to_zip((info,), (("dataset", ds),), empty_folder, "folder_name.zip")


def test__write_to_zip_4(img_series_dcm, empty_folder):
    ds, dicom_path = helper_get_ds(img_series_dcm)
    info = DicomSeriesImageInfo((dicom_path,))
    subject_writer = DicomSeriesSubjectWriter()
    subject_writer._write_to_zip((info,), (("dataset", ds),), empty_folder, "folder_name")

    with ZipFile(os.path.join(empty_folder, "folder_name.zip"), "r") as zip_ref:
        zip_ref.extractall(empty_folder)

    with open(dicom_path, "rb") as file:
        ds_original = pydicom.dcmread(file)

    with open(os.path.join(empty_folder, "img_serie_14.dcm"), "rb") as f:
        ds_extracted = pydicom.dcmread(f)

    assert ds_original == ds_extracted


def test_write_1():
    subject_writer = DicomSeriesSubjectWriter()
    with pytest.raises(Exception):
        subject_writer.write((("dataset", "ds"),), "fantasy_path", "folder_name", ("info",))


def test_write_2(empty_folder):
    subject_writer = DicomSeriesSubjectWriter()
    file_path = os.path.join(empty_folder, "fantasy_path.file")
    with open(file_path, "w+") as fp:
        fp.write("This is first line")
    with pytest.raises(NotADirectoryError):
        subject_writer.write((("dataset", "ds"),), file_path, "folder_name", ("info",))


def test_write_3(img_series_dcm, empty_folder):
    ds, dicom_path = helper_get_ds(img_series_dcm)
    subject_writer = DicomSeriesSubjectWriter(as_zip=False)
    subject_writer.write((("dataset", ds),), empty_folder, "folder_name", None)
    assert os.path.exists(os.path.join(empty_folder, "folder_name"))
    assert "dcm" not in os.listdir(os.path.join(empty_folder, "folder_name"))


def test_write_4(img_series_dcm, empty_folder):
    ds, dicom_path = helper_get_ds(img_series_dcm)
    subject_writer = DicomSeriesSubjectWriter(as_zip=False)
    info = DicomSeriesImageInfo((dicom_path,))
    subject_writer.write((("dataset", ds),), empty_folder, "folder_name", (info,))
    assert os.path.exists(os.path.join(empty_folder, "folder_name"))
    assert "dcm" not in os.listdir(os.path.join(empty_folder, "folder_name"))


def test_write_5(img_series_dcm, empty_folder):
    ds, dicom_path = helper_get_ds(img_series_dcm)
    subject_writer = DicomSeriesSubjectWriter(as_zip=True)
    subject_writer.write((("dataset", ds),), empty_folder, "folder_name", None)
    assert os.path.exists(os.path.join(empty_folder, "folder_name.zip"))


def test_write_6(img_series_dcm, empty_folder):
    ds, dicom_path = helper_get_ds(img_series_dcm)
    subject_writer = DicomSeriesSubjectWriter(as_zip=True)
    info = DicomSeriesImageInfo((dicom_path,))
    subject_writer.write((("dataset", ds),), empty_folder, "folder_name", (info,))
    assert os.path.exists(os.path.join(empty_folder, "folder_name.zip"))
