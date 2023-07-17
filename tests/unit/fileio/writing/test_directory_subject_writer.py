import os
from zipfile import ZipFile

import pydicom
import pytest

from pyradise.fileio import DirectorySubjectWriter


def helper_get_ds(img_series_dcm) -> pydicom.Dataset:
    series = os.listdir(img_series_dcm)
    dicom_path = os.path.join(img_series_dcm, series[0])
    with open(dicom_path, "rb") as file:
        ds = pydicom.dcmread(file)
    return ds


def test__init__():
    subject_writer = DirectorySubjectWriter()
    assert subject_writer.as_zip is False


def test__write_to_zip_1(img_series_dcm, empty_folder):
    ds = helper_get_ds(img_series_dcm)
    subject_writer = DirectorySubjectWriter()
    subject_writer._write_to_zip((("dataset", ds),), img_series_dcm, empty_folder, "folder_name")
    assert os.path.exists(os.path.join(empty_folder, "folder_name.zip"))


def test__write_to_zip_2(img_series_dcm, empty_folder):
    ds = helper_get_ds(img_series_dcm)
    subject_writer = DirectorySubjectWriter()
    subject_writer._write_to_zip((("dataset", ds),), img_series_dcm, empty_folder, "folder_name.zip")
    assert os.path.exists(os.path.join(empty_folder, "folder_name.zip"))


def test__write_to_zip_3(img_series_dcm, empty_folder):
    ds = helper_get_ds(img_series_dcm)
    subject_writer = DirectorySubjectWriter()
    os.mkdir(os.path.join(empty_folder, "folder_name.zip"))
    with pytest.raises(Exception):
        subject_writer._write_to_zip((("dataset", ds),), img_series_dcm, empty_folder, "folder_name.zip")


def test__write_to_zip_4(img_series_dcm, empty_folder):
    count_dcm = len(os.listdir(img_series_dcm))
    ds = helper_get_ds(img_series_dcm)
    subject_writer = DirectorySubjectWriter()
    subject_writer._write_to_zip((("dataset", ds),), img_series_dcm, empty_folder, "folder_name")
    assert os.path.exists(os.path.join(empty_folder, "folder_name.zip"))

    with ZipFile(os.path.join(empty_folder, "folder_name.zip"), "r") as zip_ref:
        zip_ref.extractall(empty_folder)

    assert count_dcm == len([x for x in os.listdir(empty_folder) if x.endswith(".dcm")]) - 1


def test__write_to_folder_1(img_series_dcm, empty_folder):
    count_dcm = len(os.listdir(img_series_dcm))
    ds = helper_get_ds(img_series_dcm)
    subject_writer = DirectorySubjectWriter()
    subject_writer._write_to_folder((("dataset", ds),), img_series_dcm, empty_folder, None)
    assert count_dcm == len([x for x in os.listdir(empty_folder) if x.endswith(".dcm")]) - 1


def test__write_to_folder_2(img_series_dcm, empty_folder):
    count_dcm = len(os.listdir(img_series_dcm))
    ds = helper_get_ds(img_series_dcm)
    subject_writer = DirectorySubjectWriter()
    subject_writer._write_to_folder((("dataset", ds),), img_series_dcm, empty_folder, "folder_name")
    assert os.path.exists(os.path.join(empty_folder, "folder_name"))
    assert (
        count_dcm == len([x for x in os.listdir(os.path.join(empty_folder, "folder_name")) if x.endswith(".dcm")]) - 1
    )


def test__write_to_folder_3(img_series_dcm, empty_folder):
    ds = helper_get_ds(img_series_dcm)
    subject_writer = DirectorySubjectWriter()
    subject_writer._write_to_folder((("dataset", ds),), None, empty_folder, "folder_name")
    assert "dataset.dcm" in os.listdir(os.path.join(empty_folder, "folder_name"))


def test__write_to_folder_4(img_series_dcm, empty_folder):
    subject_writer = DirectorySubjectWriter()
    subject_writer._write_to_folder(None, None, empty_folder, "folder_name")
    assert "folder_name" in os.listdir(empty_folder)
    assert len(os.listdir(os.path.join(empty_folder, "folder_name"))) == 0


def test_write_1():
    subject_writer = DirectorySubjectWriter()
    with pytest.raises(Exception):
        subject_writer.write(None, None, None, "fantasy_path")


def test_write_2(empty_folder):
    subject_writer = DirectorySubjectWriter()
    with open(os.path.join(empty_folder, "test.txt"), "w") as file:
        file.write("")
    with pytest.raises(NotADirectoryError):
        subject_writer.write(None, None, None, os.path.join(empty_folder, "test.txt"))


def test_write_3(empty_folder):
    subject_writer = DirectorySubjectWriter()
    with pytest.raises(Exception):
        subject_writer.write(None, "fantasy_path", None, empty_folder)


def test_write_4(empty_folder):
    subject_writer = DirectorySubjectWriter()
    with open(os.path.join(empty_folder, "test.txt"), "w") as file:
        file.write("")
    with pytest.raises(NotADirectoryError):
        subject_writer.write(None, os.path.join(empty_folder, "test.txt"), None, empty_folder)


def test_write_5(img_series_dcm, empty_folder):
    subject_writer = DirectorySubjectWriter(as_zip=False)
    subject_writer.write(None, empty_folder, "folder_name", img_series_dcm)


def test_write_6(img_series_dcm, empty_folder):
    subject_writer = DirectorySubjectWriter(as_zip=True)
    subject_writer.write(None, empty_folder, "folder_name", img_series_dcm)


def test_write_7(img_series_dcm, empty_folder):
    subject_writer = DirectorySubjectWriter(as_zip=True)
    with pytest.raises(ValueError):
        subject_writer.write(None, empty_folder, False, img_series_dcm)


def test_write_8(img_series_dcm, empty_folder):
    subject_writer = DirectorySubjectWriter(as_zip=True)
    subject_writer.write(None, empty_folder, "folder_name", img_series_dcm)
    assert os.path.exists(os.path.join(empty_folder, "folder_name.zip"))
