import os

import pytest
from pydicom.dataset import Dataset

from pyradise.data import Modality
from pyradise.fileio.series_info import (
    DicomSeriesImageInfo,
    DicomSeriesRegistrationInfo,
    Tag,
)


def test__init__1(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)
    dsri = DicomSeriesRegistrationInfo(paths[0], [dsii], False)
    assert dsri._is_updated is True


def test__init__2(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)
    with pytest.raises(ValueError):
        DicomSeriesRegistrationInfo(paths, [dsii], False)


def test__init__3(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    with pytest.raises(ValueError):
        DicomSeriesRegistrationInfo(paths[0], None, False)


def test__init__4(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)
    dsri = DicomSeriesRegistrationInfo(paths[0], [dsii], False)
    dsri.dataset = None
    dsri.image_infos = [dsii]
    dsri.update()
    assert isinstance(dsri.dataset, Dataset)


def test_get_dicom_base_info(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)
    dsri = DicomSeriesRegistrationInfo(paths[0], [dsii], False)
    assert isinstance(dsri._get_dicom_base_info([Tag(0x0008, 0x1115)]), Dataset)


def test_get_referenced_series_info(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)
    dsri = DicomSeriesRegistrationInfo(paths[0], [dsii], False)

    dataset = {
        "ReferencedSeriesSequence": [{"SeriesInstanceUID": "series_instance_uid"}],
        "StudyInstanceUID": "study_instance_uid",
        "StudiesContainingOtherReferencedInstancesSequence": [
            {
                "ReferencedSeriesSequence": [
                    {"SeriesInstanceUID": "series_instance_uid"}
                ],
                "StudyInstanceUID": "study_instance_uid",
            },
        ],
    }

    assert isinstance(dsri.get_referenced_series_info(dataset), tuple)
    assert (
        dsri.get_referenced_series_info(dataset)[0].series_instance_uid
        == "series_instance_uid"
    )
    assert (
        dsri.get_referenced_series_info(dataset)[0].study_instance_uid
        == "study_instance_uid"
    )


def test_get_registration_sequence_info_1(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)
    dsri = DicomSeriesRegistrationInfo(paths[0], [dsii], False)

    dataset = {
        "RegistrationSequence": [
            {"FrameOfReferenceUID": "frame_of_reference_uid"},
            {
                "MatrixRegistrationSequence": [
                    {
                        "MatrixSequence": [
                            {
                                "FrameOfReferenceTransformationMatrixType": "RIGID",
                                "FrameOfReferenceTransformationMatrix": [
                                    1,
                                    0,
                                    0,
                                    0,
                                    0,
                                    1,
                                    0,
                                    0,
                                    0,
                                    0,
                                    1,
                                    0,
                                    0,
                                    0,
                                    0,
                                    1,
                                ],
                            },
                        ]
                    }
                ]
            },
        ]
    }
    assert (
        dsri._get_registration_sequence_info(dataset)[0].frame_of_reference_uid
        == "frame_of_reference_uid"
    )
    assert isinstance(dsri._get_registration_sequence_info(dataset), tuple)


def test_get_registration_sequence_info_2(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)
    dsri = DicomSeriesRegistrationInfo(paths[0], [dsii], False)

    dataset = {
        "RegistrationSequence": [
            {"FrameOfReferenceUID": "frame_of_reference_uid"},
            {
                "MatrixRegistrationSequence": [
                    {
                        "MatrixSequence": [
                            {
                                "FrameOfReferenceTransformationMatrixType": None,
                                "FrameOfReferenceTransformationMatrix": [
                                    1,
                                    0,
                                    0,
                                    0,
                                    0,
                                    1,
                                    0,
                                    0,
                                    0,
                                    0,
                                    1,
                                    0,
                                    0,
                                    0,
                                    0,
                                    1,
                                ],
                            },
                        ]
                    }
                ]
            },
        ]
    }
    assert (
        dsri._get_registration_sequence_info(dataset)[0].frame_of_reference_uid
        == "frame_of_reference_uid"
    )
    assert isinstance(dsri._get_registration_sequence_info(dataset), tuple)


def test_get_unique_instance_uid_entries_1(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)
    dsri = DicomSeriesRegistrationInfo(paths[0], [dsii], False)
    dataset = {"SeriesInstanceUID": "series_instance_uid"}
    assert isinstance(dsri._get_unique_series_instance_uid_entries([dataset]), tuple)
    assert (
        dsri._get_unique_series_instance_uid_entries([dataset])[0]["SeriesInstanceUID"]
        == "series_instance_uid"
    )


def test_get_unique_instance_uid_entries_2(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)
    dsri = DicomSeriesRegistrationInfo(paths[0], [dsii], False)
    assert isinstance(dsri._get_unique_series_instance_uid_entries([dsii]), tuple)
    assert dsri._get_unique_series_instance_uid_entries([dsii])[0].modality == Modality(
        "UNKNOWN"
    )


def test_get_registration_info_1(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)
    dsri = DicomSeriesRegistrationInfo(paths[0], [dsii], False)
    dataset = {}
    dsri.get_registration_infos(dataset, [dsii])
    assert isinstance(dsri.get_registration_infos(dataset, [dsii]), tuple)


def test_get_registration_info_2(img_series_dcm):
    """tbd"""
    pass


def test_set_image_infos(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)
    dsri = DicomSeriesRegistrationInfo(paths[0], [dsii], False)
    dsri.set_image_infos(dsii)
    assert dsri.image_infos == dsii


def test_get_image_infos(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)
    dsri = DicomSeriesRegistrationInfo(paths[0], [dsii], False)
    dsri.set_image_infos(dsii)
    assert dsri.get_image_infos() == dsii


def test_update_1(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)
    dsri = DicomSeriesRegistrationInfo(paths[0], [dsii], False)
    with pytest.raises(ValueError):
        dsri.update()


def test_update_2(img_series_dcm):
    paths = os.listdir(img_series_dcm)
    paths = [os.path.join(img_series_dcm, path) for path in paths]
    dsii = DicomSeriesImageInfo(paths)

    with pytest.raises(ValueError):
        dsri = DicomSeriesRegistrationInfo(paths, [dsii], False)
        dsri.update()
