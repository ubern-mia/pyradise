import pytest

from pyradise.fileio.series_info import DicomSeriesInfo, Tag


class TestDicomSeriesInfo(DicomSeriesInfo):
    def __init__(self, path):
        super().__init__(path)

    def update(self) -> None:
        pass


def test__init__1(img_file_dcm):
    tdsi = TestDicomSeriesInfo(img_file_dcm)
    assert tdsi.patient_id == "1234567890"
    assert tdsi.patient_name == "WAS_084_1953"
    assert tdsi.series_description == "ep2d_diff_3scan_p3_m128_TRACEW"
    assert tdsi.series_number == "2"
    assert tdsi.sop_class_uid == "1.2.840.10008.5.1.4.1.1.4.1"
    assert tdsi.dicom_modality == "MR"


def test__init__2(img_file_dcm):
    tdsi = TestDicomSeriesInfo(img_file_dcm)
    tdsi._get_dicom_base_info([Tag(0x0010, 0x0020)])
    assert tdsi.patient_id == "1234567890"


def test__init__3(img_file_dcm_no_meta):
    with pytest.raises(ValueError):
        TestDicomSeriesInfo(img_file_dcm_no_meta)
