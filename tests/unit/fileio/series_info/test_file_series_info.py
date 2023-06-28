from pyradise.fileio.series_info import FileSeriesInfo


class TestFileSeriesInfo(FileSeriesInfo):
    def __init__(self, path, patient_name):
        super().__init__(path, patient_name)

    def update(self) -> None:
        pass


def test__init__1(img_file_nii):
    tsi = TestFileSeriesInfo(img_file_nii, "test_name")
    assert tsi.path == (img_file_nii,)
    assert tsi.patient_name == "test_name"
    assert tsi.patient_id == "test_name"
