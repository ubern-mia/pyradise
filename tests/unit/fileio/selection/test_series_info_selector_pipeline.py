from pyradise.fileio import IntensityFileSeriesInfo, SeriesInfoSelectorPipeline


def test__init__1(img_file_nii):
    info = IntensityFileSeriesInfo(img_file_nii, "test_name", "modality")
    series = SeriesInfoSelectorPipeline([info])
    assert series.selectors[0].modality.name == "modality"


def test__init__2(img_file_nii):
    info_1 = IntensityFileSeriesInfo(img_file_nii, "test_name_1", "modality_1")
    info_2 = IntensityFileSeriesInfo(img_file_nii, "test_name_2", "modality_2")
    series = SeriesInfoSelectorPipeline([info_1, info_2])
    assert series.selectors[0].modality.name == "modality_1"
    assert series.selectors[1].modality.name == "modality_2"


def test_add_selector_1(img_file_nii):
    info = IntensityFileSeriesInfo(img_file_nii, "test_name", "modality")
    series = SeriesInfoSelectorPipeline([info])
    assert len(series.selectors) == 1
    series.add_selector(info)
    assert len(series.selectors) == 2
