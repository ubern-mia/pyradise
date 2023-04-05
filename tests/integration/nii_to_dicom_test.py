import os
from typing import Optional, Tuple

import pyradise.fileio as ps_fio
from pyradise.data import Modality


# pylint: disable=duplicate-code
class ModalityExtractorExample(ps_fio.ModalityExtractor):
    """A modality extractor that always returns the same modality."""

    def __init__(self, modalities: Tuple[str, ...], identifier: str = "img_", return_default: bool = False) -> None:
        super().__init__(return_default)

        self.modalities = modalities
        self.identifier = identifier

    def extract_from_dicom(self, path: str) -> Optional[Modality]:
        return None

    def extract_from_path(self, path: str) -> Optional[Modality]:
        file_name = os.path.basename(path)

        if not file_name.startswith(self.identifier):
            return None

        for modality in self.modalities:
            if modality in file_name:
                return Modality(modality)
        return None


def test_dataset_nii_crawler_direct(nii_test_dataset_path: str) -> None:
    # get the extractors
    modality_extractor = ModalityExtractorExample(("T1", "T2"))
    organ_extractor = ps_fio.SimpleOrganExtractor(
        ("Cochlea", "Skull", "TV", "Vol2016", "Cochlea", "AN", "T1+ 13", "Vol2015", "tv", "Vol2018")
    )
    annotator_extractor = ps_fio.SimpleAnnotatorExtractor(("NA",))

    # construct the crawler
    crawler = ps_fio.DatasetFileCrawler(
        nii_test_dataset_path, ".nii.gz", modality_extractor, organ_extractor, annotator_extractor
    )

    series_info = crawler.execute()

    # check if the number of series info is correct
    assert len(series_info) == 5

    # check the number of entries
    for series_info_entry in series_info:
        image_series_info = [entry for entry in series_info_entry if isinstance(entry, ps_fio.IntensityFileSeriesInfo)]
        assert len(image_series_info) == 2

        seg_series_info = [entry for entry in series_info_entry if isinstance(entry, ps_fio.SegmentationFileSeriesInfo)]
        assert len(seg_series_info) >= 3


def test_dataset_nii_crawler_iterative(nii_test_dataset_path: str) -> None:
    # get the extractors
    modality_extractor = ModalityExtractorExample(("T1", "T2"))
    organ_extractor = ps_fio.SimpleOrganExtractor(
        ("Cochlea", "Skull", "TV", "Vol2016", "Cochlea", "AN", "T1+ 13", "Vol2015", "tv", "Vol2018")
    )
    annotator_extractor = ps_fio.SimpleAnnotatorExtractor(("NA",))

    # construct the crawler
    crawler = ps_fio.DatasetFileCrawler(
        nii_test_dataset_path, ".nii.gz", modality_extractor, organ_extractor, annotator_extractor
    )

    # loop through the series info entries
    num_subject_series_infos = 0
    for series_info in crawler:
        num_subject_series_infos += 1

        image_series_info = [entry for entry in series_info if isinstance(entry, ps_fio.IntensityFileSeriesInfo)]
        assert len(image_series_info) == 2

        seg_series_info = [entry for entry in series_info if isinstance(entry, ps_fio.SegmentationFileSeriesInfo)]
        assert len(seg_series_info) >= 3

    assert num_subject_series_infos == 5


def test_subject_nii_crawler(nii_test_subj_path: str) -> None:
    # get the extractors
    modality_extractor = ModalityExtractorExample(("T1", "T2"))
    organ_extractor = ps_fio.SimpleOrganExtractor(("Cochlea", "Skull", "TV", "Vol2016"))
    annotator_extractor = ps_fio.SimpleAnnotatorExtractor(("NA",))

    # construct the crawler
    crawler = ps_fio.SubjectFileCrawler(
        nii_test_subj_path, "VS-SEG-001", ".nii.gz", modality_extractor, organ_extractor, annotator_extractor
    )

    # execute the crawler
    series_info = crawler.execute()

    # check if the number of series info is correct
    assert len(series_info) == 6

    # check the number of entries
    image_series_info = [entry for entry in series_info if isinstance(entry, ps_fio.IntensityFileSeriesInfo)]
    assert len(image_series_info) == 2

    seg_series_info = [entry for entry in series_info if isinstance(entry, ps_fio.SegmentationFileSeriesInfo)]
    assert len(seg_series_info) == 4
