import os
from pathlib import Path

import pyradise.fileio as ps_fio


def test_dicom_dataset_crawler_direct(dicom_test_dataset_path: str) -> None:
    # construct the crawler
    crawler = ps_fio.DatasetDicomCrawler(dicom_test_dataset_path)
    series_info = crawler.execute()

    # check if the number of series info is correct
    assert len(series_info) == 5

    # check the number of entries
    for series_info_entry in series_info:
        image_series_info = [entry for entry in series_info_entry if isinstance(entry, ps_fio.DicomSeriesImageInfo)]
        assert len(image_series_info) == 2

        reg_series_info = [
            entry for entry in series_info_entry if isinstance(entry, ps_fio.DicomSeriesRegistrationInfo)
        ]
        assert len(reg_series_info) == 0

        rtss_series_info = [entry for entry in series_info_entry if isinstance(entry, ps_fio.DicomSeriesRTSSInfo)]
        assert len(rtss_series_info) == 1


def test_dicom_dataset_crawler_iterative(dicom_test_dataset_path: str) -> None:
    # construct the crawler
    crawler = ps_fio.DatasetDicomCrawler(dicom_test_dataset_path)
    num_subject_series_infos = 0

    # check the number of entries
    for series_info in crawler:
        num_subject_series_infos += 1

        image_series_info = [entry for entry in series_info if isinstance(entry, ps_fio.DicomSeriesImageInfo)]
        assert len(image_series_info) == 2

        reg_series_info = [entry for entry in series_info if isinstance(entry, ps_fio.DicomSeriesRegistrationInfo)]
        assert len(reg_series_info) == 0

        rtss_series_info = [entry for entry in series_info if isinstance(entry, ps_fio.DicomSeriesRTSSInfo)]
        assert len(rtss_series_info) == 1

    assert num_subject_series_infos == 5


def test_dicom_subject_crawler(dicom_test_subj_path: str) -> None:
    # construct the crawler
    crawler = ps_fio.SubjectDicomCrawler(dicom_test_subj_path)
    series_info = crawler.execute()

    # check if the number of series info entries is correct
    assert len(series_info) == 3

    # check the number of entries
    image_series_info = [entry for entry in series_info if isinstance(entry, ps_fio.DicomSeriesImageInfo)]
    assert len(image_series_info) == 2

    reg_series_info = [entry for entry in series_info if isinstance(entry, ps_fio.DicomSeriesRegistrationInfo)]
    assert len(reg_series_info) == 0

    rtss_series_info = [entry for entry in series_info if isinstance(entry, ps_fio.DicomSeriesRTSSInfo)]
    assert len(rtss_series_info) == 1


def test_dicom_convert_to_dicom_to_nii_with_iter(dicom_test_dataset_path: str, tmp_path: Path) -> None:
    # crawl for the subjects
    crawler = ps_fio.DatasetDicomCrawler(dicom_test_dataset_path)

    # initialize a subject loader
    loader = ps_fio.SubjectLoader()

    # initialize a subject writer
    writer = ps_fio.SubjectWriter()

    # load the subject
    for series_info in crawler:
        subject = loader.load(series_info)

        # write the output
        writer.write_to_subject_folder(str(tmp_path), subject)

    # check the number of output directories
    num_output_dirs = len(os.listdir(str(tmp_path)))
    assert num_output_dirs == 5

    # check the number of output files
    num_output_files = 0
    for root, _, files in os.walk(str(tmp_path)):
        num_output_files += len(files)
    assert num_output_files == 33
