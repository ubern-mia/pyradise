import pyradise.fileio as ps_fio

from .utils import ModalityExtractorNifti


class TestSubjectFileCrawler:
    def test_execute(self, nii_test_subj_path: str) -> None:
        modality_extractor = ModalityExtractorNifti(("T1", "T2"))
        organ_extractor = ps_fio.SimpleOrganExtractor(("Cochlea", "Skull", "TV", "Vol2016"))
        annotator_extractor = ps_fio.SimpleAnnotatorExtractor(("NA",))

        # construct the crawler
        crawler = ps_fio.SubjectFileCrawler(
            nii_test_subj_path, "subject1", ".nii.gz", modality_extractor, organ_extractor, annotator_extractor
        )

        # crawl
        series_info = crawler.execute()

        # check the number of entries
        image_series_info = [entry for entry in series_info if isinstance(entry, ps_fio.IntensityFileSeriesInfo)]
        assert len(image_series_info) == 2

        seg_series_info = [entry for entry in series_info if isinstance(entry, ps_fio.SegmentationFileSeriesInfo)]
        assert len(seg_series_info) == 4

        # check the subject name
        for series_info_entry in series_info:
            assert series_info_entry.get_patient_name() == "subject1"

        # load the subject
        loader = ps_fio.SubjectLoader()
        subject = loader.load(series_info)

        # check the subject name
        assert subject.get_name() == "subject1"
