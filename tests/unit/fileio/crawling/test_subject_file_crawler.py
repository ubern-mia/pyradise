# from tests.unit.helpers.image_helpers import img_series_dcm, img_file_dcm, img_file_nii
#
#
#
# from pyradise.fileio import SubjectFileCrawler, ModalityExtractor, OrganExtractor, AnnotatorExtractor
#
#
# class TestModalityExtractor(ModalityExtractor):
#
#         def __init__(self):
#             super().__init__()
#
#         def extract_from_dicom(self, subject):
#             return subject
#
#
# me = TestModalityExtractor()
#
# oe = TestOrganExtractor()
#
# ae = TestAnnotatorExtractor()
#
#
#
# def test_subject_file_crawler_1():
#     crawler = SubjectFileCrawler('path', 'name', 'ext', me, oe, ae)
#
