from pyradise.fileio.extraction import ModalityExtractor, SimpleModalityExtractor

from pyradise.data import Modality


class TestModalityExtractor(ModalityExtractor):

    def __init__(self):
        super().__init__()

    def extract_from_dicom(self, path):
        return Modality('dcm')

    def extract_from_path(self, path):
        return Modality('nii')


def test_modality_extractor_1(img_file_nii):
    me = TestModalityExtractor()
    assert me.extract_from_path(path=img_file_nii) == Modality('nii')


def test_modality_extractor_2(img_file_dcm):
    me = SimpleModalityExtractor(('MRI',))
    assert me.extract_from_dicom(path=img_file_dcm) == Modality('MR')


def test_modality_extractor_3(img_file_nii):
    te = TestModalityExtractor()
    assert te.extract_from_path(path=img_file_nii) == Modality('nii')


def test_modality_extractor_4(img_file_dcm):
    me = SimpleModalityExtractor(('MRI',))
    assert me.extract_from_dicom(path=img_file_dcm) == Modality('MR')


def test_modality_extractor_5(img_file_dcm):
    te = TestModalityExtractor()
    assert te.extract(path=img_file_dcm) == Modality('dcm')


def test_modality_extractor_6(img_file_nii):
    me = SimpleModalityExtractor(('nii',))
    assert me.extract(path=img_file_nii) == Modality('nii')


def test_modality_extractor_7(img_file_dcm):
    me = SimpleModalityExtractor(('',))
    assert me._get_next_default_modality_name() == 'UnknownModality_0'
    assert me._get_next_default_modality_name() == 'UnknownModality_1'
    assert me._get_next_default_modality_name() == 'UnknownModality_2'


def test_modality_extractor_8(img_file_dcm):
    me = SimpleModalityExtractor(('MRI',))
    assert me.is_enumerated_default_modality(modality=None) is False
    assert me.is_enumerated_default_modality(modality=Modality('test')) is False
    assert me.is_enumerated_default_modality(modality=Modality('UnknownModality_0')) is True
    assert me.is_enumerated_default_modality(modality='test') is False
    assert me.is_enumerated_default_modality(modality='UnknownModality_1') is True


def test_modality_extractor_9(img_file_dcm):
    me = SimpleModalityExtractor(('MRI',))
    assert me.extract(img_file_dcm) == Modality('MR')


def test_modality_extractor_10(png_file):
    te = SimpleModalityExtractor(('nii',))
    assert te.extract(png_file) is None


def test_modality_extractor_11(png_file):
    te = SimpleModalityExtractor(('nii',), return_default=True)
    assert te.extract(png_file) == Modality('UnknownModality_0')


def test_modality_extractor_12(img_file_dcm_no_meta):
    te = SimpleModalityExtractor(('nii',), return_default=True)
    assert te.extract_from_dicom(img_file_dcm_no_meta) is None
