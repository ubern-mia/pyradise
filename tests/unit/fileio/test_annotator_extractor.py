from pyradise.fileio.extraction import AnnotatorExtractor, SimpleAnnotatorExtractor

from pyradise.data import Annotator
import pytest


def test_annotator_extractor_1(img_file_nii):
    soe = SimpleAnnotatorExtractor(('Hans',))
    assert soe.extract(path=img_file_nii) is None


def test_annotator_extractor_2(img_file_nii):
    soe = SimpleAnnotatorExtractor(('nii',))
    assert soe.extract(path=img_file_nii) == Annotator('nii')


def test_annotator_extractor_3(img_file_nii):
    soe = AnnotatorExtractor()
    with pytest.raises(NotImplementedError):
        soe.extract(img_file_nii)
