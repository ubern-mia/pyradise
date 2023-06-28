import pytest

from pyradise.data import Organ
from pyradise.fileio.extraction import OrganExtractor, SimpleOrganExtractor


def test_organ_extractor_1(img_file_nii):
    soe = SimpleOrganExtractor(("Brain",))
    assert soe.extract(path=img_file_nii) is None


def test_organ_extractor_2(img_file_nii):
    soe = SimpleOrganExtractor(("nii",))
    assert soe.extract(path=img_file_nii) == Organ("nii")


def test_organ_extractor_3(img_file_nii):
    soe = OrganExtractor()
    with pytest.raises(NotImplementedError):
        soe.extract(img_file_nii)
