import pytest

from pyradise.data import Modality
from pyradise.fileio import ModalityInfoSelector


def test__init__1():
    mis = ModalityInfoSelector(("modality",))
    assert mis.keep[0].name == "modality"


def test__init__2():
    mis = ModalityInfoSelector((Modality("modality_1"), Modality("modality_2")))
    assert mis.keep[0].name == "modality_1"
    assert mis.keep[1].name == "modality_2"


def test__init__3():
    with pytest.raises(ValueError):
        ModalityInfoSelector()
