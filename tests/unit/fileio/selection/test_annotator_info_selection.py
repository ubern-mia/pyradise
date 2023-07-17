import pytest

from pyradise.data import Annotator
from pyradise.fileio import AnnotatorInfoSelector


def test__init__1():
    ais = AnnotatorInfoSelector(("annotator",))
    assert ais.keep[0].name == "annotator"


def test__init__2():
    ais = AnnotatorInfoSelector((Annotator("annotator_1"), Annotator("annotator_2")))
    assert ais.keep[0].name == "annotator_1"
    assert ais.keep[1].name == "annotator_2"


def test__init__3():
    with pytest.raises(ValueError):
        AnnotatorInfoSelector()
