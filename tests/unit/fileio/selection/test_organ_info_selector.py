import pytest

from pyradise.data import Organ
from pyradise.fileio import OrganInfoSelector


def test__init__1():
    ois = OrganInfoSelector(("organ",))
    assert ois.keep[0].name == "organ"


def test__init__2():
    ois = OrganInfoSelector((Organ("organ_1"), Organ("organ_2")))
    assert ois.keep[0].name == "organ_1"
    assert ois.keep[1].name == "organ_2"


def test__init__3():
    with pytest.raises(ValueError):
        OrganInfoSelector()
