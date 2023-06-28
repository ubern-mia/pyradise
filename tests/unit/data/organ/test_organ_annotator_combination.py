from pyradise.data import Annotator, Organ, OrganAnnotatorCombination


def test__init__1():
    o = Organ("organ_name", 1)
    a = Annotator("annotator_name", "abbreviation")
    oa = OrganAnnotatorCombination(o, a)
    assert oa.organ == o
    assert isinstance(oa.organ, Organ)
    assert oa.annotator == a
    assert isinstance(oa.annotator, Annotator)


def test__init__2():
    oa = OrganAnnotatorCombination("organ", "annotator")
    assert oa.organ == Organ("organ")
    assert isinstance(oa.organ, Organ)
    assert oa.annotator == Annotator("annotator")
    assert isinstance(oa.annotator, Annotator)


def test__str():
    oa = OrganAnnotatorCombination("organ", "annotator")
    assert str(oa) == "organ_annotator"
    assert isinstance(oa.__str__(), str)


def test__eq__():
    oa = OrganAnnotatorCombination("organ", "annotator")
    ob = OrganAnnotatorCombination("organ", "annotator")
    assert oa.__eq__(ob)
    assert isinstance(oa.__eq__(ob), bool)


def test__hash__():
    oa = OrganAnnotatorCombination("organ", "annotator")
    ob = OrganAnnotatorCombination("organ", "annotator")
    assert isinstance(oa.__hash__(), int)
    assert oa.__hash__() == ob.__hash__()
