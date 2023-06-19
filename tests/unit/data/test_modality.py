from pytest import mark

from pyradise.data.modality import Modality


def test__init__():
    m = Modality("name")
    assert m.name == "name"
    assert m.default_name == "UNKNOWN"
    assert isinstance(m.name, str)
    assert isinstance(m.default_name, str)


def test_get_default():
    m = Modality("name")
    assert m.get_default() == Modality("UNKNOWN")
    assert isinstance(m.get_default(), Modality)


@mark.parametrize("name, result", [("name", False), ("UNKNOWN", True)])
def test_is_default(name, result):
    m = Modality(name)
    assert m.is_default() is result
    assert isinstance(m.is_default(), bool)


def test_get_name():
    m = Modality("name")
    assert m.get_name() == "name"
    assert isinstance(m.get_name(), str)


def test__str__():
    m = Modality("name")
    assert str(m) == "name"
    assert isinstance(m.__str__(), str)


def test__eq__1():
    m = Modality("name")
    n = Modality("name")
    assert m == n
    assert isinstance(m.__eq__(n), bool)


def test__eq__2():
    m = Modality("name")
    o = object
    assert m.__eq__(o) is False
    assert isinstance(m.__eq__(o), bool)


def test_not__eq__():
    m = Modality("name")
    n = Modality("test")
    assert m != n
    assert isinstance(m.__eq__(n), bool)


def test__hash__():
    m = Modality("name")
    assert hash(m) == hash(m.name)
    assert isinstance(hash(m), int)
