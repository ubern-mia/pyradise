from pyradise.data.organ import Organ

from pytest import mark


def test_get_name():
    o = Organ('name')
    assert o.get_name() == 'name'
    assert isinstance(o.get_name(), str)


def test_index():
    o = Organ('name', 1)
    assert o.index == 1
    assert isinstance(o.index, int)


def test_set_name():
    o = Organ('default')
    o.set_name('name')
    assert o.get_name() == 'name'
    assert isinstance(o.get_name(), str)


def test__str__():
    o = Organ('name')
    assert str(o) == 'name'
    assert isinstance(o.__str__(), str)


def test__eq__1():
    o = Organ('name', 1)
    p = Organ('test', 2)
    assert o == p
    assert isinstance(o.__eq__(p), bool)


def test__eq__2():
    o = Organ('name', 1)
    p = object
    assert o.__eq__(p) is False
    assert isinstance(o.__eq__(p), bool)


def test__hash__():
    o = Organ('name', 1)
    assert isinstance(o.__hash__(), int)
