from pyradise.data.organ import Organ

from pytest import mark


@mark.parametrize('name', ['name', 'test'])
def test_get_name(name):
    o = Organ(name)
    assert o.get_name() == name
    assert isinstance(o.get_name(), str)


@mark.parametrize('name, index', [('name', 1), ('test', 2)])
def test_index(name, index):
    o = Organ(name, index)
    assert o.index == index
    assert isinstance(o.index, int)


@mark.parametrize('name', ['name', 'test'])
def test_set_name(name):
    o = Organ('default')
    o.set_name(name)
    assert o.get_name() == name
    assert isinstance(o.get_name(), str)


@mark.parametrize('name', ['name', 'test'])
def test__str__(name):
    o = Organ(name)
    assert str(o) == name
    assert isinstance(o.__str__(), str)


@mark.parametrize('name, index', [('name', 1), ('test', 2)])
def test__eq__(name, index):
    o = Organ(name, index)
    p = Organ(name, index)
    assert o == p
    assert isinstance(o.__eq__(p), bool)


@mark.parametrize('name, index', [('name', 1), ('test', 2)])
def test__hash__(name, index):
    o = Organ(name, index)
    assert isinstance(o.__hash__(), int)
