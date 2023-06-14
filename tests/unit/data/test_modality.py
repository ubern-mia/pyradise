
from pytest import mark

from pyradise.data.modality import Modality


@mark.parametrize('name', ['name', 'test'])
def test_get_default(name):
    m = Modality(name)
    assert m.get_default() == Modality("UNKNOWN")
    assert isinstance(m.get_default(), Modality)


@mark.parametrize('name, result', [('name', False), ('UNKNOWN', True)])
def test_is_default(name, result):
    m = Modality(name)
    assert m.is_default() is result
    assert isinstance(m.is_default(), bool)


@mark.parametrize('name', ['name'])
def test_get_name(name):
    m = Modality(name)
    assert m.get_name() == name
    assert isinstance(m.get_name(), str)


@mark.xfail()
@mark.parametrize('name, result', [('name', 'test')])
def test_get_name(name, result):
    m = Modality(name)
    assert m.get_name() != result
    assert isinstance(m.get_name(), str)


@mark.parametrize('name', ['name', 'test'])
def test__str__(name):
    m = Modality(name)
    assert str(m) == name
    assert isinstance(m.__str__(), str)


@mark.parametrize('name', ['name', 'test'])
def test__eq__(name):
    m = Modality(name)
    n = Modality(name)
    assert m == n
    assert isinstance(m.__eq__(n), bool)


def test_not__eq__():
    m = Modality('name')
    n = Modality('test')
    assert m != n
    assert isinstance(m.__eq__(n), bool)


def test__hash__():
    m = Modality('name')
    assert hash(m) == hash(m.name)
    assert isinstance(hash(m), int)
