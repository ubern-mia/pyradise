from pyradise.data.annotator import Annotator

from pytest import mark
import pytest


def test__init__1():
    a = Annotator('name', 'abbreviation')
    assert a.name == 'name'
    assert a.abbreviation == 'abbreviation'
    assert isinstance(a.name, str)
    assert isinstance(a.abbreviation, str)


def test__init__2():
    with pytest.raises(ValueError):
        a = Annotator('', 'abbreviation')


def test_get_name():
    a = Annotator('name', 'abbreviation')
    assert a.get_name() == 'name'
    assert isinstance(a.get_name(), str)


def test_get_abbreviation():
    a = Annotator('name', 'abbreviation')
    assert a.get_abbreviation() == 'abbreviation'
    assert isinstance(a.get_abbreviation(), str)


def test_get_default():
    a = Annotator(Annotator.default_annotator_name, Annotator.default_annotator_abbreviation)
    assert a.get_default().name == Annotator.default_annotator_name
    assert isinstance(a.get_default().name, str)
    assert a.get_default().abbreviation == Annotator.default_annotator_abbreviation
    assert isinstance(a.get_default().abbreviation, str)


def test_is_default():
    assert Annotator.default_annotator_name == 'NA'
    assert isinstance(Annotator.default_annotator_name, str)
    assert Annotator.default_annotator_abbreviation == 'NA'
    assert isinstance(Annotator.default_annotator_abbreviation, str)


@mark.parametrize('abbreviation, result', [('[<>:/\\|?*"]|[\0-\31]', '[\\][-]'), ('test', 'test'), ('name>', 'name')])
def test_remove_illegal_characters(abbreviation, result):
    a = Annotator('None', abbreviation)
    legal_string = a._remove_illegal_characters(abbreviation)
    assert legal_string == result
    assert isinstance(legal_string, str)


def test__str__():
    a = Annotator('name', 'abbreviation')
    assert str(a) == 'name abbreviation'


def test__eq__1():
    a = Annotator('name', 'abbreviation')
    b = Annotator('name', 'abbreviation')
    assert a == b
    assert isinstance(a, Annotator)
    assert isinstance(b, Annotator)


def test__eq__2():
    a = Annotator('name', 'abbreviation')
    o = object()
    assert a.__eq__(o) is False
    assert isinstance(a.__eq__(o), bool)


def test_not__eq__():
    a = Annotator('name', 'abbreviation')
    b = Annotator('1', '2')
    assert a != b
    assert isinstance(a, Annotator)
    assert isinstance(b, Annotator)

