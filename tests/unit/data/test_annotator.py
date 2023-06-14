from pyradise.data.annotator import Annotator

from pytest import mark


@mark.parametrize('name, result', [('name', 'name'), ('test', 'test')])
def test_get_name(name, result):
    a = Annotator(name, 'None')
    assert a.get_name() == result
    assert isinstance(a.get_name(), str)


@mark.parametrize('abbreviation, result', [('name', 'name'), ('test', 'test')])
def test_get_abbreviation(abbreviation, result):
    a = Annotator('None', abbreviation)
    assert a.get_abbreviation() == result
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


@mark.parametrize('name, abbreviation, result', [('name', 'abbreviation', 'name (abbreviation)'), ('1', '2', '1 (2)')])
def test__str__(name, abbreviation, result):
    a = Annotator(name, abbreviation)
    assert str(a) == result


@mark.parametrize('name, abbreviation', [('name', 'abbreviation'), ('1', '2')])
def test__eq__(name, abbreviation):
    a = Annotator(name, abbreviation)
    b = Annotator(name, abbreviation)
    assert a == b
    assert isinstance(a, Annotator)
    assert isinstance(b, Annotator)


def test_not__eq__():
    a = Annotator('name', 'abbreviation')
    b = Annotator('1', '2')
    assert a != b
    assert isinstance(a, Annotator)
    assert isinstance(b, Annotator)
