from pyradise.data.utils import (
    str_to_modality,
    seq_to_modalities,
    str_to_organ,
    seq_to_organs,
    str_to_annotator,
    seq_to_annotators,
    str_to_organ_annotator_combination,
    seq_to_organ_annotator_combinations,
)

from pyradise.data.modality import Modality
from pyradise.data.organ import Organ
from pyradise.data.annotator import Annotator
from pyradise.data.organ import OrganAnnotatorCombination


def test_str_to_modality_1():
    m = Modality("modality")
    modality = str_to_modality(m)
    assert m.name == "modality"
    assert isinstance(modality, Modality)


def test_str_to_modality_2():
    modality = str_to_modality("modality")
    assert modality.name == "modality"
    assert isinstance(modality, Modality)


def test_seq_to_modalities_1():
    modality_1 = Modality("modality_1")
    modality_2 = Modality("modality_2")
    modalities = seq_to_modalities([modality_1, modality_2])
    assert len(modalities) == 2
    assert modalities[0].name == "modality_1"
    assert isinstance(modalities[0], Modality)
    assert modalities[1].name == "modality_2"
    assert isinstance(modalities[1], Modality)


def test_seq_to_modalities_2():
    modalities = seq_to_modalities(["modality_1", "modality_2"])
    assert len(modalities) == 2
    assert modalities[0].name == "modality_1"
    assert isinstance(modalities[0], Modality)
    assert modalities[1].name == "modality_2"
    assert isinstance(modalities[1], Modality)


def test_str_to_organ_1():
    o = Organ("organ")
    organ = str_to_organ(o)
    assert o.name == "organ"
    assert isinstance(organ, Organ)


def test_str_to_organ_2():
    organ = str_to_organ("organ")
    assert organ.name == "organ"
    assert isinstance(organ, Organ)


def test_seq_to_organs_1():
    organ_1 = Organ("organ_1")
    organ_2 = Organ("organ_2")
    organs = seq_to_organs([organ_1, organ_2])
    assert len(organs) == 2
    assert organs[0].name == "organ_1"
    assert isinstance(organs[0], Organ)
    assert organs[1].name == "organ_2"
    assert isinstance(organs[1], Organ)


def test_seq_to_organs_2():
    organs = seq_to_organs(["organ_1", "organ_2"])
    assert len(organs) == 2
    assert organs[0].name == "organ_1"
    assert isinstance(organs[0], Organ)
    assert organs[1].name == "organ_2"
    assert isinstance(organs[1], Organ)


def test_str_to_annotator_1():
    a = Annotator("annotator")
    annotator = str_to_annotator(a)
    assert a.name == "annotator"
    assert isinstance(annotator, Annotator)


def test_str_to_annotator_2():
    annotator = str_to_annotator("annotator")
    assert annotator.name == "annotator"
    assert isinstance(annotator, Annotator)


def test_seq_to_annotator_1():
    annotator_1 = Annotator("annotator_1")
    annotator_2 = Annotator("annotator_2")
    annotators = seq_to_annotators([annotator_1, annotator_2])
    assert len(annotators) == 2
    assert annotators[0].name == "annotator_1"
    assert isinstance(annotators[0], Annotator)
    assert annotators[1].name == "annotator_2"
    assert isinstance(annotators[1], Annotator)


def test_seq_to_annotator_2():
    annotator = seq_to_annotators(["annotator_1", "annotator_2"])
    assert len(annotator) == 2
    assert annotator[0].name == "annotator_1"
    assert isinstance(annotator[0], Annotator)
    assert annotator[1].name == "annotator_2"
    assert isinstance(annotator[1], Annotator)


def test_str_to_organ_annotator_combination_1():
    oa = OrganAnnotatorCombination("organ", "annotator")
    combination = str_to_organ_annotator_combination(oa)
    assert oa.organ.name == "organ"
    assert oa.annotator.name == "annotator"
    assert isinstance(combination, OrganAnnotatorCombination)


def test_str_to_organ_annotator_combination_2():
    organ_annotator = str_to_organ_annotator_combination(("organ", "annotator"))
    assert organ_annotator.organ.name == "organ"
    assert organ_annotator.annotator.name == "annotator"
    assert isinstance(organ_annotator, OrganAnnotatorCombination)


def test_seq_to_organ_annotator_combination_1():
    organ_annotator_1 = OrganAnnotatorCombination("organ_1", "annotator_1")
    organ_annotator_2 = OrganAnnotatorCombination("organ_2", "annotator_2")
    organ_annotator_combinations = seq_to_organ_annotator_combinations(
        [organ_annotator_1, organ_annotator_2]
    )
    assert len(organ_annotator_combinations) == 2
    assert organ_annotator_combinations[0].organ.name == "organ_1"
    assert organ_annotator_combinations[0].annotator.name == "annotator_1"
    assert isinstance(organ_annotator_combinations[0], OrganAnnotatorCombination)
    assert organ_annotator_combinations[1].organ.name == "organ_2"
    assert organ_annotator_combinations[1].annotator.name == "annotator_2"
    assert isinstance(organ_annotator_combinations[1], OrganAnnotatorCombination)


def test_seq_to_organ_annotator_combination_2():
    organ_annotator = seq_to_organ_annotator_combinations(
        [("organ_1", "annotator_1"), ("organ_2", "annotator_2")]
    )
    assert len(organ_annotator) == 2
    assert organ_annotator[0].organ.name == "organ_1"
    assert organ_annotator[0].annotator.name == "annotator_1"
    assert isinstance(organ_annotator[0], OrganAnnotatorCombination)
    assert organ_annotator[1].organ.name == "organ_2"
    assert organ_annotator[1].annotator.name == "annotator_2"
    assert isinstance(organ_annotator[1], OrganAnnotatorCombination)
