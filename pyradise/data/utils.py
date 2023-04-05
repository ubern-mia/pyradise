from typing import Optional, Sequence, Tuple, Union

from .annotator import Annotator
from .modality import Modality
from .organ import Organ, OrganAnnotatorCombination

__all__ = [
    "str_to_modality",
    "seq_to_modalities",
    "str_to_organ",
    "seq_to_organs",
    "str_to_annotator",
    "seq_to_annotators",
    "str_to_organ_annotator_combination",
    "seq_to_organ_annotator_combinations",
]


def str_to_modality(text: Union[str, Modality]) -> Modality:
    """Converts a string to a :class:`~pyradise.data.modality.Modality` instance.

    Args:
        text (Union[str, Modality]): A string or a :class:`~pyradise.data.modality.Modality` instance.

    Returns:
        Modality: A :class:`~pyradise.data.modality.Modality` instance.
    """
    if isinstance(text, Modality):
        return text

    return Modality(text)


def seq_to_modalities(seq: Sequence[Union[str, Modality]]) -> Tuple[Modality, ...]:
    """Converts a sequence of strings to a tuple of :class:`~pyradise.data.modality.Modality` instances.

    Args:
        seq (Sequence[Union[str, Modality]]): A sequence of strings or :class:`~pyradise.data.modality.Modality`
         instances.

    Returns:
        Tuple[Modality, ...]: A tuple of :class:`~pyradise.data.modality.Modality` instances.
    """
    return tuple(str_to_modality(text) for text in seq)


def str_to_organ(name: Union[str, Organ]) -> Organ:
    """Converts a string to a :class:`~pyradise.data.organ.Organ` instance.

    Args:
        name (Union[str, Organ]): A string or a :class:`~pyradise.data.organ.Organ` instance.

    Returns:
        Organ: A :class:`~pyradise.data.organ.Organ` instance.
    """
    if isinstance(name, Organ):
        return name

    return Organ(name)


def seq_to_organs(seq: Sequence[Union[str, Organ]]) -> Tuple[Organ, ...]:
    """Converts a sequence of strings to a tuple of :class:`~pyradise.data.organ.Organ` instances.

    Args:
        seq (Sequence[Union[str, Organ]]): A sequence of strings or :class:`~pyradise.data.organ.Organ` instances.

    Returns:
        Tuple[Organ, ...]: A tuple of :class:`~pyradise.data.organ.Organ` instances.
    """
    return tuple(str_to_organ(text) for text in seq)


def str_to_annotator(name: Union[str, Annotator]) -> Annotator:
    """Converts a string to a :class:`~pyradise.data.annotator.Annotator` instance.

    Args:
        name (Union[str, Annotator]): A string or a :class:`~pyradise.data.annotator.Annotator` instance.

    Returns:
        Annotator: A :class:`~pyradise.data.annotator.Annotator` instance.
    """
    if isinstance(name, Annotator):
        return name

    return Annotator(name)


def seq_to_annotators(seq: Sequence[Union[str, Annotator]]) -> Tuple[Annotator, ...]:
    """Converts a sequence of strings to a tuple of :class:`~pyradise.data.annotator.Annotator` instances.

    Args:
        seq (Sequence[Union[str, Annotator]]): A sequence of strings or :class:`~pyradise.data.annotator.Annotator`
         instances.

    Returns:
        Tuple[Annotator, ...]: A tuple of :class:`~pyradise.data.annotator.Annotator` instances.
    """
    return tuple(str_to_annotator(text) for text in seq)


def str_to_organ_annotator_combination(
    data_or_organ_name: Union[str, Tuple[str, str], OrganAnnotatorCombination], annotator_name: Optional[str] = None
) -> OrganAnnotatorCombination:
    """Converts a string to a :class:`~pyradise.data.organ.OrganAnnotatorCombination` instance.

    Args:
        data_or_organ_name (Union[str, Tuple[str, str], OrganAnnotatorCombination]): A string for the organ name, a
         tuple of two strings for the organ name and the annotator name, or a
         :class:`~pyradise.data.organ.OrganAnnotatorCombination` instance.
        annotator_name (Optional[str], optional): A string for the annotator's name (default: None).

    Returns:
        OrganAnnotatorCombination: A :class:`~pyradise.data.organ.OrganAnnotatorCombination` instance.
    """
    if isinstance(data_or_organ_name, OrganAnnotatorCombination):
        return data_or_organ_name

    elif isinstance(data_or_organ_name, tuple):
        return OrganAnnotatorCombination(*data_or_organ_name)

    else:
        if annotator_name is None:
            raise ValueError("`annotator_name` must be provided if `data_or_organ_name` is a string.")
        return OrganAnnotatorCombination(data_or_organ_name, annotator_name)


def seq_to_organ_annotator_combinations(
    seq: Sequence[Union[Tuple[str, str], OrganAnnotatorCombination]],
) -> Tuple[OrganAnnotatorCombination, ...]:
    """Converts a sequence of string tuples to a tuple of :class:`~pyradise.data.organ.OrganAnnotatorCombination`
    instances.

    Args:
        seq (Sequence[Union[Tuple[str, str], OrganAnnotatorCombination]]): A sequence of tuples of two strings for the
         organ names and the annotator names or a sequence of :class:`~pyradise.data.organ.OrganAnnotatorCombination`
         instances.

    Returns:
        Tuple[OrganAnnotatorCombination, ...]: A tuple of :class:`~pyradise.data.organ.OrganAnnotatorCombination`
        instances.
    """
    return tuple(str_to_organ_annotator_combination(text) for text in seq)
