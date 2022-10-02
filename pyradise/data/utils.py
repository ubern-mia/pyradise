from typing import (
    Union,
    Sequence,
    Tuple,
    Optional)

from .modality import Modality
from .organ import (
    Organ,
    OrganRaterCombination)
from .rater import Rater


__all__ = ['str_to_modality', 'seq_to_modalities',
           'str_to_organ', 'seq_to_organs',
           'str_to_rater', 'seq_to_raters',
           'str_to_organ_rater_combination', 'seq_to_organ_rater_combinations']


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


def str_to_rater(name: Union[str, Rater]) -> Rater:
    """Converts a string to a :class:`~pyradise.data.rater.Rater` instance.

    Args:
        name (Union[str, Rater]): A string or a :class:`~pyradise.data.rater.Rater` instance.

    Returns:
        Rater: A :class:`~pyradise.data.rater.Rater` instance.
    """
    if isinstance(name, Rater):
        return name

    return Rater(name)


def seq_to_raters(seq: Sequence[Union[str, Rater]]) -> Tuple[Rater, ...]:
    """Converts a sequence of strings to a tuple of :class:`~pyradise.data.rater.Rater` instances.

    Args:
        seq (Sequence[Union[str, Rater]]): A sequence of strings or :class:`~pyradise.data.rater.Rater` instances.

    Returns:
        Tuple[Rater, ...]: A tuple of :class:`~pyradise.data.rater.Rater` instances.
    """
    return tuple(str_to_rater(text) for text in seq)


def str_to_organ_rater_combination(data_or_organ_name: Union[str, Tuple[str, str], OrganRaterCombination],
                                   rater_name: Optional[str] = None
                                   ) -> OrganRaterCombination:
    """Converts a string to a :class:`~pyradise.data.organ.OrganRaterCombination` instance.

    Args:
        data_or_organ_name (Union[str, Tuple[str, str], OrganRaterCombination]): A string for the organ name, a tuple
         of two strings for the organ name and the rater name, or a :class:`~pyradise.data.organ.OrganRaterCombination`
         instance.
        rater_name (Optional[str], optional): A string for the rater's name (default: None).

    Returns:
        OrganRaterCombination: A :class:`~pyradise.data.organ.OrganRaterCombination` instance.
    """
    if isinstance(data_or_organ_name, OrganRaterCombination):
        return data_or_organ_name

    elif isinstance(data_or_organ_name, tuple):
        return OrganRaterCombination(*data_or_organ_name)

    else:
        if rater_name is None:
            raise ValueError('`rater_name` must be provided if `data_or_organ_name` is a string.')
        return OrganRaterCombination(data_or_organ_name, rater_name)


def seq_to_organ_rater_combinations(seq: Sequence[Union[Tuple[str, str], OrganRaterCombination]],
                                    ) -> Tuple[OrganRaterCombination, ...]:
    """Converts a sequence of string tuples to a tuple of :class:`~pyradise.data.organ.OrganRaterCombination` instances.

    Args:
        seq (Sequence[Union[Tuple[str, str], OrganRaterCombination]]): A sequence of tuples of two strings for the
         organ names and the rater names or a sequence of :class:`~pyradise.data.organ.OrganRaterCombination` instances.

    Returns:
        Tuple[OrganRaterCombination, ...]: A tuple of :class:`~pyradise.data.organ.OrganRaterCombination` instances.
    """
    return tuple(str_to_organ_rater_combination(text) for text in seq)
