from typing import (
    Optional,
    Tuple,
    Union)

import numpy as np

from .rater import Rater


__all__ = ['Organ', 'OrganFactory', 'OrganRaterCombination']


class Organ:
    """Representation of an organ which is used in combination with a :class:`SegmentationImage` to identify the
    content of the :class:`SegmentationImage`.

    Args:
        name (str): The name of the :class:`Organ`.
        index (Optional[int]): The index of the :class:`Organ` (default: None).
    """

    def __init__(self,
                 name: str,
                 index: Optional[int] = None
                 ) -> None:
        super().__init__()

        self.name: str = name
        self.index: Optional[int] = index

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return self.name == other.name


class OrganFactory:
    """An :class:`Organ` producing factory class.

    Args:
        names (Tuple[str, ...]): The names of the available :class:`Organ` s.
        indices (Optional[Tuple[int]]): The indices of the available :class:`Organ` s (default: None).
        auto_enumerate (bool): Indicates if the provided :class:`Organ` names should be automatically enumerated
         (default: True).
    """

    def __init__(self,
                 names: Tuple[str, ...],
                 indices: Optional[Tuple[int]] = None,
                 auto_enumerate: bool = True
                 ) -> None:
        super().__init__()

        if isinstance(indices, tuple):
            assert len(names) == len(indices), f'The number of provided names and indices must be equal, ' \
                                               f'but is not ({len(names)} (names) vs. {len(indices)} (indices))!'

        self.names = names

        if not indices and auto_enumerate:
            self.indices: Tuple[int] = tuple(np.arange(len(names)))

        else:
            self.indices: Optional[Tuple[int]] = indices

    def produce(self, name: str) -> Organ:
        """Produces a new :class:`Organ`.

        Args:
            name (str): The name of the new :class:`Organ`.

        Returns:
            Organ: The newly produced :class:`Organ`.
        """
        if name not in self.names:
            raise ValueError(f'The name {name} is not contained in the factory!')

        name_idx = self.names.index(name)
        if self.indices:
            index = self.indices[name_idx]
        else:
            index = None

        return Organ(name, index)


class OrganRaterCombination:
    """A class combining an :class:`Organ` with a :class:`Rater`.

    Args:
        organ (Union[Organ, str]): The :class:`Organ` or its name.
        rater (Union[Rater, str]): The :class:`Rater` or its name.
    """

    def __init__(self,
                 organ: Union[Organ, str],
                 rater: Union[Rater, str]
                 ) -> None:
        super().__init__()

        if isinstance(organ, str):
            organ = Organ(organ)

        if isinstance(rater, str):
            rater = Rater(rater)

        self.organ: Union[Organ, str] = organ
        self.rater: Union[Rater, str] = rater

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(str(self))

    @property
    def name(self) -> str:
        """Get the name of the :class:`OrganRaterCombination`.

        Returns:
            str: The combined name.
        """
        return self.rater.name + '_' + self.organ.name
