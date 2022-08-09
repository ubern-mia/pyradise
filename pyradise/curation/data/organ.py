from typing import (
    Optional,
    Tuple,
    Union)

import numpy as np

from .rater import Rater


class Organ:
    """A class representing an organ."""

    def __init__(self,
                 name: str,
                 index: Optional[int] = None
                 ) -> None:
        """Constructs an organ.

        Args:
            name (str): The name of the organ.
            index (Optional[int]): The index of the organ (default=None).
        """
        super().__init__()

        self.name = name
        self.index = index

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return self.name == other.name


class OrganFactory:
    """A factory class producing organs."""

    def __init__(self,
                 names: Tuple[str, ...],
                 indices: Optional[Tuple[int]] = None,
                 auto_enumerate: bool = True
                 ) -> None:
        """Constructs an organ factory.

        Args:
            names (Tuple[str, ...]): The names of the available organs.
            indices (Optional[Tuple[int]]): The indices of the available organs (default=None).
            auto_enumerate (bool): Indicates if the provided organ names should be automatically enumerated
             (default=True).
        """
        super().__init__()

        if isinstance(indices, tuple):
            assert len(names) == len(indices), f'The number of provided names and indices must be equal, ' \
                                               f'but is not ({len(names)} (names) vs. {len(indices)} (indices))!'

        self.names = names

        if not indices and auto_enumerate:
            self.indices = tuple(np.arange(len(names)))

        else:
            self.indices = indices

    def produce(self, name: str) -> Organ:
        """Produces a new organ.

        Args:
            name (str): The name of the new organ.

        Returns:
            Organ: The newly produced organ.
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
    """A class combining an organ with a rater."""

    def __init__(self,
                 organ: Union[Organ, str],
                 rater: Union[Rater, str]
                 ) -> None:
        """Constructs an organ rater combination.

        Args:
            organ (Union[Organ, str]): The organ or its name.
            rater (Union[Rater, str]): The rater or its name.
        """
        super().__init__()

        if isinstance(organ, str):
            organ = Organ(organ)

        if isinstance(rater, str):
            rater = Rater(rater)

        self.organ = organ
        self.rater = rater

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(str(self))

    @property
    def name(self) -> str:
        """Get the name of the combination.

        Returns:
            str: The combined name.
        """
        return self.rater.name + '_' + self.organ.name
