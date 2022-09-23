from typing import (
    Optional,
    Union)

from .rater import Rater


__all__ = ['Organ', 'OrganRaterCombination']


class Organ:
    """A class for identifying an organ.

    Notes:
        The :class:`Organ` is used to identify the organ segmented on a :class:`~pyradise.data.image.SegmentationImage`.
        If multiple organs are segmented on a single :class:`~pyradise.data.image.SegmentationImage`, the
        :class:`Organ` may be assigned an artificial name describing the set of organs.


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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Organ):
            return False

        return self.name == other.name

    def __hash__(self):
        return hash(str(self))


class OrganRaterCombination:
    """A class combining an :class:`Organ` with a :class:`~pyradise.data.rater.Rater`.

    Args:
        organ (Union[Organ, str]): The :class:`Organ` or its name.
        rater (Union[Rater, str]): The :class:`~pyradise.data.rater.Rater` or its name.
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

        self.organ: Organ = organ
        self.rater: Rater = rater

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
        return self.organ.name + '_' + self.rater.name
