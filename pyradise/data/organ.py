from typing import Optional, Union

from .annotator import Annotator

__all__ = ["Organ", "OrganAnnotatorCombination"]


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

    def __init__(self, name: str, index: Optional[int] = None) -> None:
        super().__init__()

        self.name: str = name
        self.index: Optional[int] = index

    def get_name(self) -> str:
        """Get the name of the :class:`Organ`.

        Returns:
            str: The name of the :class:`Organ`.
        """
        return self.name

    def set_name(self, name: str) -> None:
        """Set the name of the :class:`Organ`.

        Args:
            name (str): The name of the :class:`Organ`.
        """
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Organ):
            return False

        return self.name == other.name

    def __hash__(self):
        return hash(str(self))


class OrganAnnotatorCombination:
    """A class combining an :class:`Organ` with a :class:`~pyradise.data.annotator.Annotator`.

    Args:
        organ (Union[Organ, str]): The :class:`Organ` or its name.
        annotator (Union[Annotator, str]): The :class:`~pyradise.data.annotator.Annotator` or its name.
    """

    def __init__(self, organ: Union[Organ, str], annotator: Union[Annotator, str]) -> None:
        super().__init__()

        if isinstance(organ, str):
            organ = Organ(organ)

        if isinstance(annotator, str):
            annotator = Annotator(annotator)

        self.organ: Organ = organ
        self.annotator: Annotator = annotator

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(str(self))

    @property
    def name(self) -> str:
        """Get the name of the :class:`OrganAnnotatorCombination`.

        Returns:
            str: The combined name.
        """
        return self.organ.name + "_" + self.annotator.name
