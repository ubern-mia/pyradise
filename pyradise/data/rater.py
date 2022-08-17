from typing import (
    Optional,
    Tuple)
from re import sub


__all__ = ['Rater', 'RaterFactory']


class Rater:
    """Representation of a human or automatic rater which is typically used in combination with a
    :class:`SegmentationImage` to render the assignment of multiple :class:`SegmentationImage` entries in a
    :class:`Subject` feasible.

    Notes:
        Instead of the rater's name an abbreviation can be used in addition to anonymize the raters name on outputs.

    Args:
        name (str): The name of the rater.
        abbreviation (Optional[str]): The abbreviation of the rater (default: None).
    """

    def __init__(self,
                 name: str,
                 abbreviation: Optional[str] = None
                 ) -> None:
        super().__init__()
        name_ = self._remove_illegal_characters(name)

        if not name_:
            raise ValueError(f"The rater's consists exclusively of illegal characters!")

        self.name: str = name_
        self.abbreviation: Optional[str] = abbreviation

    def get_name(self) -> str:
        """Get the name of the :class:`Rater`.

        Returns:
            str: The name of the :class:`Rater`.
        """
        return self.name

    def get_abbreviation(self) -> Optional[str]:
        """Get the abbreviation of the :class:`Rater`.

        Returns:
            Optional[str]: The abbreviation of the :class:`Rater` if contained, otherwise None.
        """
        return self.abbreviation

    @staticmethod
    def _remove_illegal_characters(text: str) -> str:
        illegal_characters = "[<>:/\\|?*\"]|[\0-\31]"
        return sub(illegal_characters, "", text)

    def __str__(self) -> str:
        if self.abbreviation:
            return f'{self.name} ({self.abbreviation})'

        return self.name


class RaterFactory:
    """A :class:`Rater` producing factory class.

    Args:
        raters (Tuple[str, ...]): All available :class:`Rater` names.
        abbreviations (Optional[Tuple[str, ...]]): All abbreviations to the raters (default: None).
    """

    def __init__(self,
                 raters: Tuple[str, ...],
                 abbreviations: Optional[Tuple[str, ...]] = None
                 ) -> None:
        super().__init__()

        if abbreviations:
            assert len(raters) == len(raters), f'The number of raters must be equal to the number of abbreviations ' \
                                               f'({len(raters)} (raters) vs. {len(abbreviations)} (abbrev.))!'

        self.raters = raters
        self.abbreviations = abbreviations

    def produce(self, name: str) -> Rater:
        """Produce a new :class:`Rater` if possible.

        Args:
            name (str): The name of the :class:`Rater`.

        Returns:
            Rater: The constructed :class:`Rater`.
        """
        if name not in self.raters:
            raise ValueError(f'The name {name} is not specified as a rater!')

        name_idx = self.raters.index(name)
        if self.abbreviations:
            abbreviation = self.abbreviations[name_idx]

        else:
            abbreviation = None

        return Rater(name, abbreviation)
