from typing import Optional
from re import sub


__all__ = ['Rater']


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
    default_rater_name = 'NA'
    default_rater_abbreviation = 'NA'

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

    @classmethod
    def get_default(cls) -> 'Rater':
        """Get the default :class:`Rater`.

        Returns:
            Rater: The default :class:`Rater`.
        """
        return Rater(Rater.default_rater_name, Rater.default_rater_abbreviation)

    def is_default(self) -> bool:
        """Check if the :class:`Rater` is the default :class:`Rater`.

        Returns:
            bool: True if the :class:`Rater` is the default :class:`Rater`, otherwise False.
        """
        return self.name == Rater.default_rater_name and self.abbreviation == Rater.default_rater_name

    @staticmethod
    def _remove_illegal_characters(text: str) -> str:
        illegal_characters = "[<>:/\\|?*\"]|[\0-\31]"
        return sub(illegal_characters, "", text)

    def __str__(self) -> str:
        if self.abbreviation:
            return f'{self.name} ({self.abbreviation})'

        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rater):
            return False

        return self.name == other.name and self.abbreviation == other.abbreviation
