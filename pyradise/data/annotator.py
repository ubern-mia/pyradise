from re import sub
from typing import Optional

__all__ = ["Annotator"]


class Annotator:
    """A class for identifying the annotator who segmented a certain organ. Because the name of the annotator takes
    every value, the annotator can either be a human expert or an auto-segmentation algorithm.

    Args:
        name (str): The name of the annotator.
        abbreviation (Optional[str]): The abbreviation of the annotator (default: None).
    """

    default_annotator_name = "NA"
    default_annotator_abbreviation = "NA"

    def __init__(self, name: str, abbreviation: Optional[str] = None) -> None:
        super().__init__()
        name_ = self._remove_illegal_characters(name)

        if not name_:
            raise ValueError(f"The annotator's consists exclusively of illegal characters!")

        self.name: str = name_
        self.abbreviation: Optional[str] = abbreviation

    def get_name(self) -> str:
        """Get the name of the :class:`Annotator`.

        Returns:
            str: The name of the :class:`Annotator`.
        """
        return self.name

    def get_abbreviation(self) -> Optional[str]:
        """Get the abbreviation of the :class:`Annotator`.

        Returns:
            Optional[str]: The abbreviation of the :class:`Annotator` if contained, otherwise :data:`None`.
        """
        return self.abbreviation

    @classmethod
    def get_default(cls) -> "Annotator":
        """Get the default :class:`Annotator`.

        The default :class:`Annotator` name is 'NA' and its abbreviation is also 'NA'.

        Returns:
            Annotator: The default :class:`Annotator`.
        """
        return Annotator(Annotator.default_annotator_name, Annotator.default_annotator_abbreviation)

    def is_default(self) -> bool:
        """Check if the :class:`Annotator` is the default :class:`Annotator`.

        Returns:
            bool: True if the :class:`Annotator` is the default :class:`Annotator`, otherwise False.
        """
        return self.name == Annotator.default_annotator_name and self.abbreviation == Annotator.default_annotator_name

    @staticmethod
    def _remove_illegal_characters(text: str) -> str:
        """Remove a set of illegal characters from a string.

        Args:
            text (str): The string to remove illegal characters from.

        Returns:
            str: The string without illegal characters.
        """
        illegal_characters = '[<>:/\\|?*"]|[\0-\31]'
        return sub(illegal_characters, "", text)

    def __str__(self) -> str:
        if self.abbreviation:
            return f"{self.name} ({self.abbreviation})"

        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Annotator):
            return False

        return self.name == other.name and self.abbreviation == other.abbreviation
