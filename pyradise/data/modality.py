__all__ = ["Modality"]


class Modality:
    """A class for identifying the imaging modality and its details.

    Notes:
        The :class:`Modality` class is used to discriminate between different imaging modalities and its details
        (e.g. the MR-sequence (T1c or T1w)). We are aware that the name modality may be misleading and does not follow
        precisely the professional taxonomy of the community, but we decided to stick to it for the sake of clarity and
        ease of use.

    Args:
        name (str): The name of the modality.

    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.default_name = "UNKNOWN"

    @classmethod
    def get_default(cls) -> "Modality":
        """Get the default :class:`Modality`.

        Notes:
            The default :class:`Modality` is 'UNKNOWN'.

        Returns:
            Modality: The default :class:`Modality`.
        """
        return Modality("UNKNOWN")

    def is_default(self) -> bool:
        """Check if the :class:`Modality` is the default :class:`Modality`.

        Notes:
            The default :class:`Modality` is 'UNKNOWN'.

        Returns:
            bool: True if the :class:`Modality` is the default :class:`Modality`, otherwise False.
        """
        return self.name == self.default_name

    def get_name(self) -> str:
        """Get the name of the :class:`Modality`.

        Returns:
            str: The name of the :class:`Modality`.
        """
        return self.name

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Modality):
            raise False

        return self.name == other.name

    def __hash__(self):
        return hash(self.name)
