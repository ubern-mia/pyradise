from enum import Enum


class Modality(Enum):
    """A modality class specifying the modality of IntensityImages."""
    # pylint: disable=invalid-name
    UNKNOWN = 0
    T1c = 1
    T1w = 2
    T2w = 3
    FLAIR = 4
    CT = 5
    T1c_7T = 101
    T1w_7T = 102
    T2w_7T = 103
    FLAIR_7T = 104
    T1c_3T = 201
    T1w_3T = 202
    T2w_3T = 203
    FLAIR_3T = 204
    LB = 500

    def __str__(self) -> str:
        return self.name


class ModalityFactory:
    """A factory class producing Modalities."""

    @staticmethod
    def produce(name: str) -> Modality:
        """Produces Modalities based on the modality name.

        Args:
            name (str): The name of the modality.

        Returns:
            Modality: The correct Modality if possible.
        """
        try:
            return Modality[name]

        except KeyError:
            return Modality.UNKNOWN
