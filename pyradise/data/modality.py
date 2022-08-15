from enum import Enum


__all__ = ['Modality', 'ModalityFactory']


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
    PLACEHOLDER_0 = 300
    PLACEHOLDER_1 = 301
    PLACEHOLDER_2 = 302
    PLACEHOLDER_3 = 303
    PLACEHOLDER_4 = 304
    PLACEHOLDER_5 = 305
    PLACEHOLDER_6 = 306
    PLACEHOLDER_7 = 307
    PLACEHOLDER_8 = 308
    PLACEHOLDER_9 = 309
    PLACEHOLDER_10 = 310
    PLACEHOLDER_11 = 311
    PLACEHOLDER_12 = 312
    PLACEHOLDER_13 = 313
    PLACEHOLDER_14 = 314
    PLACEHOLDER_15 = 315
    PLACEHOLDER_16 = 316
    PLACEHOLDER_17 = 317
    PLACEHOLDER_18 = 318
    PLACEHOLDER_19 = 319
    PLACEHOLDER_20 = 320
    LB = 500

    def __str__(self) -> str:
        return self.name


class ModalityFactory:
    """A factory class producing Modalities."""

    @staticmethod
    def produce(name: str) -> Modality:
        """Produce a :class:`Modality` based on its name.

        Args:
            name (str): The name of the modality.

        Returns:
            Modality: The correct :class:`Modality` if possible.
        """
        try:
            return Modality[name]

        except KeyError:
            return Modality.UNKNOWN
