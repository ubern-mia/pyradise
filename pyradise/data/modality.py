from enum import Enum


__all__ = ['Modality', 'ModalityFactory']


class Modality(Enum):
    """Representation of a modality for identifying the imaging modality or the MR sequence of an 
    :class:`IntensityImage`. For adding new modalities the class can be inherited and extended.
    
    Notes:
        In PyRaDiSe the taxonomy for identifying the imaging modality or the MR sequence is identical. 
        The name :class:`Modality` was a design choice for which we believe that it is easily understandable and is 
        taxonomically sufficient precise.
    """
    # pylint: disable=invalid-name
    UNKNOWN = 0
    """Default value for unidentifiable modalities"""
    
    T1c = 1
    """Identification of a T1-weighted MR sequence with contrast bolus agent."""
    
    T1w = 2
    """Identification of a T1-weighted MR sequence without contrast bolus agent."""
    
    T2w = 3
    """Identification of a T2-weighted MR sequence."""
    
    FLAIR = 4
    """Identification of a T2-weighted fluid-attenuated inverse recovery (FLAIR) MR sequence."""
    
    CT = 5
    """Identification of the CT modality."""

    T1c_1_5T = 101
    """Identification of a 1.5 tesla T1-weighted MR sequence with contrast bolus agent."""

    T1w_1_5T = 102
    """Identification of a 1.5 tesla T1-weighted MR sequence without contrast bolus agent."""

    T2w_1_5T = 103
    """Identification of a 1.5 tesla T2-weighted MR sequence."""

    FLAIR_1_5T = 104
    """Identification of a 1.5 tesla T2-weighted fluid-attenuated inverse recovery (FLAIR) MR sequence."""

    T1c_3T = 111
    """Identification of a 3 tesla T1-weighted MR sequence with contrast bolus agent."""

    T1w_3T = 112
    """Identification of a 3 tesla T1-weighted MR sequence without contrast bolus agent."""

    T2w_3T = 113
    """Identification of a 3 tesla T2-weighted MR sequence."""

    FLAIR_3T = 114
    """Identification of a 3 tesla T2-weighted fluid-attenuated inverse recovery (FLAIR) MR sequence."""
    
    T1c_7T = 121
    """Identification of a 7 tesla T1-weighted MR sequence with contrast bolus agent."""
    
    T1w_7T = 122
    """Identification of a 7 tesla T1-weighted MR sequence without contrast bolus agent."""
    
    T2w_7T = 123
    """Identification of a 7 tesla T2-weighted MR sequence."""
    
    FLAIR_7T = 124
    """Identification of a 7 tesla T2-weighted fluid-attenuated inverse recovery (FLAIR) MR sequence."""

    MODALITY_0 = 300
    """Placeholder for a yet unidentifiable modality."""

    MODALITY_1 = 301
    """Placeholder for a yet unidentifiable modality."""

    MODALITY_2 = 302
    """Placeholder for a yet unidentifiable modality."""

    MODALITY_3 = 303
    """Placeholder for a yet unidentifiable modality."""

    MODALITY_4 = 304
    """Placeholder for a yet unidentifiable modality."""

    MODALITY_5 = 305
    """Placeholder for a yet unidentifiable modality."""

    MODALITY_6 = 306
    """Placeholder for a yet unidentifiable modality."""

    MODALITY_7 = 307
    """Placeholder for a yet unidentifiable modality."""

    MODALITY_8 = 308
    """Placeholder for a yet unidentifiable modality."""

    MODALITY_9 = 309
    """Placeholder for a yet unidentifiable modality."""

    MODALITY_10 = 310
    """Placeholder for a yet unidentifiable modality."""

    def __str__(self) -> str:
        return self.name


class ModalityFactory:
    """A :class:`Modality` producing factory class."""

    @staticmethod
    def produce(name: str) -> Modality:
        """Produce a :class:`Modality` based on its name.

        Args:
            name (str): The name of the :class:`Modality`.

        Returns:
            Modality: The correct :class:`Modality` if possible, otherwise returns :class:`Modality.UNKNOWN`.
        """
        try:
            return Modality[name]

        except KeyError:
            return Modality.UNKNOWN
