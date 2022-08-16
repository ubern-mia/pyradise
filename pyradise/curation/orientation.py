from enum import Enum
from typing import Union

import SimpleITK as sitk

from pyradise.data import TransformationInformation
from .base import (
    Subject,
    Filter,
    FilterParameters)


class Coord(Enum):
    """An enum class containing all available medical directions."""
    ITK_COORDINATE_UNKNOWN = 0
    RIGHT = 2
    LEFT = 3
    POSTERIOR = 4  # back
    ANTERIOR = 5  # front
    INFERIOR = 8  # below
    SUPERIOR = 9  # above


class MajorTerms(Enum):
    """An enum class representing the possible axes within an image to describe the orientation."""
    PRIMARY_MINOR = 0
    SECONDARY_MINOR = 8
    TERTIARY_MINOR = 16


class SpatialOrientation(Enum):
    """An enum class for all possible medical image orientations."""
    INVALID = Coord.ITK_COORDINATE_UNKNOWN

    RIP = (Coord.RIGHT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    LIP = (Coord.LEFT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    RSP = (Coord.RIGHT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    LSP = (Coord.LEFT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    RIA = (Coord.RIGHT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    LIA = (Coord.LEFT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    RSA = (Coord.RIGHT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    LSA = (Coord.LEFT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    IRP = (Coord.INFERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    ILP = (Coord.INFERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    SRP = (Coord.SUPERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    SLP = (Coord.SUPERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    IRA = (Coord.INFERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    ILA = (Coord.INFERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    SRA = (Coord.SUPERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    SLA = (Coord.SUPERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    RPI = (Coord.RIGHT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    LPI = (Coord.LEFT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    RAI = (Coord.RIGHT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    LAI = (Coord.LEFT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    RPS = (Coord.RIGHT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    LPS = (Coord.LEFT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    RAS = (Coord.RIGHT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    LAS = (Coord.LEFT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    PRI = (Coord.POSTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    PLI = (Coord.POSTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    ARI = (Coord.ANTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    ALI = (Coord.ANTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    PRS = (Coord.POSTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    PLS = (Coord.POSTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    ARS = (Coord.ANTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    ALS = (Coord.ANTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.TERTIARY_MINOR.value)

    IPR = (Coord.INFERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.TERTIARY_MINOR.value)

    SPR = (Coord.SUPERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.TERTIARY_MINOR.value)

    IAR = (Coord.INFERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.TERTIARY_MINOR.value)

    SAR = (Coord.SUPERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.TERTIARY_MINOR.value)

    IPL = (Coord.INFERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.TERTIARY_MINOR.value)

    SPL = (Coord.SUPERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.TERTIARY_MINOR.value)

    IAL = (Coord.INFERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.TERTIARY_MINOR.value)

    SAL = (Coord.SUPERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.TERTIARY_MINOR.value)

    PIR = (Coord.POSTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.TERTIARY_MINOR.value)

    PSR = (Coord.POSTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.TERTIARY_MINOR.value)

    AIR = (Coord.ANTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.TERTIARY_MINOR.value)

    ASR = (Coord.ANTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.TERTIARY_MINOR.value)

    PIL = (Coord.POSTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.TERTIARY_MINOR.value)

    PSL = (Coord.POSTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.TERTIARY_MINOR.value)

    AIL = (Coord.ANTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.TERTIARY_MINOR.value)

    ASL = (Coord.ANTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.TERTIARY_MINOR.value)


# pylint: disable = too-few-public-methods
class OrientationFilterParameters(FilterParameters):
    """A class representing the parameters for an OrientationFilter."""

    def __init__(self,
                 output_orientation: Union[SpatialOrientation, str]
                 ) -> None:
        """Constructs the parameters for an OrientationFilter.

        Args:
            output_orientation (Union[SpatialOrientation, str]): The desired output orientation.
        """
        super().__init__()
        if isinstance(output_orientation, str):
            self.output_orientation = SpatialOrientation[output_orientation]
        else:
            self.output_orientation = output_orientation


class OrientationFilter(Filter):
    """A class for reorienting image data."""

    def execute(self,
                subject: Subject,
                params: OrientationFilterParameters
                ) -> Subject:
        """Executes the image reorientation procedure.

        Args:
            subject (Subject): The subject to be processed.
            params (OrientationFilterParameters): The filters parameters.

        Returns:
            Subject: The processed subject.
        """
        images = []
        images.extend(subject.intensity_images)
        images.extend(subject.segmentation_images)

        for image in images:
            sitk_image = image.get_image(True)

            orient_filter = sitk.DICOMOrientImageFilter()
            orient_filter.SetDesiredCoordinateOrientation(params.output_orientation.name)
            oriented_sitk_image = orient_filter.Execute(sitk_image)

            image.set_image(oriented_sitk_image)

            identity_transform = sitk.AffineTransform(3)
            identity_transform.SetIdentity()

            transform_info = TransformationInformation.from_images(self.__class__.__name__,
                                                                   identity_transform,
                                                                   sitk_image, oriented_sitk_image)
            image.get_transform_tape().record(transform_info)

        return subject
