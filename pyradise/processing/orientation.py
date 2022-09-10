from enum import Enum
from typing import Union

import SimpleITK as sitk

from pyradise.data import (
    Subject,
    TransformationInformation)
from .base import (
    Filter,
    FilterParameters)


class Coord(Enum):
    """An enum class containing all available medical directions to build the :class:`SpatialOrientation` entries."""

    ITK_COORDINATE_UNKNOWN = 0
    """Default value for a unidentifiable coordinate direction."""

    RIGHT = 2
    """Coordinate direction right."""

    LEFT = 3
    """Coordinate direction left."""

    POSTERIOR = 4  # back
    """Coordinate direction posterior."""

    ANTERIOR = 5  # front
    """Coordinate direction anterior."""

    INFERIOR = 8  # below
    """Coordinate direction inferior."""

    SUPERIOR = 9  # above
    """Coordinate direction superior."""


class MajorTerms(Enum):
    """An enum class for the possible axes within an image to describe the orientation. This enum is used to build
    the :class:`SpatialOrientation`"""

    PRIMARY_MINOR = 0
    """The primary minor axis."""

    SECONDARY_MINOR = 8
    """The secondary minor axis."""

    TERTIARY_MINOR = 16
    """The tertiary minor axis."""


class SpatialOrientation(Enum):
    """An enum class for all possible medical image orientations an image can possess."""

    INVALID = Coord.ITK_COORDINATE_UNKNOWN
    """The default value for an unidentifiable image orientation."""

    RIP = (Coord.RIGHT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The right inferior posterior (RIP) orientation."""

    LIP = (Coord.LEFT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The left inferior posterior (LIP) orientation."""


    RSP = (Coord.RIGHT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The right superior posterior (RSP) orientation."""

    LSP = (Coord.LEFT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The left superior posterior (LSP) orientation."""

    RIA = (Coord.RIGHT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The right inferior anterior (RIA) orientation."""

    LIA = (Coord.LEFT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The left inferior anterior (LIA) orientation."""

    RSA = (Coord.RIGHT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The right superior anterior (RSA) orientation."""

    LSA = (Coord.LEFT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The left superior anterior (LSA) orientation."""

    IRP = (Coord.INFERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The inferior right posterior (IRP) orientation."""

    ILP = (Coord.INFERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The inferior left posterior (ILP) orientation."""

    SRP = (Coord.SUPERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The superior right posterior (SRP) orientation."""

    SLP = (Coord.SUPERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The superior left posterior (SLP) orientation."""

    IRA = (Coord.INFERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The inferior right anterior (IRA) orientation."""

    ILA = (Coord.INFERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The inferior left anterior (ILA) orientation."""

    SRA = (Coord.SUPERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The superior right anterior (SRA) orientation."""

    SLA = (Coord.SUPERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The superior left anterior (SLA) orientation."""

    RPI = (Coord.RIGHT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The right posterior inferior (RPI) orientation."""

    LPI = (Coord.LEFT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The left posterior inferior (LPI) orientation."""

    RAI = (Coord.RIGHT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The right anterior inferior (RAI) orientation."""

    LAI = (Coord.LEFT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The left anterior inferior (LAI) orientation."""

    RPS = (Coord.RIGHT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The right posterior superior (RPS) orientation."""

    LPS = (Coord.LEFT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The left posterior superior (LPS) orientation."""

    RAS = (Coord.RIGHT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The right anterior superior (RAS) orientation."""

    LAS = (Coord.LEFT.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The left anterior superior (LAS) orientation."""

    PRI = (Coord.POSTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The posterior right inferior (PRI) orientation."""

    PLI = (Coord.POSTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The posterior left inferior (PLI) orientation."""

    ARI = (Coord.ANTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The anterior right inferior (ARI) orientation."""

    ALI = (Coord.ANTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The anterior left inferior (ALI) orientation."""

    PRS = (Coord.POSTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The posterior right superior (PRS) orientation."""

    PLS = (Coord.POSTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The posterior left superior (PLS) orientation."""

    ARS = (Coord.ANTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The anterior right superior (ARS) orientation."""

    ALS = (Coord.ANTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.TERTIARY_MINOR.value)
    """The anterior left superior (ALS) orientation."""

    IPR = (Coord.INFERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.TERTIARY_MINOR.value)
    """The inferior posterior right (IPR) orientation."""

    SPR = (Coord.SUPERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.TERTIARY_MINOR.value)
    """The superior posterior right (SPR) orientation."""

    IAR = (Coord.INFERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.TERTIARY_MINOR.value)
    """The inferior anterior right (IAR) orientation."""

    SAR = (Coord.SUPERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.TERTIARY_MINOR.value)
    """The superior anterior right (SAR) orientation."""

    IPL = (Coord.INFERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.TERTIARY_MINOR.value)
    """The inferior posterior left (IPL) orientation."""

    SPL = (Coord.SUPERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.POSTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.TERTIARY_MINOR.value)
    """The superior posterior left (SPL) orientation."""

    IAL = (Coord.INFERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.TERTIARY_MINOR.value)
    """The inferior anterior left (IAL) orientation."""

    SAL = (Coord.SUPERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.ANTERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.TERTIARY_MINOR.value)
    """The superior anterior left (SAL) orientation."""

    PIR = (Coord.POSTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.TERTIARY_MINOR.value)
    """The posterior inferior right (PIR) orientation."""

    PSR = (Coord.POSTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.TERTIARY_MINOR.value)
    """The posterior superior right (PSR) orientation."""

    AIR = (Coord.ANTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.TERTIARY_MINOR.value)
    """The anterior inferior right (AIR) orientation."""

    ASR = (Coord.ANTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.RIGHT.value << MajorTerms.TERTIARY_MINOR.value)
    """The anterior superior right (ASR) orientation."""

    PIL = (Coord.POSTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.TERTIARY_MINOR.value)
    """The posterior inferior left (PIL) orientation."""

    PSL = (Coord.POSTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.TERTIARY_MINOR.value)
    """The posterior superior left (PSL) orientation."""

    AIL = (Coord.ANTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.INFERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.TERTIARY_MINOR.value)
    """The anterior inferior left (AIL) orientation."""

    ASL = (Coord.ANTERIOR.value << MajorTerms.PRIMARY_MINOR.value) + \
          (Coord.SUPERIOR.value << MajorTerms.SECONDARY_MINOR.value) + \
          (Coord.LEFT.value << MajorTerms.TERTIARY_MINOR.value)
    """The anterior superior left (ASL) orientation."""


# pylint: disable = too-few-public-methods
class OrientationFilterParameters(FilterParameters):
    """A filter parameter class for an :class:`OrientationFilter`.

    Args:
        output_orientation: The desired output orientation of all images.
    """

    def __init__(self,
                 output_orientation: Union[SpatialOrientation, str]
                 ) -> None:
        super().__init__()
        if isinstance(output_orientation, str):
            self.output_orientation = SpatialOrientation[output_orientation]
        else:
            self.output_orientation = output_orientation


class OrientationFilter(Filter):
    """A filter class for reorienting image data of a :class:`Subject`."""

    def execute(self,
                subject: Subject,
                params: OrientationFilterParameters
                ) -> Subject:
        """Execute the image reorientation procedure.

        Args:
            subject (Subject): The :class:`Subject` instance to be processed.
            params (OrientationFilterParameters): The filters parameters.

        Returns:
            Subject: The :class:`Subject` instance with oriented :class:`IntensityImage` and
             :class:`SegmentationImage` entries.
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
