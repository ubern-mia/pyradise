from enum import Enum
from typing import Optional, Union

import SimpleITK as sitk

from pyradise.data import (IntensityImage, SegmentationImage, Subject,
                           TransformInfo)

from .base import Filter, FilterParams

__all_ = ["OrientationFilterParameters", "OrientationFilter", "SpatialOrientation"]


class _Coord(Enum):
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


class _MajorTerms(Enum):
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

    INVALID = _Coord.ITK_COORDINATE_UNKNOWN
    """The default value for an unidentifiable image orientation."""

    RIP = (
        (_Coord.RIGHT.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.INFERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.POSTERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The right inferior posterior (RIP) orientation."""

    LIP = (
        (_Coord.LEFT.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.INFERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.POSTERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The left inferior posterior (LIP) orientation."""

    RSP = (
        (_Coord.RIGHT.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.SUPERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.POSTERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The right superior posterior (RSP) orientation."""

    LSP = (
        (_Coord.LEFT.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.SUPERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.POSTERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The left superior posterior (LSP) orientation."""

    RIA = (
        (_Coord.RIGHT.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.INFERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.ANTERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The right inferior anterior (RIA) orientation."""

    LIA = (
        (_Coord.LEFT.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.INFERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.ANTERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The left inferior anterior (LIA) orientation."""

    RSA = (
        (_Coord.RIGHT.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.SUPERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.ANTERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The right superior anterior (RSA) orientation."""

    LSA = (
        (_Coord.LEFT.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.SUPERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.ANTERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The left superior anterior (LSA) orientation."""

    IRP = (
        (_Coord.INFERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.RIGHT.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.POSTERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The inferior right posterior (IRP) orientation."""

    ILP = (
        (_Coord.INFERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.LEFT.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.POSTERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The inferior left posterior (ILP) orientation."""

    SRP = (
        (_Coord.SUPERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.RIGHT.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.POSTERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The superior right posterior (SRP) orientation."""

    SLP = (
        (_Coord.SUPERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.LEFT.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.POSTERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The superior left posterior (SLP) orientation."""

    IRA = (
        (_Coord.INFERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.RIGHT.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.ANTERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The inferior right anterior (IRA) orientation."""

    ILA = (
        (_Coord.INFERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.LEFT.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.ANTERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The inferior left anterior (ILA) orientation."""

    SRA = (
        (_Coord.SUPERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.RIGHT.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.ANTERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The superior right anterior (SRA) orientation."""

    SLA = (
        (_Coord.SUPERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.LEFT.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.ANTERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The superior left anterior (SLA) orientation."""

    RPI = (
        (_Coord.RIGHT.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.POSTERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.INFERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The right posterior inferior (RPI) orientation."""

    LPI = (
        (_Coord.LEFT.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.POSTERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.INFERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The left posterior inferior (LPI) orientation."""

    RAI = (
        (_Coord.RIGHT.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.ANTERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.INFERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The right anterior inferior (RAI) orientation."""

    LAI = (
        (_Coord.LEFT.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.ANTERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.INFERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The left anterior inferior (LAI) orientation."""

    RPS = (
        (_Coord.RIGHT.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.POSTERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.SUPERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The right posterior superior (RPS) orientation."""

    LPS = (
        (_Coord.LEFT.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.POSTERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.SUPERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The left posterior superior (LPS) orientation."""

    RAS = (
        (_Coord.RIGHT.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.ANTERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.SUPERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The right anterior superior (RAS) orientation."""

    LAS = (
        (_Coord.LEFT.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.ANTERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.SUPERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The left anterior superior (LAS) orientation."""

    PRI = (
        (_Coord.POSTERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.RIGHT.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.INFERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The posterior right inferior (PRI) orientation."""

    PLI = (
        (_Coord.POSTERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.LEFT.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.INFERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The posterior left inferior (PLI) orientation."""

    ARI = (
        (_Coord.ANTERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.RIGHT.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.INFERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The anterior right inferior (ARI) orientation."""

    ALI = (
        (_Coord.ANTERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.LEFT.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.INFERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The anterior left inferior (ALI) orientation."""

    PRS = (
        (_Coord.POSTERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.RIGHT.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.SUPERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The posterior right superior (PRS) orientation."""

    PLS = (
        (_Coord.POSTERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.LEFT.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.SUPERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The posterior left superior (PLS) orientation."""

    ARS = (
        (_Coord.ANTERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.RIGHT.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.SUPERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The anterior right superior (ARS) orientation."""

    ALS = (
        (_Coord.ANTERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.LEFT.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.SUPERIOR.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The anterior left superior (ALS) orientation."""

    IPR = (
        (_Coord.INFERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.POSTERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.RIGHT.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The inferior posterior right (IPR) orientation."""

    SPR = (
        (_Coord.SUPERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.POSTERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.RIGHT.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The superior posterior right (SPR) orientation."""

    IAR = (
        (_Coord.INFERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.ANTERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.RIGHT.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The inferior anterior right (IAR) orientation."""

    SAR = (
        (_Coord.SUPERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.ANTERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.RIGHT.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The superior anterior right (SAR) orientation."""

    IPL = (
        (_Coord.INFERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.POSTERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.LEFT.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The inferior posterior left (IPL) orientation."""

    SPL = (
        (_Coord.SUPERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.POSTERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.LEFT.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The superior posterior left (SPL) orientation."""

    IAL = (
        (_Coord.INFERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.ANTERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.LEFT.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The inferior anterior left (IAL) orientation."""

    SAL = (
        (_Coord.SUPERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.ANTERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.LEFT.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The superior anterior left (SAL) orientation."""

    PIR = (
        (_Coord.POSTERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.INFERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.RIGHT.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The posterior inferior right (PIR) orientation."""

    PSR = (
        (_Coord.POSTERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.SUPERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.RIGHT.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The posterior superior right (PSR) orientation."""

    AIR = (
        (_Coord.ANTERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.INFERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.RIGHT.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The anterior inferior right (AIR) orientation."""

    ASR = (
        (_Coord.ANTERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.SUPERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.RIGHT.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The anterior superior right (ASR) orientation."""

    PIL = (
        (_Coord.POSTERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.INFERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.LEFT.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The posterior inferior left (PIL) orientation."""

    PSL = (
        (_Coord.POSTERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.SUPERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.LEFT.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The posterior superior left (PSL) orientation."""

    AIL = (
        (_Coord.ANTERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.INFERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.LEFT.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The anterior inferior left (AIL) orientation."""

    ASL = (
        (_Coord.ANTERIOR.value << _MajorTerms.PRIMARY_MINOR.value)
        + (_Coord.SUPERIOR.value << _MajorTerms.SECONDARY_MINOR.value)
        + (_Coord.LEFT.value << _MajorTerms.TERTIARY_MINOR.value)
    )
    """The anterior superior left (ASL) orientation."""


# pylint: disable = too-few-public-methods
class OrientationFilterParams(FilterParams):
    """A filter parameter class for the :class:`~pyradise.process.orientation.OrientationFilter`.

    The orientation is a string or :class:`~pyradise.process.orientation.SpatialOrientation` value consisting of three
    characters. These three characters describe the orientation of the image with the following values:

    * ``I``: Inferior
    * ``S``: Superior
    * ``A``: Anterior
    * ``P``: Posterior
    * ``R``: Right
    * ``L``: Left

    The orientation of the image is described by the order of the characters. The first character describes the primary
    orientation of patient in the positive x-axis direction. The second character describes the secondary orientation of
    patient in the positive y-axis direction. The third character describes the tertiary orientation of patient in the
    positive z-axis direction. For example, the orientation ``RAS`` means that the patient is facing right in the
    positive x-axis direction, facing anterior in the positive y-axis direction, and facing superior in the positive
    z-axis direction. For more details we refer to appropriate literature (e.g.
    `website <http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm>`_).


    Args:
        output_orientation (Union[SpatialOrientation, str]): The desired output orientation of all provided images.
    """

    def __init__(self, output_orientation: Union[SpatialOrientation, str]) -> None:
        super().__init__()
        if isinstance(output_orientation, str):
            try:
                self.output_orientation: SpatialOrientation = SpatialOrientation[output_orientation]
            except KeyError:
                raise ValueError(f"Invalid output orientation: {output_orientation}")
        else:
            self.output_orientation: SpatialOrientation = output_orientation


class OrientationFilter(Filter):
    """A filter class for reorienting all :class:`~pyradise.data.image.Image` instances of a
    :class:`~pyradise.data.subject.Subject` instance to a specified
    :class:`~pyradise.process.orientation.SpatialOrientation`.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Returns whether the filter is invertible or not.

        Returns:
            bool: True because the reorientation of images is invertible.
        """
        return True

    def execute(self, subject: Subject, params: OrientationFilterParams) -> Subject:
        """Execute the image reorientation procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            params (OrientationFilterParams): The filters parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with oriented
            :class:`~pyradise.data.image.IntensityImage` and :class:`~pyradise.data.image.SegmentationImage` instances.
        """
        for image in subject.get_images():
            # get the image data as SimpleITK image
            sitk_image = image.get_image_data()

            # reorient the image
            orient_filter = sitk.DICOMOrientImageFilter()
            orient_filter.SetDesiredCoordinateOrientation(params.output_orientation.name)
            oriented_sitk_image = orient_filter.Execute(sitk_image)

            # set the oriented image data to the image
            image.set_image_data(oriented_sitk_image)

            # track the necessary information
            pre_orientation = orient_filter.GetOrientationFromDirectionCosines(sitk_image.GetDirection())
            self.tracking_data["original_orientation"] = pre_orientation
            self._register_tracked_data(image, sitk_image, oriented_sitk_image, params)

        return subject

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Execute the inverse image reorientation procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            transform_info (TransformInfo): The :class:`~pyradise.data.taping.TransformInfo` instance.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with reoriented
            :class:`~pyradise.data.image.IntensityImage` and :class:`~pyradise.data.image.SegmentationImage` instances.
        """
        for image in subject.get_images():
            if target_image is not None and image != target_image:
                continue

            # get the original orientation
            original_orient = transform_info.get_data("original_orientation")

            # reorient the image
            orient_filter = sitk.DICOMOrientImageFilter()
            orient_filter.SetDesiredCoordinateOrientation(original_orient)
            oriented_sitk_image = orient_filter.Execute(image.get_image_data())

            # set the oriented image data to the image
            image.set_image_data(oriented_sitk_image)

        return subject
