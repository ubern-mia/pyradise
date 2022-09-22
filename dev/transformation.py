from typing import (
    Tuple,
    Union,
    Optional)
from copy import deepcopy

import numpy as np
import SimpleITK as sitk

from pyradise.data import (
    Subject,
    Modality,
    IntensityImage,
    SegmentationImage,
    TransformTape,
    ImageProperties,
    TransformInfo,
    OrganRaterCombination)
from pyradise.process.base import (
    Filter,
    FilterParams)


__all__ = ['ApplyTransformationTapeFilterParams', 'ApplyTransformationTapeFilter',
           'BackTransformIntensityImageFilterParams', 'BackTransformIntensityImageFilter',
           'BackTransformSegmentationFilterParams', 'BackTransformSegmentationFilter',
           'CopyReferenceTransformTapeFilterParams', 'CopyReferenceTransformTapeFilter']


# pylint: disable = too-few-public-methods
class ApplyTransformationTapeFilterParams(FilterParams):
    """A class representing the parameters for a ApplyTransformationTapeFilter.

    Args:
        targets (Optional[Tuple[Union[Modality, OrganRaterCombination]]]): The targets to which the transformations
         should be applied.
        transform_source (Optional[Union[Modality, OrganRaterCombination, TransformTape]]): The reference where the
         transformation is defined.
        backward_playback (bool): Indicates if the transform tape should be replayed forward or backward.
        clear_transformation_tapes (bool): Indicates if the transformation tapes should be cleared.
    """

    def __init__(self,
                 targets: Optional[Tuple[Union[Modality, OrganRaterCombination]]] = None,
                 transform_source: Optional[Union[Modality, OrganRaterCombination, TransformTape]] = None,
                 backward_playback: bool = True,
                 clear_transformation_tapes: bool = False
                 ) -> None:
        super().__init__()
        self.targets: Optional[Tuple[Union[Modality, OrganRaterCombination]]] = targets
        self.transform_source: Optional[Union[Modality, OrganRaterCombination, TransformTape]] = transform_source
        self.backward_playback = backward_playback
        self.clear_transformation_tapes = clear_transformation_tapes


class ApplyTransformationTapeFilter(Filter):
    """A class which reapplies transformations to the selected images of a subject."""

    @staticmethod
    def _get_images_to_process(subject: Subject,
                               params: ApplyTransformationTapeFilterParams
                               ) -> Tuple[Union[IntensityImage, SegmentationImage]]:
        """Get the images to process.

        Args:
            subject (Subject): The subject holding the data.
            params (ApplyTransformationTapeFilterParams): The parameters specifying the reference.

        Returns:
            Tuple[Union[IntensityImage, SegmentationImage]]: The images to process.
        """
        if not params.targets:
            to_process = []
            to_process.extend(subject.intensity_images)
            to_process.extend(subject.segmentation_images)
            return tuple(to_process)

        to_process = []

        for target in params.targets:
            if isinstance(target, Modality):
                for image in subject.intensity_images:
                    if image.get_modality() == target:
                        to_process.append(image)

            elif isinstance(target, OrganRaterCombination):
                for image in subject.segmentation_images:
                    if image.get_organ_rater_combination() == target:
                        to_process.append(image)

        return tuple(to_process)

    @staticmethod
    def _get_transform_tape(subject: Subject,
                            params: ApplyTransformationTapeFilterParams
                            ) -> Optional[TransformTape]:
        """Get the correct transformation tape according to the specification in the parameters.

        Args:
            subject (Subject): The subject holding the data.
            params (ApplyTransformationTapeFilterParams): The parameters holding the reference.

        Returns:
            Optional[TransformTape]: The transform tape to apply.
        """
        if not params.transform_source:
            return None

        if isinstance(params.transform_source, Modality):

            for image in subject.intensity_images:
                if image.get_modality() == params.transform_source:
                    return image.get_transform_tape()

        elif isinstance(params.transform_source, TransformTape):
            return params.transform_source

        else:
            for image in subject.segmentation_images:
                if image.get_organ_rater_combination() == params.transform_source:
                    return image.get_transform_tape()

        return None

    # pylint: disable = duplicate-code
    # noinspection DuplicatedCode
    @staticmethod
    def _is_reorient_only(transform_info: TransformInfo,
                          invert: bool
                          ) -> bool:
        transform = transform_info.get_transform(invert)

        identity_transform = sitk.Transform(transform.GetDimension(), transform.GetTransformEnum())
        identity_transform.SetIdentity()

        criteria = (transform.GetParameters() == identity_transform.GetParameters(),
                    transform_info.get_size(True) == transform_info.get_size(False))

        return all(criteria)

    # pylint: disable = duplicate-code
    # noinspection DuplicatedCode
    @staticmethod
    def _apply_transform(image: Union[IntensityImage, SegmentationImage],
                         transform_tape: Optional[TransformTape],
                         params: ApplyTransformationTapeFilterParams
                         ) -> Union[IntensityImage, SegmentationImage]:
        """Apply the transformation to the image and clears the transformation tape if allowed.

        Args:
            image (Union[IntensityImage, SegmentationImage]): The image to apply the transformations to.
            transform_tape (Optional[TransformTape]): The transformation tape holding the transformations.
             If none is provided the transformation type of the image itself is used.
            params (ApplyTransformationTapeFilterParams): The filter parameters.

        Returns:
            Union[IntensityImage, SegmentationImage]: The modified image.
        """
        if not transform_tape:
            transform_tape_ = image.get_transform_tape()
        else:
            transform_tape_ = transform_tape

        transform_infos = transform_tape_.get_recorded_elements(params.backward_playback)

        image_sitk = image.get_image_data(as_sitk=True)

        for transform_info in transform_infos:

            image_sitk_pre = deepcopy(image_sitk)

            if transform_tape_.is_reorient_only(transform_info, True):
                transform = sitk.AffineTransform(3)
                transform.SetIdentity()

                image_sitk = sitk.DICOMOrient(image_sitk, transform_info.get_orientation(params.backward_playback))

            else:

                if image.is_intensity_image():
                    interpolator = sitk.sitkBSpline
                    default_pixel_value = np.min(sitk.GetArrayFromImage(image_sitk))
                else:
                    interpolator = sitk.sitkNearestNeighbor
                    default_pixel_value = 0

                transform = transform_info.get_transform(inverse=True)

                resample_filter = sitk.ResampleImageFilter()
                resample_filter.SetTransform(transform)
                resample_filter.SetInterpolator(interpolator)
                resample_filter.SetDefaultPixelValue(default_pixel_value)
                resample_filter.SetSize(transform_info.get_size(params.backward_playback))
                resample_filter.SetOutputOrigin(transform_info.get_origin(params.backward_playback))
                resample_filter.SetOutputDirection(transform_info.get_direction(params.backward_playback, False))
                resample_filter.SetOutputSpacing(transform_info.get_spacing(params.backward_playback))

                image_sitk = resample_filter.Execute(image_sitk)

            # TODO Correct this
            # new_transform_info = TransformationInformation.from_images(f'Inverse of {transform_info.name}',
            #                                                            transform,
            #                                                            image_sitk_pre, image_sitk)
            # image.get_transform_tape().record(new_transform_info)

        image.set_image_data(image_sitk)

        if params.clear_transformation_tapes:
            image.get_transform_tape().reset()

        return image

    def execute(self,
                subject: Subject,
                params: ApplyTransformationTapeFilterParams
                ) -> Subject:
        """Execute the filter and reapplies the transformations.

        Args:
            subject (Subject): The subject to process.
            params (ApplyTransformationTapeFilterParams): The filters parameters.

        Returns:
            Subject: The processed subject.
        """
        to_process = self._get_images_to_process(subject, params)

        transform_tape = self._get_transform_tape(subject, params)

        for image in to_process:
            self._apply_transform(image, transform_tape, params)

        return subject


class BackTransformSegmentationFilterParams(FilterParams):
    """A class for the parameters of a BackTransformSegmentationFilter."""


class BackTransformSegmentationFilter(Filter):
    """A filter class for playing back the transforms on all segmentations."""

    def execute(self,
                subject: Subject,
                params: Optional[BackTransformSegmentationFilterParams]
                ) -> Subject:
        """Execute the filter.

        Args:
            subject (Subject): The subject to be processed.
            params (Optional[BackTransformSegmentationFilterParams]): The filter parameters.

        Returns:
            Subject: The processed subject.
        """
        for segmentation in subject.segmentation_images:
            TransformTape.playback(segmentation)

        return subject


class BackTransformIntensityImageFilterParams(FilterParams):
    """A class for the parameters of a BackTransformIntensityImageFilter."""


class BackTransformIntensityImageFilter(Filter):
    """A filter class for playing back the transforms on all intensity images."""

    def execute(self,
                subject: Subject,
                params: Optional[BackTransformIntensityImageFilterParams]
                ) -> Subject:
        """Execute the filter.

        Args:
            subject (Subject): The subject to be processed.
            params (Optional[BackTransformIntensityImageFilterParams]): The filter parameters.

        Returns:
            Subject: The processed subject.
        """
        for image in subject.intensity_images:
            TransformTape.playback(image)

        return subject


# pylint: disable = too-few-public-methods
class CopyReferenceTransformTapeFilterParams(FilterParams):
    """A class for the parameters of a CopyReferenceTransformTapeFilter.

    Args:
        reference_modality (Modality): The reference modality.
        excluded_organs (Tuple[OrganRaterCombination, ...]): The organs to exclude (default: ()).
    """

    def __init__(self,
                 reference_modality: Modality,
                 excluded_organs: Tuple[OrganRaterCombination, ...] = ()
                 ) -> None:
        super().__init__()

        self.reference_modality = reference_modality
        self.excluded_organs = excluded_organs


class CopyReferenceTransformTapeFilter(Filter):
    """A filter class for copying the transformation tape from an intensity image to all segmentation images."""

    @staticmethod
    def _get_reference_transformation_tape(subject: Subject,
                                           reference_modality: Modality
                                           ) -> TransformTape:
        for image in subject.intensity_images:
            if image.get_modality() == reference_modality:
                return image.get_transform_tape()

        raise Exception('No transform tape found matching the reference modality!')

    def execute(self,
                subject: Subject,
                params: CopyReferenceTransformTapeFilterParams
                ) -> Subject:
        """Execute the copying procedure for the transformation tape.

        Args:
            subject (Subject): The subject.
            params (CopyReferenceTransformTapeFilterParams): The filter parameters.

        Returns:
            Subject: The modified subject.
        """
        reference_transform_tape = self._get_reference_transformation_tape(subject, params.reference_modality)

        for segmentation in subject.segmentation_images:
            segmentation.transform_tape = reference_transform_tape

        return subject
