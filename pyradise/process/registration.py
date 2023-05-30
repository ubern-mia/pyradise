from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk

from pyradise.data import (Image, IntensityImage, Modality, SegmentationImage,
                           Subject, TransformInfo, str_to_modality)

from .base import Filter, FilterParams

__all__ = [
    "IntraSubjectRegistrationFilterParams",
    "IntraSubjectRegistrationFilter",
    "InterSubjectRegistrationFilterParams",
    "InterSubjectRegistrationFilter",
    "RegistrationType",
]


class RegistrationType(Enum):
    """An enum class representing the different registration transform types."""

    AFFINE = 1
    """Affine registration."""

    SIMILARITY = 2
    """Similarity registration."""

    RIGID = 3
    """Rigid registration."""

    BSPLINE = 4
    """BSpline registration."""


def get_interpolator(image: Image) -> Optional[int]:
    """Get the appropriate interpolator for the given image depending on the image type.

    Args:
        image (Image): The image.

    Returns:
        Optional[int]: The interpolator.
    """
    if isinstance(image, IntensityImage):
        return sitk.sitkBSpline
    elif isinstance(image, SegmentationImage):
        return sitk.sitkNearestNeighbor
    else:
        return None


def get_registration_method(
    registration_type: RegistrationType = RegistrationType.RIGID,
    number_of_histogram_bins: int = 200,
    learning_rate: float = 1.0,
    step_size: float = 0.001,
    number_of_iterations: int = 1500,
    relaxation_factor: float = 0.5,
    shrink_factors: Tuple[int, ...] = (2, 2, 1),
    smoothing_sigmas: Tuple[float, ...] = (2, 1, 0),
    sampling_percentage: float = 0.2,
    deterministic: bool = True,
) -> sitk.ImageRegistrationMethod:
    """Get the registration method based on the provided parameters.

    Args:
        registration_type (RegistrationType): The type of registration (default: RegistrationType.RIGID).
        number_of_histogram_bins (int): The number of histogram bins for registration (default: 200).
        learning_rate (float): The learning rate of the optimizer (default: 1.0).
        step_size (float): The step size of the optimizer (default: 0.001).
        number_of_iterations (int): The maximal number of optimization iterations (default: 1500).
        relaxation_factor (float): The relaxation factor (default: 0.5).
        shrink_factors (Tuple[int, ...): The shrink factors for the image pyramid (default: (2, 2, 1))).
        smoothing_sigmas (Tuple[float, ...]): The smoothing sigmas (default: (2, 1, 0))).
        sampling_percentage (float): The sampling percentage of the voxels to incorporate into the optimization
         (default: 0.2).
        deterministic (bool): Deterministic processing with a fixed seed and a single thread (default: True).

    Returns:
        sitk.ImageRegistrationMethod: The registration method.
    """
    registration = sitk.ImageRegistrationMethod()

    registration.SetMetricAsMattesMutualInformation(number_of_histogram_bins)

    if deterministic:
        # https://simpleitk.readthedocs.io/en/master/registrationOverview.html
        registration.SetGlobalDefaultNumberOfThreads(0)
        sampling_seed = 42
        registration.SetMetricSamplingPercentage(sampling_percentage, sampling_seed)
    else:
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(sampling_percentage, sitk.sitkWallClock)

    registration.SetMetricUseFixedImageGradientFilter(False)
    registration.SetMetricUseMovingImageGradientFilter(False)

    registration.SetInterpolator(sitk.sitkLinear)

    if registration_type == RegistrationType.BSPLINE:
        registration.SetOptimizerAsLBFGSB()
    else:
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=learning_rate,
            minStep=step_size,
            numberOfIterations=number_of_iterations,
            relaxationFactor=relaxation_factor,
            gradientMagnitudeTolerance=1e-4,
            estimateLearningRate=registration.EachIteration,
            maximumStepSizeInPhysicalUnits=0.0,
        )

    registration.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework
    registration.SetShrinkFactorsPerLevel(shrink_factors)
    registration.SetSmoothingSigmasPerLevel(smoothing_sigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    return registration


def register_images(
    moving_image: sitk.Image,
    fixed_image: sitk.Image,
    registration_type: RegistrationType,
    registration_method: sitk.ImageRegistrationMethod,
) -> sitk.Transform:
    """Register the moving image to the fixed image and return the transformation.

    Args:
        moving_image (sitk.Image): The moving image.
        fixed_image (sitk.Image): The fixed image.
        registration_type (RegistrationType): The registration type.
        registration_method (sitk.ImageRegistrationMethod): The registration method.

    Returns:
        sitk.Transform: The registration transformation.
    """
    if moving_image.GetDimension() != fixed_image.GetDimension():
        raise ValueError("The floating and fixed image dimensions do not match!")

    dims = moving_image.GetDimension()
    if dims not in (2, 3):
        raise ValueError("The image must have 2 or 3 dimensions. Different number of dimensions are not supported!")

    moving_image_f32 = sitk.Cast(moving_image, sitk.sitkFloat32)
    fixed_image_f32 = sitk.Cast(fixed_image, sitk.sitkFloat32)

    if registration_type == RegistrationType.BSPLINE:
        transform_domain_mesh_size = [10] * dims
        initial_transform = sitk.BSplineTransformInitializer(fixed_image, transform_domain_mesh_size)
    else:
        if registration_type == RegistrationType.RIGID:
            transform_type = sitk.VersorRigid3DTransform() if dims == 3 else sitk.Euler2DTransform()

        elif registration_type == RegistrationType.AFFINE:
            transform_type = sitk.AffineTransform(dims)

        elif registration_type == RegistrationType.SIMILARITY:
            transform_type = sitk.Similarity3DTransform() if dims == 3 else sitk.Similarity2DTransform()

        else:
            raise ValueError(f"The registration type ({registration_type.name}) is not supported!")

        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image_f32, moving_image_f32, transform_type, sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

    registration_method.SetInitialTransform(initial_transform, inPlace=True)

    transform = registration_method.Execute(fixed_image_f32, moving_image_f32)

    return transform


# pylint: disable = too-few-public-methods
class IntraSubjectRegistrationFilterParams(FilterParams):
    """A filter parameter class for the :class:`~pyradise.process.registration.IntraSubjectRegistrationFilter` class.

    Args:
        reference_modality (Union[Modality, str]): The reference modality.
        registration_type (RegistrationType): The type of registration (default: RegistrationType.RIGID).
        number_of_histogram_bins (int): The number of histogram bins for registration (default: 200).
        learning_rate (float): The learning rate of the optimizer (default: 1.0).
        step_size (float): The step size of the optimizer (default: 0.001).
        number_of_iterations (int): The maximal number of optimization iterations (default: 1500).
        relaxation_factor (float): The relaxation factor (default: 0.5).
        shrink_factors (Tuple[int, ...): The shrink factors for the image pyramid (default: (2, 2, 1))).
        smoothing_sigmas (Tuple[float, ...]): The smoothing sigmas (default: (2, 1, 0))).
        sampling_percentage (float): The sampling percentage of the voxels to incorporate into the optimization
         (default: 0.2).
        resampling_interpolator (int): The resampling interpolator (default: sitk.sitkBSpline).
        deterministic (bool): Deterministic processing with a fixed seed and a single thread (default: True).
    """

    def __init__(
        self,
        reference_modality: Union[Modality, str],
        registration_type: RegistrationType = RegistrationType.RIGID,
        number_of_histogram_bins: int = 200,
        learning_rate: float = 1.0,
        step_size: float = 0.001,
        number_of_iterations: int = 1500,
        relaxation_factor: float = 0.5,
        shrink_factors: Tuple[int, ...] = (2, 2, 1),
        smoothing_sigmas: Tuple[float, ...] = (2, 1, 0),
        sampling_percentage: float = 0.2,
        resampling_interpolator: int = sitk.sitkBSpline,
        deterministic: bool = True,
    ) -> None:
        super().__init__()

        if len(shrink_factors) != len(smoothing_sigmas):
            raise ValueError("The shrink_factors and smoothing_sigmas need to have the same length!")

        self.reference_modality: Modality = str_to_modality(reference_modality)
        self.registration_type = registration_type
        self.number_of_histogram_bins = number_of_histogram_bins
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.number_of_iterations = number_of_iterations
        self.relaxation_factor = relaxation_factor
        self.shrink_factors: Tuple[int, ...] = shrink_factors
        self.smoothing_sigmas: Tuple[float, ...] = smoothing_sigmas
        self.sampling_percentage = sampling_percentage
        self.resampling_interpolator = resampling_interpolator
        self.deterministic = deterministic


class IntraSubjectRegistrationFilter(Filter):
    """An invertible intra-subject registration filter class which registers all
    :class:`~pyradise.data.image.IntensityImage` instances to a reference :class:`~pyradise.data.image.IntensityImage`
    instance.

    Important:
        This filter assumes that the :class:`~pyradise.data.image.SegmentationImage` instances are already registered
        to the reference :class:`~pyradise.data.image.IntensityImage` instance. No transformation will be applied to
        the :class:`~pyradise.data.image.SegmentationImage` instances.

    Warning:
        The inverse registration procedure may not yield the expected results if successive
        :class:`~pyradise.process.base.Filter` s are applied to the same :class:`~pyradise.data.image.Image` instances.
        Thus, it's recommended to use the invertibility feature with appropriate caution.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Return whether the filter is invertible or not.

        Returns:
            bool: True because the registration filter is invertible.
        """
        return True

    # noinspection DuplicatedCode
    def _process_image(
        self,
        moving_image: Image,
        fixed_image: sitk.Image,
        params: IntraSubjectRegistrationFilterParams,
        transform: Optional[sitk.Transform] = None,
        track_infos: bool = True,
    ) -> Image:
        """Apply the transformation or register the image to the reference image.

        Args:
            moving_image (Image): The moving image.
            fixed_image (sitk.Image): The fixed image.
            params (IntraSubjectRegistrationFilterParams): The filter parameters.
            transform (Optional[sitk.Transform]): The transformation to apply to the image (default: None).
            track_infos (bool): Whether to track the processing information or not (default: True).

        Returns:
            Image: The registered image.
        """
        # get the moving image as SimpleITK image
        moving_image_sitk = moving_image.get_image_data()

        # cast the image if its pixels are not of type float32
        if isinstance(moving_image, IntensityImage):
            moving_image_sitk = sitk.Cast(moving_image_sitk, sitk.sitkFloat32)
            fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)

        # register the moving image to the fixed image if no transform is given
        if transform is None:
            # get the registration method
            registration_method = get_registration_method(
                params.registration_type,
                params.number_of_histogram_bins,
                params.learning_rate,
                params.step_size,
                params.number_of_iterations,
                params.relaxation_factor,
                params.shrink_factors,
                params.smoothing_sigmas,
                params.sampling_percentage,
                params.deterministic,
            )

            transform = register_images(moving_image_sitk, fixed_image, params.registration_type, registration_method)

        # get the interpolator according to the image type
        interpolator = get_interpolator(moving_image)
        if interpolator is None:
            return moving_image

        # resample the moving image
        min_intensity = float(np.min(sitk.GetArrayFromImage(moving_image_sitk)))
        new_image_sitk = sitk.Resample(
            moving_image_sitk, fixed_image, transform, interpolator, min_intensity, moving_image_sitk.GetPixelIDValue()
        )

        # set the new image data to the image
        moving_image.set_image_data(new_image_sitk)

        # track the necessary information
        if track_infos:
            self.tracking_data.update(
                {
                    "original_origin": moving_image_sitk.GetOrigin(),
                    "original_spacing": moving_image_sitk.GetSpacing(),
                    "original_direction": moving_image_sitk.GetDirection(),
                    "original_size": moving_image_sitk.GetSize(),
                }
            )
            self._register_tracked_data(moving_image, moving_image_sitk, new_image_sitk, params, transform)

        return moving_image

    def execute(self, subject: Subject, params: IntraSubjectRegistrationFilterParams) -> Subject:
        """Execute the intra-subject registration procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            params (IntraSubjectRegistrationFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with registered
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        # get the reference image
        reference_image = subject.get_image_by_modality(params.reference_modality)
        reference_image_sitk = reference_image.get_image_data()

        # perform the registration
        for image in subject.get_images():
            if isinstance(image, IntensityImage):
                if image.get_modality() == params.reference_modality:
                    continue

                self._process_image(image, reference_image_sitk, params, track_infos=True)

        return subject

    # noinspection DuplicatedCode
    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Execute the inverse of the intra-subject registration procedure.

        Args:
            subject: The :class:`~pyradise.data.subject.Subject` instance to be processed.
            transform_info: The transform information.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with unregistered
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        # construct the original image as a reference
        original_image_props = transform_info.get_image_properties(pre_transform=True)

        reference_image_np = np.zeros(original_image_props.size[::-1], dtype=float)
        reference_image_sitk = sitk.GetImageFromArray(reference_image_np)
        reference_image_sitk.SetOrigin(original_image_props.origin)
        reference_image_sitk.SetSpacing(original_image_props.spacing)
        reference_image_sitk.SetDirection(original_image_props.direction)

        # get the inverse transform
        transform = transform_info.get_transform(True)

        # perform the inverse registration
        for image in subject.get_images():
            if target_image is not None and image != target_image:
                continue

            if isinstance(image, IntensityImage):
                if image.get_modality() == transform_info.params.reference_modality:
                    continue

                self._process_image(
                    image, reference_image_sitk, transform_info.get_params(), transform, track_infos=False
                )

        return subject


# pylint: disable = too-few-public-methods
class InterSubjectRegistrationFilterParams(FilterParams):
    """A filter parameter class for the :class:`~pyradise.process.registration.InterSubjectRegistrationFilter` class.

    Args:
        reference_subject (Subject): The reference subject to which the subject will be registered.
        reference_modality (Union[Modality, str]): The modality of the reference image (fixed image) to be used for
         registration.
        subject_modality (Optional[Union[Modality, str]]): The modality of the subject image (moving image) to be used
         for registration. If ``None``, the same modality as the reference image will be used (default: None).
        registration_type (RegistrationType): The type of registration (default: RegistrationType.RIGID).
        number_of_histogram_bins (int): The number of histogram bins for registration (default: 200).
        learning_rate (float): The learning rate of the optimizer (default: 1.0).
        step_size (float): The step size of the optimizer (default: 0.001).
        number_of_iterations (int): The maximal number of optimization iterations (default: 1500).
        relaxation_factor (float): The relaxation factor (default: 0.5).
        shrink_factors (Tuple[int, ...): The shrink factors for the image pyramid (default: (2, 2, 1))).
        smoothing_sigmas (Tuple[float, ...]): The smoothing sigmas (default: (2, 1, 0))).
        sampling_percentage (float): The sampling percentage of the voxels to incorporate into the optimization
         (default: 0.2).
        resampling_interpolator (int): The interpolator to use for resampling the image.
        deterministic (bool): Deterministic processing with a fixed seed and a single thread (default: True).
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(
        self,
        reference_subject: Subject,
        reference_modality: Union[Modality, str],
        subject_modality: Optional[Union[Modality, str]] = None,
        registration_type: RegistrationType = RegistrationType.RIGID,
        number_of_histogram_bins: int = 200,
        learning_rate: float = 1.0,
        step_size: float = 0.001,
        number_of_iterations: int = 1500,
        relaxation_factor: float = 0.5,
        shrink_factors: Tuple[int, ...] = (2, 2, 1),
        smoothing_sigmas: Tuple[float, ...] = (2, 1, 0),
        sampling_percentage: float = 0.2,
        resampling_interpolator: int = sitk.sitkBSpline,
        deterministic: bool = True,
    ) -> None:
        super().__init__()

        if len(shrink_factors) != len(smoothing_sigmas):
            raise ValueError("The shrink_factors and smoothing_sigmas need to have the same length!")

        self.reference_subject = reference_subject
        self.reference_modality: Modality = str_to_modality(reference_modality)
        self.subject_modality: Modality = (
            str_to_modality(subject_modality) if subject_modality is not None else reference_modality
        )
        self.registration_type = registration_type
        self.number_of_histogram_bins = number_of_histogram_bins
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.number_of_iterations = number_of_iterations
        self.relaxation_factor = relaxation_factor
        self.shrink_factors: Tuple[int, ...] = shrink_factors
        self.smoothing_sigmas: Tuple[float, ...] = smoothing_sigmas
        self.sampling_percentage = sampling_percentage
        self.resampling_interpolator = resampling_interpolator
        self.deterministic = deterministic


class InterSubjectRegistrationFilter(Filter):
    """An invertible inter-subject registration filter class which registers all
    :class:`~pyradise.data.image.IntensityImage` instances of the provided :class:`~pyradise.data.subject.Subject` to a
    reference :class:`~pyradise.data.image.IntensityImage` instance of another :class:`~pyradise.data.subject.Subject`.

    Important:
        This filter assumes that all :class:`~pyradise.data.image.Image` instances of the provided
        :class:`~pyradise.data.subject.Subject` are co-registered such that the
        :class:`~pyradise.data.image.SegmentationImage` instances do not require special treatment.

    Warning:
        The inverse registration procedure may not yield the expected results if successive
        :class:`~pyradise.process.base.Filter` s are applied to the same :class:`~pyradise.data.image.Image` instances.
        Thus, it's recommended to use the invertibility feature with appropriate caution.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Return whether the filter is invertible.

        Returns:
            bool: True because the inter-subject registration is invertible.
        """
        return True

    # noinspection DuplicatedCode
    def _apply_transform(
        self,
        subject: Subject,
        transform: sitk.Transform,
        reference_image: sitk.Image,
        params: InterSubjectRegistrationFilterParams,
    ) -> Subject:
        """Apply the provided transformation to the subject.

        Args:
            subject (Subject): The subject.
            transform (sitk.Transform): The transformation to apply to the subject.
            reference_image (sitk.Image): The reference image.
            params (InterSubjectRegistrationFilterParams): The filters parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with transformed
            :class:`~pyradise.data.image.Image` instances.
        """
        # transform and resample the images
        for image in subject.get_images():
            interpolator = get_interpolator(image)
            if interpolator is None:
                continue

            # get the image data and cast if necessary
            image_sitk = image.get_image_data()
            if isinstance(image, IntensityImage):
                image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)

            # resample the image
            min_intensity = float(np.min(sitk.GetArrayFromImage(image_sitk)))
            new_image_sitk = sitk.Resample(
                image_sitk, reference_image, transform, interpolator, min_intensity, image_sitk.GetPixelIDValue()
            )

            # set the new image data to the image
            image.set_image_data(new_image_sitk)

            # track the necessary data
            self._register_tracked_data(image, image_sitk, new_image_sitk, params, transform)

        return subject

    # noinspection DuplicatedCode
    @staticmethod
    def _apply_inverse_transform(
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Apply the inverse transformation to the subject.

        Args:
            subject (Subject): The subject.
            transform_info (TransformInfo): The transformation information.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with back-transformed
            :class:`~pyradise.data.image.Image` instances.
        """
        # construct the original image as a reference
        original_image_props = transform_info.get_image_properties(pre_transform=True)

        reference_image_np = np.zeros(original_image_props.size[::-1], dtype=float)
        reference_image_sitk = sitk.GetImageFromArray(reference_image_np)
        reference_image_sitk.SetOrigin(original_image_props.origin)
        reference_image_sitk.SetSpacing(original_image_props.spacing)
        reference_image_sitk.SetDirection(original_image_props.direction)

        # get the inverse transform
        transform = transform_info.get_transform(True)

        # transform and resample the images
        for image in subject.get_images():
            if target_image is not None and image != target_image:
                continue

            # get the image data and cast if necessary
            image_sitk = image.get_image_data()
            if isinstance(image, IntensityImage):
                image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)

            # the interpolator
            interpolator = get_interpolator(image)
            if interpolator is None:
                continue

            # resample the image
            min_intensity = float(np.min(sitk.GetArrayFromImage(image_sitk)))
            new_image_sitk = sitk.Resample(
                image_sitk, reference_image_sitk, transform, interpolator, min_intensity, image_sitk.GetPixelIDValue()
            )

            # set the new image data to the image
            image.set_image_data(new_image_sitk)

        return subject

    @staticmethod
    def _register_image(
        subject: Subject, reference_image: sitk.Image, params: InterSubjectRegistrationFilterParams
    ) -> sitk.Transform:
        """Register the subject image to the specific modality of the reference subject.

        Args:
            subject (Subject): The subject to register.
            reference_image (sitk.Image): The reference image.
            params (InterSubjectRegistrationFilterParams): The filters parameters.

        Returns:
            sitk.Transform: The registration transformation.
        """
        moving_image = subject.get_image_by_modality(params.subject_modality)
        moving_image_sitk = moving_image.get_image_data()

        # get the registration method
        registration_method = get_registration_method(
            params.registration_type,
            params.number_of_histogram_bins,
            params.learning_rate,
            params.step_size,
            params.number_of_iterations,
            params.relaxation_factor,
            params.shrink_factors,
            params.smoothing_sigmas,
            params.sampling_percentage,
            params.deterministic,
        )

        return register_images(moving_image_sitk, reference_image, params.registration_type, registration_method)

    def execute(self, subject: Subject, params: InterSubjectRegistrationFilterParams) -> Subject:
        """Executes the inter-subject registration procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            params (InterSubjectRegistrationFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with all
            :class:`~pyradise.data.image.IntensityImage` instances registered to the reference subject
            :class:`~pyradise.data.image.IntensityImage` instance.
        """
        # get the reference image
        reference_image = params.reference_subject.get_image_by_modality(params.reference_modality)
        reference_image_sitk = reference_image.get_image_data()

        # register the subject to the reference image
        transform = self._register_image(subject, reference_image_sitk, params)

        # apply the transform to the other images of the subject
        subject = self._apply_transform(subject, transform, reference_image_sitk, params)

        return subject

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Execute the inverse of the inter-subject registration procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            transform_info (TransformInfo): The transform information.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with unregistered
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        subject = self._apply_inverse_transform(subject, transform_info, target_image)

        return subject
