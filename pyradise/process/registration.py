from enum import Enum
from typing import (
    Tuple,
    Optional)

import numpy as np
import SimpleITK as sitk

from pyradise.data import (
    Subject,
    Modality,
    Image,
    IntensityImage,
    SegmentationImage,
    TransformInfo)
from .base import (
    Filter,
    FilterParams)

__all__ = ['IntraSubjectRegistrationFilter', 'IntraSubjectRegistrationFilterParams',
           'InterSubjectRegistrationFilter', 'InterSubjectRegistrationFilterParams', 'RegistrationType']


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


def register_images(moving_image: sitk.Image,
                    fixed_image: sitk.Image,
                    registration_type: RegistrationType,
                    registration_method: sitk.ImageRegistrationMethod
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
            raise ValueError(f'The registration type ({registration_type.name}) is not supported!')

        initial_transform = sitk.CenteredTransformInitializer(fixed_image_f32, moving_image_f32, transform_type,
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method.SetInitialTransform(initial_transform, inPlace=True)

    transform = registration_method.Execute(fixed_image_f32, moving_image_f32)

    return transform


# pylint: disable = too-few-public-methods
class InterSubjectRegistrationFilterParams(FilterParams):
    """A class representing the parameters for the :class:`InterSubjectRegistrationFilter`.

    Args:
        reference_subject (Subject): The reference subject to which the subject will be registered.
        reference_modality (Modality): The modality of the reference image (fixed image) to be used for registration.
        subject_modality (Optional[Modality]): The modality of the subject image (moving image) to be used for
         registration. If ``None``, the same modality as the reference image will be used (default: None).
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
        resampling_interpolator (int): The interpolator to use for resampling the image
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(self,
                 reference_subject: Subject,
                 reference_modality: Modality,
                 subject_modality: Optional[Modality] = None,
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
                 ) -> None:
        super().__init__()

        if len(shrink_factors) != len(smoothing_sigmas):
            raise ValueError("The shrink_factors and smoothing_sigmas need to have the same length!")

        self.reference_subject = reference_subject
        self.reference_modality = reference_modality
        self.subject_modality: Modality = subject_modality if subject_modality is not None else reference_modality
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

        registration = sitk.ImageRegistrationMethod()

        registration.SetMetricAsMattesMutualInformation(self.number_of_histogram_bins)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(self.sampling_percentage, sitk.sitkWallClock)

        registration.SetMetricUseFixedImageGradientFilter(False)
        registration.SetMetricUseMovingImageGradientFilter(False)

        registration.SetInterpolator(sitk.sitkLinear)

        if self.registration_type == RegistrationType.BSPLINE:
            registration.SetOptimizerAsLBFGSB()
        else:
            registration.SetOptimizerAsRegularStepGradientDescent(learningRate=self.learning_rate,
                                                                  minStep=self.step_size,
                                                                  numberOfIterations=self.number_of_iterations,
                                                                  relaxationFactor=self.relaxation_factor,
                                                                  gradientMagnitudeTolerance=1e-4,
                                                                  estimateLearningRate=registration.EachIteration,
                                                                  maximumStepSizeInPhysicalUnits=0.0)

        registration.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework
        registration.SetShrinkFactorsPerLevel(self.shrink_factors)
        registration.SetSmoothingSigmasPerLevel(self.smoothing_sigmas)
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        self.registration = registration


class InterSubjectRegistrationFilter(Filter):
    """A registration filter class for inter-subject registration. This filter registers a specified subject image to
    a reference image from another subject. After registration, the retrieved transform is applied also to the other
    images of the subject.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Return whether the filter is invertible.

        Returns:
            bool: True because the filter is invertible.
        """
        return True

    # noinspection DuplicatedCode
    def _apply_transform(self,
                         subject: Subject,
                         transform: sitk.Transform,
                         reference_image: sitk.Image,
                         params: InterSubjectRegistrationFilterParams,
                         track_infos: bool
                         ) -> Subject:
        """Apply the transformation to the subject.

        Args:
            subject (Subject): The subject to apply the transformation.
            transform (sitk.Transform): The transformation to apply to the subject.
            reference_image (sitk.Image): The reference image.
            params (InterSubjectRegistrationFilterParams): The filters parameters.
            track_infos (bool): Whether to track the transformation information.

        Returns:
            Subject: The transformed subject.
        """
        # transform and resample the images
        for image in subject.get_images():
            interpolator = get_interpolator(image)
            if interpolator is None:
                continue

            # resample the image data and add the data to the image
            image_sitk = image.get_image_data(as_sitk=True)
            if isinstance(image, IntensityImage):
                image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)

            min_intensity = float(np.min(sitk.GetArrayFromImage(image_sitk)))
            new_image_sitk = sitk.Resample(image_sitk, reference_image, transform, interpolator,
                                           min_intensity, image_sitk.GetPixelIDValue())
            image.set_image_data(new_image_sitk)

            # track the properties of the original image
            if track_infos:
                self.tracking_data.update({'original_origin': image_sitk.GetOrigin(),
                                           'original_spacing': image_sitk.GetSpacing(),
                                           'original_direction': image_sitk.GetDirection(),
                                           'original_size': image_sitk.GetSize(),
                                           'transform': transform})

                # create the transform info
                transform_info = self._create_transform_info(image_sitk, new_image_sitk, params,
                                                             self.filter_args, self.tracking_data)
                image.add_transform_info(transform_info)
                self.tracking_data.clear()

        return subject

    # noinspection DuplicatedCode
    @staticmethod
    def _apply_inverse_transform(subject: Subject,
                                 transform_info: TransformInfo,
                                 ) -> Subject:
        """Apply the inverse transformation to the subject.

        Args:
            subject (Subject): The subject to apply the transformation.
            transform_info (TransformInfo): The transformation information.

        Returns:
            Subject: The inversely transformed subject.
        """
        # construct the original image as a reference
        image_np = np.zeros(transform_info.get_data('original_size')).astype(np.float)
        reference_image = sitk.GetImageFromArray(image_np)
        reference_image.SetOrigin(transform_info.get_data('original_origin'))
        reference_image.SetSpacing(transform_info.get_data('original_spacing'))
        reference_image.SetDirection(transform_info.get_data('original_direction'))

        # get the transform
        transform = transform_info.get_data('transform').GetInverse()

        # transform and resample the images
        for image in subject.get_images():

            interpolator = get_interpolator(image)
            if interpolator is None:
                continue

            image_sitk = image.get_image_data(as_sitk=True)
            min_intensity = float(np.min(sitk.GetArrayFromImage(image_sitk)))
            new_image_sitk = sitk.Resample(image_sitk, reference_image, transform, interpolator,
                                           min_intensity, image_sitk.GetPixelIDValue())
            image.set_image_data(new_image_sitk)

        return subject

    @staticmethod
    def _register_image(subject: Subject,
                        reference_image: sitk.Image,
                        params: InterSubjectRegistrationFilterParams
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
        moving_image_sitk = moving_image.get_image_data(as_sitk=True)

        return register_images(moving_image_sitk, reference_image, params.registration_type, params.registration)

    def execute(self,
                subject: Subject,
                params: InterSubjectRegistrationFilterParams
                ) -> Subject:
        """Executes the reference subject registration procedure.

        Args:
            subject (Subject): The subject to be processed.
            params (InterSubjectRegistrationFilterParams): The filter parameters.

        Returns:
            Subject: The processed subject.
        """
        # get the reference image
        reference_image = params.reference_subject.get_image_by_modality(params.reference_modality)
        reference_image_sitk = reference_image.get_image_data(as_sitk=True)

        # register the subject to the reference image
        transform = self._register_image(subject, reference_image_sitk, params)
        self.tracking_data.update({'transform': transform})

        subject = self._apply_transform(subject, transform, reference_image_sitk, params, track_infos=True)

        return subject

    def execute_inverse(self,
                        subject: Subject,
                        transform_info: TransformInfo
                        ) -> Subject:
        """Executes the inverse registration procedure.

        Args:
            subject (Subject): The subject to be processed.
            transform_info (TransformInfo): The transform info.

        Returns:
            Subject: The registered subject.
        """
        transform = transform_info.get_data('transform')
        if transform is None:
            raise ValueError('The transform is not available in the transform info!')

        subject = self._apply_inverse_transform(subject, transform_info)

        return subject


class IntraSubjectRegistrationFilterParams(FilterParams):

    def __init__(self,
                 reference_modality: Modality,
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
                 ) -> None:
        super().__init__()

        if len(shrink_factors) != len(smoothing_sigmas):
            raise ValueError("The shrink_factors and smoothing_sigmas need to have the same length!")

        self.reference_modality = reference_modality
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

        registration = sitk.ImageRegistrationMethod()

        registration.SetMetricAsMattesMutualInformation(self.number_of_histogram_bins)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(self.sampling_percentage, sitk.sitkWallClock)

        registration.SetMetricUseFixedImageGradientFilter(False)
        registration.SetMetricUseMovingImageGradientFilter(False)

        registration.SetInterpolator(sitk.sitkLinear)

        if self.registration_type == RegistrationType.BSPLINE:
            registration.SetOptimizerAsLBFGSB()
        else:
            registration.SetOptimizerAsRegularStepGradientDescent(learningRate=self.learning_rate,
                                                                  minStep=self.step_size,
                                                                  numberOfIterations=self.number_of_iterations,
                                                                  relaxationFactor=self.relaxation_factor,
                                                                  gradientMagnitudeTolerance=1e-4,
                                                                  estimateLearningRate=registration.EachIteration,
                                                                  maximumStepSizeInPhysicalUnits=0.0)

        registration.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework
        registration.SetShrinkFactorsPerLevel(self.shrink_factors)
        registration.SetSmoothingSigmasPerLevel(self.smoothing_sigmas)
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        self.registration = registration


class IntraSubjectRegistrationFilter(Filter):
    """An intra-subject registration filter class which registers all :class:`~pyradise.data.image.IntensityImage`
    instances to a reference :class:`~pyradise.data.image.IntensityImage` instance.

    Warning:
        This filter assumes that the :class:`~pyradise.data.image.SegmentationImage` instances are already registered
        to the reference :class:`~pyradise.data.image.IntensityImage` instance. No transformation will be applied to
        the :class:`~pyradise.data.image.SegmentationImage` instances.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Returns whether the filter is invertible or not.

        Returns:
            bool: True because the registration filter is invertible.
        """
        return True

    # noinspection DuplicatedCode
    def _register_image(self,
                        moving_image: Image,
                        fixed_image: sitk.Image,
                        params: IntraSubjectRegistrationFilterParams,
                        transform: Optional[sitk.Transform] = None
                        ) -> Image:
        moving_image_sitk = moving_image.get_image_data(as_sitk=True)

        if isinstance(moving_image, IntensityImage):
            moving_image_sitk = sitk.Cast(moving_image_sitk, sitk.sitkFloat32)
            fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)

        if transform is None:
            transform = register_images(moving_image_sitk, fixed_image, params.registration_type,
                                        params.registration)

        interpolator = get_interpolator(moving_image)
        if interpolator is None:
            return moving_image

        min_intensity = float(np.min(sitk.GetArrayFromImage(moving_image_sitk)))
        new_image_sitk = sitk.Resample(moving_image_sitk, fixed_image, transform, interpolator,
                                       min_intensity, moving_image_sitk.GetPixelIDValue())
        moving_image.set_image_data(new_image_sitk)

        # tracking the information
        self.tracking_data.update({'original_origin': moving_image_sitk.GetOrigin(),
                                   'original_spacing': moving_image_sitk.GetSpacing(),
                                   'original_direction': moving_image_sitk.GetDirection(),
                                   'original_size': moving_image_sitk.GetSize(),
                                   'transform': transform})

        # create the transform info
        transform_info = self._create_transform_info(moving_image_sitk, new_image_sitk, params,
                                                     self.filter_args, self.tracking_data)
        moving_image.add_transform_info(transform_info)
        self.tracking_data.clear()

        return moving_image

    def execute(self,
                subject: Subject,
                params: IntraSubjectRegistrationFilterParams
                ) -> Subject:
        # get the reference image
        reference_image = subject.get_image_by_modality(params.reference_modality)
        reference_image_sitk = reference_image.get_image_data(as_sitk=True)

        # perform the registration
        for image in subject.get_images():
            if isinstance(image, IntensityImage):
                if image.get_modality() == params.reference_modality:
                    continue

                self._register_image(image, reference_image_sitk, params)

        return subject

    # noinspection DuplicatedCode
    def execute_inverse(self,
                        subject: Subject,
                        transform_info: TransformInfo
                        ) -> Subject:
        """Execute the inverse of the registration procedure.

        Args:
            subject: The subject to be transformed.
            transform_info: The transform info to be used for the inverse transformation.

        Returns:
            Subject: The transformed subject.
        """
        # construct the original image as a reference
        image_np = np.zeros(transform_info.get_data('original_size')).astype(np.float)
        reference_image = sitk.GetImageFromArray(image_np)
        reference_image.SetOrigin(transform_info.get_data('original_origin'))
        reference_image.SetSpacing(transform_info.get_data('original_spacing'))
        reference_image.SetDirection(transform_info.get_data('original_direction'))

        # get the transform
        transform = transform_info.get_data('transform').GetInverse()

        # perform the inverse registration
        for image in subject.get_images():
            if isinstance(image, IntensityImage):
                if image.get_modality() == transform_info.params.reference_modality:
                    continue

                self._register_image(image, reference_image, transform_info.params, transform)

        return subject
