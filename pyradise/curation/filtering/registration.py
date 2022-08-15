from typing import Tuple
from enum import Enum

import numpy as np
import SimpleITK as sitk

from pyradise.data import (
    Subject,
    Modality,
    TransformationInformation)
from .base import (
    Filter,
    FilterParameters)


class RegistrationType(Enum):
    """An enum class representing the different types of registration."""
    AFFINE = 1
    SIMILARITY = 2
    RIGID = 3
    BSPLINE = 4


# pylint: disable = too-few-public-methods
class ReferenceSubjectRegistrationFilterParameters(FilterParameters):
    """A class representing the parameters for the ReferenceSubjectRegistrationFilter."""

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(self,
                 reference_subject: Subject,
                 reference_modality: Modality,
                 registration_type: RegistrationType = RegistrationType.RIGID,
                 number_of_histogram_bins: int = 200,
                 learning_rate: float = 1.0,
                 step_size: float = 0.001,
                 number_of_iterations: int = 1500,
                 relaxation_factor: float = 0.5,
                 shrink_factors: Tuple[int] = (2, 2, 1),
                 smoothing_sigmas: Tuple[float] = (2, 1, 0),
                 sampling_percentage: float = 0.2,
                 resampling_interpolator = sitk.sitkBSpline,
                 ) -> None:
        """Represents the parameters for a ReferenceSubjectRegistrationFilter.

        Args:
            reference_subject (Subject): The reference subject.
            reference_modality (Modality): The reference modality.
            registration_type (RegistrationType): The type of registration (default=RegistrationType.RIGID).
            number_of_histogram_bins (int): The number of histogram bins (default=200).
            learning_rate (float): The learning rate of the optimizer (default=1.0).
            step_size (float): The minimum step size of the optimizer (default=0.001).
            number_of_iterations (int): The maximum number of iterations (default=1500).
            relaxation_factor (float): The relaxation factor for the optimizer (default=0.5).
            shrink_factors ([int]): The shrink factors for the resolution pyramid (default=(2, 2, 1)).
            smoothing_sigmas ([float]): The smoothing sigmas for the resolution pyramid (default=(2, 1, 0)).
            sampling_percentage (float): The sampling percentage (0 - 1) (default=0.2).
            resampling_interpolator (int): The interpolation function type (default=sitk.sitkBSpline).
        """
        super().__init__()

        if len(shrink_factors) != len(smoothing_sigmas):
            raise ValueError("The shrink_factors and smoothing_sigmas need to have the same length!")

        self.reference_subject = reference_subject
        self.reference_modality = reference_modality
        self.registration_type = registration_type
        self.number_of_histogram_bins = number_of_histogram_bins
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.number_of_iterations = number_of_iterations
        self.relaxation_factor = relaxation_factor
        self.shrink_factors = shrink_factors
        self.smoothing_sigmas = smoothing_sigmas
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


class ReferenceSubjectRegistrationFilter(Filter):
    """A class for registering a subject to a specific modality of another subject."""

    def _apply_transform(self,
                         subject: Subject,
                         transform: sitk.Transform,
                         params: ReferenceSubjectRegistrationFilterParameters
                         ) -> Subject:
        """Apply the transformation to the subject.

        Args:
            subject (Subject): The subject to apply the transformation.
            transform (sitk.Transform): The transformation to apply to the subject.
            params (ReferenceSubjectRegistrationFilterParameters): The filters parameters.

        Returns:
            Subject: The transformed subject.
        """
        reference_image = params.reference_subject.get_image_by_modality(params.reference_modality)
        reference_image_sitk = reference_image.get_image(as_sitk=True)

        interpolator = sitk.sitkBSpline
        for image in subject.intensity_images:
            image_sitk = image.get_image(as_sitk=True)
            min_intensity_value = float(np.min(sitk.GetArrayFromImage(image_sitk)))
            resampled_image_sitk = sitk.Resample(image_sitk, reference_image_sitk, transform, interpolator,
                                                 min_intensity_value, image_sitk.GetPixelIDValue())
            image.set_image(resampled_image_sitk)

            transform_info = TransformationInformation.from_images(self.__class__.__name__, transform,
                                                                   image_sitk, resampled_image_sitk)
            image.get_transform_tape().record(transform_info)

        interpolator = sitk.sitkNearestNeighbor
        for image in subject.segmentation_images:
            image_sitk = image.get_image(as_sitk=True)
            resampled_image_sitk = sitk.Resample(image_sitk, reference_image_sitk, transform, interpolator, 0.0,
                                                 image_sitk.GetPixelIDValue())
            image.set_image(resampled_image_sitk)

            transform_info = TransformationInformation.from_images(self.__class__.__name__, transform,
                                                                   image_sitk, resampled_image_sitk)
            image.get_transform_tape().record(transform_info)

        return subject

    def _register_image(self,
                        subject: Subject,
                        params: ReferenceSubjectRegistrationFilterParameters
                        ) -> sitk.Transform:
        """Registers the subject to the specific modality of the reference subject.

        Args:
            subject (Subject): The subject to register.
            params (ReferenceSubjectRegistrationFilterParameters): The filters parameters.

        Returns:
            sitk.Transform: The registration transformation.
        """
        floating_image = subject.get_image_by_modality(params.reference_modality)
        floating_image_sitk = floating_image.get_image(as_sitk=True)

        fixed_image = params.reference_subject.get_image_by_modality(params.reference_modality)
        fixed_image_sitk = fixed_image.get_image(as_sitk=True)

        assert floating_image_sitk.GetDimension() == fixed_image_sitk.GetDimension(), \
            'The number of dimensions must be equal for the floating and the fixed image!'

        dimensions = floating_image_sitk.GetDimension()

        assert dimensions in (2, 3), f'The number of image dimensions must be 2 or 3, but is {dimensions}!'

        if params.registration_type == RegistrationType.BSPLINE:
            transform_domain_mesh_size = [10] * dimensions
            initial_transform = sitk.BSplineTransformInitializer(fixed_image_sitk, transform_domain_mesh_size)
        else:
            if params.registration_type == RegistrationType.RIGID:
                transform_type = sitk.VersorRigid3DTransform() if dimensions == 3 else sitk.Euler2DTransform()

            elif params.registration_type == RegistrationType.AFFINE:
                transform_type = sitk.AffineTransform(dimensions)

            elif params.registration_type == RegistrationType.SIMILARITY:
                transform_type = sitk.Similarity3DTransform() if dimensions == 3 else sitk.Similarity2DTransform()

            else:
                raise ValueError(f'The registration_type ({params.registration_type.name}) is not supported!')

            initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(fixed_image_sitk,
                                                                            floating_image_sitk.GetPixelIDValue()),
                                                                  floating_image_sitk,
                                                                  transform_type,
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)

        params.registration.SetInitialTransform(initial_transform, inPlace=True)

        transform = params.registration.Execute(sitk.Cast(fixed_image_sitk, sitk.sitkFloat32),
                                                sitk.Cast(floating_image_sitk, sitk.sitkFloat32))

        if self.verbose:
            print(f'{self.__class__.__name__}: \n\t'
                  f'Final metric value: {params.registration.GetMetricValue():.5f} \n\t'
                  f'Optimizer\'s stopping criterion: {params.registration.GetOptimizerStopConditionDescription()}')

        if params.number_of_iterations == params.registration.GetOptimizerIteration():
            print(f'{self.__class__.__name__}: '
                  f'Optimizer terminated without convergence on subject {subject.get_name()}!')

        return transform

    def execute(self,
                subject: Subject,
                params: ReferenceSubjectRegistrationFilterParameters
                ) -> Subject:
        """Executes the reference subject registration procedure.

        Args:
            subject (Subject): The subject to be processed.
            params (ReferenceSubjectRegistrationFilterParameters): The filter parameters.

        Returns:
            Subject: The processed subject.
        """
        if not params:
            raise ValueError(f'The parameters for {self.__class__.__name__} are missing!')

        transform = self._register_image(subject, params)

        subject = self._apply_transform(subject, transform, params)

        return subject
