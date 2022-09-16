from typing import (
    Tuple,
    Optional)

import numpy as np
import SimpleITK as sitk
import itk

from pyradise.data import (
    Subject,
    Modality,
    TransformationInformation,
    SegmentationImage,
    IntensityImage)
from .base import (
    Filter,
    FilterParameters)


# pylint: disable = too-few-public-methods
class ResamplingFilterParameters(FilterParameters):
    """A class representing the parameters of the ResamplingFilter."""

    def __init__(self,
                 output_size: Optional[Tuple[int, ...]],
                 output_spacing: Optional[Tuple[float, ...]],
                 reference_modality: Modality,
                 transform: sitk.Transform = sitk.AffineTransform(3),
                 centering_method: str = 'none',
                 rescaling_intensity_images: bool = True
                 ) -> None:
        """Specifies the parameters for a ResamplingFilter.

        Args:
            output_size (Optional[Tuple[int, ...]]) : The output size of the image.
            output_spacing (Optional[Tuple[float, ...]]): The output spacing of the image.
            reference_modality (Modality): The reference modality.
            transform (sitk.Transform): The transformation applied during resampling
             (default: sitk.AffineTransform(3)).
            centering_method (str): The method to center the image (options: 'none', 'reference', 'label_moment')
             (default: 'none').
            rescaling_intensity_images (bool): If true the intensity images will be automatically rescaled
             (default: True).
        """
        super().__init__()
        self.output_size = output_size
        self.output_spacing = output_spacing
        self.reference_modality = reference_modality
        self.transform = transform
        self.centering_method = centering_method
        self.rescaling_intensity_images = rescaling_intensity_images


class ResamplingFilter(Filter):
    """A class representing a resampling filter."""

    @staticmethod
    def compute_label_center(images: Tuple[SegmentationImage, ...]) -> np.ndarray:
        """Computes the label center.

        Args:
            images (Tuple[SegmentationImage, ...]): The segmentation images used for calculating the label center.

        Returns:
            np.ndarray: An array containing the label center.
        """
        bounding_boxes = []
        for image in images:
            origin = image.get_origin()
            distant_point = tuple(image.get_image().TransformIndexToPhysicalPoint(image.get_size()))
            bounding_boxes.append((*origin, *distant_point))

        n_dims = images[0].get_dimensions()
        lower_dim_limits = []
        upper_dim_limits = []
        for i in range(n_dims):
            sorted_dims = tuple(zip(*bounding_boxes))
            dim_limits = sorted_dims[i] + sorted_dims[i + n_dims]
            lower_dim_limits.append(min(dim_limits))
            upper_dim_limits.append(max(dim_limits))

        center = [(maximum + minimum) / 2 for maximum, minimum in zip(upper_dim_limits, lower_dim_limits)]

        return np.array(center)

    @staticmethod
    def compute_label_moment_origin(image: IntensityImage,
                                    params: ResamplingFilterParameters,
                                    label_center: np.ndarray
                                    ) -> np.ndarray:
        """Computes the label moment

        Args:
            image (IntensityImage): The intensity image used for the moment calculation.
            params (ResamplingFilterParameters): The resampling filter parameters.
            label_center (np.ndarray): The label center.

        Returns:
            np.ndarray: The moment around the center.
        """
        moment_calc = itk.ImageMomentsCalculator[itk.Image[itk.template(image)[1]]].New()
        moment_calc.SetImage(image.get_image())
        moment_calc.Compute()
        image_moment_center = np.array(moment_calc.GetCenterOfGravity())

        center = (image_moment_center + label_center) / 2

        physical_image_center = np.array(params.output_spacing) * (np.array(params.output_size) // 2)
        origin = center - np.dot(image.get_direction(), physical_image_center)

        return origin

    # noinspection DuplicatedCode
    def resample_intensity_image(self,
                                 image: IntensityImage,
                                 reference_image: IntensityImage,
                                 params: ResamplingFilterParameters,
                                 segmentation_images: Optional[Tuple[SegmentationImage, ...]] = None
                                 ) -> IntensityImage:
        """Resamples an intensity image according to the provided parameters.

        Args:
            image (IntensityImage): The intensity image to resample.
            reference_image (IntensityImage): The reference image.
            params (ResamplingFilterParameters): The parameters for the resampling.
            segmentation_images (Optional[Tuple[SegmentationImage, ...]]): Segmentation images used for the moment
             calculation (default: None).

        Returns:
            IntensityImage: The resampled intensity image.
        """
        image_sitk = image.get_image(as_sitk=True)
        reference_image_sitk = reference_image.get_image(as_sitk=True)

        image_np = sitk.GetArrayFromImage(image_sitk)
        min_intensity = np.min(image_np)

        if 'integer' in image_sitk.GetPixelIDTypeAsString():
            min_intensity = int(min_intensity)
        else:
            min_intensity = float(min_intensity)

        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetInterpolator(sitk.sitkBSpline)
        resample_filter.SetTransform(params.transform)
        resample_filter.SetDefaultPixelValue(min_intensity)

        if params.centering_method == 'none':
            resample_filter.SetOutputOrigin(image_sitk.GetOrigin())
            resample_filter.SetOutputDirection(image_sitk.GetDirection())

        elif params.centering_method == 'reference':
            resample_filter.SetOutputOrigin(reference_image_sitk.GetOrigin())
            resample_filter.SetOutputDirection(reference_image_sitk.GetDirection())

        elif params.centering_method == 'label_moment' and reference_image == image:
            if not segmentation_images:
                raise ValueError('The centering method label_moment is not available '
                                 'when no segmentations are provided!')

            label_center = self.compute_label_center(segmentation_images)
            label_moment_origin = self.compute_label_moment_origin(image, params, label_center)

            resample_filter.SetOutputOrigin(label_moment_origin)
            resample_filter.SetOutputDirection(reference_image_sitk.GetDirection())

        elif params.centering_method == 'label_moment' and reference_image != image:
            resample_filter.SetOutputOrigin(reference_image_sitk.GetOrigin())
            resample_filter.SetOutputDirection(reference_image_sitk.GetDirection())

        else:
            raise NotImplementedError(f'The centering method ({params.centering_method}) is not implemented!')

        if params.output_spacing:
            resample_filter.SetOutputSpacing(params.output_spacing)

        if params.output_size:
            resample_filter.SetSize(params.output_size)

        resampled_image_sitk = resample_filter.Execute(image_sitk)

        if params.rescaling_intensity_images:
            rescale_filter = sitk.RescaleIntensityImageFilter()
            rescale_filter.SetOutputMinimum(min_intensity)
            rescale_filter.SetOutputMaximum(np.max(image_np))
            resampled_image_sitk = rescale_filter.Execute(resampled_image_sitk)

        image.set_image(resampled_image_sitk)

        transform_info = TransformationInformation.from_images(self.__class__.__name__, params.transform, image_sitk,
                                                               resampled_image_sitk)
        image.get_transform_tape().record(transform_info)

        return image

    # noinspection DuplicatedCode
    def resample_segmentation_image(self,
                                    image: SegmentationImage,
                                    reference_image: IntensityImage,
                                    params: ResamplingFilterParameters,
                                    ) -> SegmentationImage:
        """Resamples a segmentation image according to the provided parameters.

        Args:
            image (SegmentationImage): The image to resample.
            reference_image (IntensityImage): The reference image.
            params (ResamplingFilterParameters): The parameters for the resampling.

        Returns:
            SegmentationImage: The resampled segmentation image.
        """

        image_sitk = image.get_image(as_sitk=True)
        reference_image_sitk = reference_image.get_image(as_sitk=True)

        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
        resample_filter.SetTransform(params.transform)
        resample_filter.SetDefaultPixelValue(0)

        if params.centering_method == 'none':
            resample_filter.SetOutputOrigin(image_sitk.GetOrigin())
            resample_filter.SetOutputDirection(image_sitk.GetDirection())

        elif params.centering_method in ('reference', 'label_moment'):
            resample_filter.SetOutputOrigin(reference_image_sitk.GetOrigin())
            resample_filter.SetOutputDirection(reference_image_sitk.GetDirection())

        else:
            raise NotImplementedError(f'The centering method ({params.centering_method}) is invalid!')

        if params.output_spacing:
            resample_filter.SetOutputSpacing(params.output_spacing)

        if params.output_size:
            resample_filter.SetSize(params.output_size)

        resampled_image_sitk = resample_filter.Execute(image_sitk)

        image.set_image(resampled_image_sitk)

        transform_info = TransformationInformation.from_images(self.__class__.__name__, params.transform, image_sitk,
                                                               resampled_image_sitk)
        image.get_transform_tape().record(transform_info)

        return image

    def execute(self,
                subject: Subject,
                params: ResamplingFilterParameters
                ) -> Subject:
        """Executes the resampling filter procedure.

        Args:
            subject (Subject): The subject to be processed.
            params (ResamplingFilterParameters): The parameters used for the resampling.

        Returns:
            Subject: The processed subject.
        """
        ref_image = subject.get_image_by_modality(params.reference_modality)
        new_ref_image = self.resample_intensity_image(ref_image, ref_image, params, tuple(subject.segmentation_images))
        subject.replace_image(new_ref_image, ref_image)

        ref_image = subject.get_image_by_modality(params.reference_modality)

        for image in subject.intensity_images:
            if image.get_modality() == params.reference_modality:
                continue

            new_image = self.resample_intensity_image(image, ref_image, params, tuple(subject.segmentation_images))
            subject.replace_image(new_image, image)

        for image in subject.segmentation_images:
            new_image = self.resample_segmentation_image(image, ref_image, params)
            subject.replace_image(new_image, image)

        return subject
