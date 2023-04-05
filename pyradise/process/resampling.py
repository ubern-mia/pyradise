from typing import Optional, Sequence, Tuple, Union

import itk
import numpy as np
import SimpleITK as sitk

from pyradise.data import (IntensityImage, Modality, SegmentationImage,
                           Subject, TransformInfo, str_to_modality)

from .base import Filter, FilterParams

__all__ = ["ResampleFilterParams", "ResampleFilter"]


# pylint: disable = too-few-public-methods
class ResampleFilterParams(FilterParams):
    """A filter parameter class for the :class:`~pyradise.process.resampling.ResampleFilter` class.

    The associated filter provides the following three ``centering_methods`` for resampling images:

    * The ``none`` centering_method resamples an image such that the output origin and direction does not change.
    * The ``reference`` centering_method resamples an image such that the output origin and direction is the same as
      the reference image (identified by the reference images modality).
    * The ``label_moment`` centering_method resamples an image such that the center of the resampled image is the
      average between the label center and the gravity center of the reference image. This method is a good approach
      for resampling data with bilateral and symmetric segmentations. However, it is an experimental method and should
      be used with caution.

    Args:
        output_size (Optional[Tuple[int, ...]]) : The output size of the images.
        output_spacing (Optional[Tuple[float, ...]]): The output spacing of the images.
        reference_modality (Optional[Union[Modality, str]]): The reference modality used if ``centering_method =
         'reference'`` or ``centering_method = 'label_moment'`` (default: None).
        transform (sitk.Transform): The transformation applied during resampling
         (default: sitk.AffineTransform(3) (identity transform)).
        centering_method (str): The method to center the image (options: 'none', 'reference', 'label_moment')
         (default: 'none').
        rescaling_intensity_images (bool): If True the intensity images will be automatically rescaled to the original
         intensity range (default: True).
    """

    def __init__(
        self,
        output_size: Optional[Tuple[int, ...]],
        output_spacing: Optional[Tuple[float, ...]],
        reference_modality: Optional[Union[Modality, str]] = None,
        transform: sitk.Transform = sitk.AffineTransform(3),
        centering_method: str = "none",
        rescaling_intensity_images: bool = True,
    ) -> None:
        super().__init__()

        if centering_method not in ("none", "reference", "label_moment"):
            raise ValueError(f"The centering method ({centering_method}) is invalid!")

        if centering_method in ("reference", "label_moment") and reference_modality is None:
            raise ValueError(f"A reference modality must be provided!")

        self.output_size = output_size
        self.output_spacing = output_spacing

        if reference_modality is not None:
            self.reference_modality: Optional[Modality] = str_to_modality(reference_modality)
        else:
            self.reference_modality: Optional[Modality] = reference_modality

        self.transform = transform
        self.centering_method = centering_method
        self.rescaling_intensity_images = rescaling_intensity_images


class ResampleFilter(Filter):
    """An invertible filter class for resampling all :class:`~pyradise.data.image.IntensityImage` and
    :class:`~pyradise.data.image.SegmentationImage` instances of a :class:`~pyradise.data.subject.Subject` instance.

    Warning:
        The inverse resampling procedure may not yield the expected results if successive
        :class:`~pyradise.process.base.Filter` s are applied to the same :class:`~pyradise.data.image.Image` instances.
        Thus, it's recommended to use the invertibility feature with appropriate caution.

    Note:
        Due to the limited precision of floating point numbers, the inverse normalization may not be exact.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Returns whether the filter is invertible or not.

        Returns:
            bool: True because the resampling of images is invertible.
        """
        return True

    @staticmethod
    def _get_label_center(images: Optional[Sequence[SegmentationImage]]) -> np.ndarray:
        """Get the label center of the provided segmentation images.

        This method computes the label center by calculating first the bounding box limits of all labels. Subsequently,
        for each label the center of both points is computed by averaging the coordinates. Finally, the center of all
        labels is computed by averaging the coordinates of all label centers.

        Args:
            images (Optional[Sequence[SegmentationImage]]): The segmentation images used for calculating the label
             center.

        Returns:
            np.ndarray: An array with the physical coordinates of label center.
        """
        # if no segmentation images are provided raise an error
        if not images:
            raise ValueError("No segmentation images are provided for the label center computation!")

        # compute the average label center
        bounding_box = []
        for image in images:
            image_sitk = image.get_image_data()
            num_dims = image_sitk.GetDimension()

            filter_ = sitk.LabelShapeStatisticsImageFilter()
            filter_.Execute(image_sitk)

            label_ids = filter_.GetLabels()
            for label_idx in label_ids:
                bounds = filter_.GetBoundingBox(label_idx)
                physical_bound_0 = image_sitk.TransformIndexToPhysicalPoint(bounds[0:num_dims])
                physical_bound_1 = image_sitk.TransformIndexToPhysicalPoint(bounds[num_dims : 2 * num_dims])
                bounding_box.append([physical_bound_0, physical_bound_1])

        bounding_box = np.array(bounding_box)
        label_centers = (bounding_box[:, 0, :] + bounding_box[:, 1, :]) / 2
        label_center = np.mean(label_centers, axis=0)

        return label_center

    @staticmethod
    def _get_label_moment_origin(
        image: IntensityImage, params: ResampleFilterParams, label_center: np.ndarray
    ) -> np.ndarray:
        """Get the origin of the label moment centering.

        This method computes the average between the image gravity center and the label center and calculates the
        corresponding image origin.

        Args:
            image (IntensityImage): The intensity image used for the moment calculation.
            params (ResampleFilterParams): The resampling filter parameters.
            label_center (np.ndarray): The label center.

        Returns:
            np.ndarray: The moment around the center.
        """
        # compute the image center of gravity
        image_itk = image.get_image_data(as_sitk=False)
        image_type = image.get_image_itk_type()
        moment_calc = itk.ImageMomentsCalculator[image_type].New()
        moment_calc.SetImage(image_itk)
        moment_calc.Compute()
        image_moment_center = np.array(moment_calc.GetCenterOfGravity())

        # compute the average between the image center of gravity and the label center
        center = (image_moment_center + label_center) / 2

        # compute the origin
        physical_image_center = np.array(params.output_spacing) * (np.array(params.output_size) // 2)
        origin = center - np.dot(image.get_direction(), physical_image_center)

        return origin

    # noinspection DuplicatedCode
    def _process_intensity_image(
        self,
        image: IntensityImage,
        reference_image: Optional[IntensityImage],
        params: ResampleFilterParams,
        segmentation_images: Optional[Sequence[SegmentationImage]] = None,
    ) -> IntensityImage:
        """Apply the resampling on an :class:`~pyradise.data.image.IntensityImage` instance.

        Args:
            image (IntensityImage): The intensity image to resample.
            reference_image (Optional[IntensityImage]): The reference image.
            params (ResampleFilterParams): The parameters for the resampling.
            segmentation_images (Optional[Sequence[SegmentationImage]]): Segmentation images used for the moment
             calculation (default: None).

        Returns:
            IntensityImage: The resampled intensity image.
        """
        # get the image data of the moving and fixed image as SimpleITK images
        image_sitk = image.get_image_data()

        # cast the image to a float image
        if image_sitk.GetPixelID() != sitk.sitkFloat32:
            image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)

        # get the minimum and maximum intensity values
        image_np = sitk.GetArrayFromImage(image_sitk)
        min_intensity = float(np.min(image_np))
        max_intensity = float(np.max(image_np))

        # get the output origin and direction
        if params.centering_method == "none":
            output_origin = image_sitk.GetOrigin()
            output_direction = image_sitk.GetDirection()
            output_size = image_sitk.GetSize()
            output_spacing = image_sitk.GetSpacing()

        elif params.centering_method == "reference":
            if reference_image is None:
                raise ValueError(
                    'The reference image must be provided for the centering method "reference" and ' '"label_moment"!'
                )
            reference_image_sitk = reference_image.get_image_data()
            output_origin = reference_image_sitk.GetOrigin()
            output_direction = reference_image_sitk.GetDirection()
            output_size = reference_image_sitk.GetSize()
            output_spacing = reference_image_sitk.GetSpacing()

        elif params.centering_method == "label_moment":
            if reference_image is None:
                raise ValueError(
                    'The reference image must be provided for the centering method "reference" and ' '"label_moment"!'
                )
            reference_image_sitk = reference_image.get_image_data()
            output_size = reference_image_sitk.GetSize()
            output_spacing = reference_image_sitk.GetSpacing()

            if reference_image == image:
                label_center = self._get_label_center(segmentation_images)
                label_moment_origin = self._get_label_moment_origin(image, params, label_center)

                output_origin = label_moment_origin
                output_direction = reference_image_sitk.GetDirection()
            else:
                output_origin = reference_image_sitk.GetOrigin()
                output_direction = reference_image_sitk.GetDirection()

        else:
            raise NotImplementedError(f"The centering method ({params.centering_method}) is not supported!")

        # apply the resampling filter
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetInterpolator(sitk.sitkBSpline)
        resample_filter.SetTransform(params.transform)
        resample_filter.SetDefaultPixelValue(min_intensity)
        resample_filter.SetOutputOrigin(output_origin)
        resample_filter.SetOutputDirection(output_direction)

        if params.output_spacing:
            resample_filter.SetOutputSpacing(params.output_spacing)
        else:
            resample_filter.SetOutputSpacing(output_spacing)

        if params.output_size:
            resample_filter.SetSize(params.output_size)
        else:
            resample_filter.SetSize(output_size)

        new_image_sitk = resample_filter.Execute(image_sitk)

        # rescale the image to the original intensity range
        if params.rescaling_intensity_images:
            rescale_filter = sitk.RescaleIntensityImageFilter()
            rescale_filter.SetOutputMinimum(min_intensity)
            rescale_filter.SetOutputMaximum(max_intensity)
            new_image_sitk = rescale_filter.Execute(new_image_sitk)

        # add the new image data to the intensity image
        image.set_image_data(new_image_sitk)

        # track the necessary parameters
        self.tracking_data["min_intensity"] = min_intensity
        self.tracking_data["max_intensity"] = max_intensity
        self.tracking_data["is_intensity"] = True
        self._register_tracked_data(image, image_sitk, new_image_sitk, params, params.transform)

        return image

    @staticmethod
    def _inverse_process_intensity_image(image: IntensityImage, transform_info: TransformInfo) -> IntensityImage:
        """Apply the inverse resampling on an :class:`~pyradise.data.image.IntensityImage` instance.

        Args:
            image (IntensityImage): The intensity image to inversely resample.
            transform_info (TransformInfo): The transform information.

        Returns:
            IntensityImage: The inversely resampled intensity image.
        """
        # get the image data and the pre-transform image properties
        image_sik = image.get_image_data()
        pre_transform_props = transform_info.get_image_properties(True)

        # apply the resampling filter
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetInterpolator(sitk.sitkBSpline)
        resample_filter.SetTransform(transform_info.get_transform(True))
        resample_filter.SetDefaultPixelValue(float(transform_info.get_data("min_intensity")))
        resample_filter.SetOutputOrigin(pre_transform_props.origin)
        resample_filter.SetOutputDirection(pre_transform_props.direction)
        resample_filter.SetOutputSpacing(pre_transform_props.spacing)
        resample_filter.SetSize(pre_transform_props.size)

        new_image_sitk = resample_filter.Execute(image_sik)

        # rescale the image to the original intensity range (if necessary)
        params = transform_info.get_params()
        if params.rescaling_intensity_images:
            rescale_filter = sitk.RescaleIntensityImageFilter()
            rescale_filter.SetOutputMinimum(transform_info.get_data("min_intensity"))
            rescale_filter.SetOutputMaximum(transform_info.get_data("max_intensity"))
            new_image_sitk = rescale_filter.Execute(new_image_sitk)

        # set the new image data to the intensity image
        image.set_image_data(new_image_sitk)

        return image

    # noinspection DuplicatedCode
    def _process_segmentation_image(
        self,
        image: SegmentationImage,
        reference_image: Optional[IntensityImage],
        params: ResampleFilterParams,
    ) -> SegmentationImage:
        """Apply the resampling on an :class:`~pyradise.data.image.SegmentationImage` instance.

        Args:
            image (SegmentationImage): The image to resample.
            reference_image (Optional[IntensityImage]): The reference image.
            params (ResampleFilterParams): The parameters for the resampling.

        Returns:
            SegmentationImage: The resampled segmentation image.
        """
        # get the image data of the moving and fixed image as SimpleITK images
        image_sitk = image.get_image_data()

        # get the output origin and direction
        if params.centering_method == "none":
            output_origin = image_sitk.GetOrigin()
            output_direction = image_sitk.GetDirection()
            output_size = image_sitk.GetSize()
            output_spacing = image_sitk.GetSpacing()

        elif params.centering_method in ("reference", "label_moment"):
            if reference_image is None:
                raise ValueError(
                    'The reference image must be provided for the centering method "reference" and ' '"label_moment"!'
                )
            reference_image_sitk = reference_image.get_image_data()
            output_origin = reference_image_sitk.GetOrigin()
            output_direction = reference_image_sitk.GetDirection()
            output_size = reference_image_sitk.GetSize()
            output_spacing = reference_image_sitk.GetSpacing()

        else:
            raise NotImplementedError(f"The centering method ({params.centering_method}) is invalid!")

        # apply the resampling filter
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
        resample_filter.SetTransform(params.transform)
        resample_filter.SetDefaultPixelValue(0)
        resample_filter.SetOutputOrigin(output_origin)
        resample_filter.SetOutputDirection(output_direction)

        if params.output_spacing:
            resample_filter.SetOutputSpacing(params.output_spacing)
        else:
            resample_filter.SetOutputSpacing(output_spacing)

        if params.output_size:
            resample_filter.SetSize(params.output_size)
        else:
            resample_filter.SetSize(output_size)

        new_image_sitk = resample_filter.Execute(image_sitk)

        # add the new image data to the segmentation image
        image.set_image_data(new_image_sitk)

        # track the necessary parameters
        self.tracking_data["min_intensity"] = 0.0
        self.tracking_data["max_intensity"] = 1.0
        self.tracking_data["is_intensity"] = False
        self._register_tracked_data(image, image_sitk, new_image_sitk, params, params.transform)

        return image

    @staticmethod
    def _inverse_process_segmentation_image(
        image: SegmentationImage, transform_info: TransformInfo
    ) -> SegmentationImage:
        """Apply the inverse resampling on an :class:`~pyradise.data.image.SegmentationImage` instance.

        Args:
            image (SegmentationImage): The image to resample.
            transform_info (TransformInfo): The transform information.

        Returns:
            SegmentationImage: The inversely resampled segmentation image.
        """
        # get the image data and the pre-transform image properties
        image_sik = image.get_image_data()
        pre_transform_props = transform_info.get_image_properties(True)

        # apply the resampling filter
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
        resample_filter.SetTransform(transform_info.get_transform(True))
        resample_filter.SetDefaultPixelValue(0)
        resample_filter.SetOutputOrigin(pre_transform_props.origin)
        resample_filter.SetOutputDirection(pre_transform_props.direction)
        resample_filter.SetOutputSpacing(pre_transform_props.spacing)
        resample_filter.SetSize(pre_transform_props.size)

        new_image_sitk = resample_filter.Execute(image_sik)

        # set the new image data to the intensity image
        image.set_image_data(new_image_sitk)

        return image

    def execute(self, subject: Subject, params: ResampleFilterParams) -> Subject:
        """Executes the resampling filter procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            params (ResampleFilterParams): The parameters used for the resampling.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with resampled
            :class:`~pyradise.data.image.IntensityImage` and :class:`~pyradise.data.image.SegmentationImage` instances.
        """
        # get the segmentation images
        segmentation_images = subject.get_images_by_type(SegmentationImage)
        segmentation_images = [img for img in segmentation_images if isinstance(img, SegmentationImage)]

        # get the reference images if necessary
        if params.centering_method != "none":
            # resample the reference intensity image
            ref_image = subject.get_image_by_modality(params.reference_modality)
            self._process_intensity_image(ref_image, ref_image, params, segmentation_images)
        else:
            ref_image = None

        # resample the intensity images
        for image in subject.intensity_images:
            # exclude the reference image if not centering_method == 'none'
            if params.centering_method != "none":
                if image.get_modality() == params.reference_modality:
                    continue

            self._process_intensity_image(image, ref_image, params, segmentation_images)

        # resample the segmentation images
        for image in subject.segmentation_images:
            self._process_segmentation_image(image, ref_image, params)

        return subject

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Executes the inverse resampling filter procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            transform_info (TransformInfo): The transform information.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with inversely resampled
            :class:`~pyradise.data.image.IntensityImage` and :class:`~pyradise.data.image.SegmentationImage` instances.
        """
        for image in subject.intensity_images:
            if target_image is None or image == target_image:
                self._inverse_process_intensity_image(image, transform_info)

        for image in subject.segmentation_images:
            if target_image is None or image == target_image:
                self._inverse_process_segmentation_image(image, transform_info)

        return subject
