from abc import abstractmethod
from typing import Any, Optional, Tuple, Union
from warnings import warn

import numpy as np
import SimpleITK as sitk

from pyradise.data import (IntensityImage, Modality, SegmentationImage,
                           Subject, TransformInfo, seq_to_modalities)

from .base import Filter, FilterParams, LoopEntryFilter, LoopEntryFilterParams

__all__ = [
    "IntensityFilterParams",
    "IntensityFilter",
    "IntensityLoopFilterParams",
    "IntensityLoopFilter",
    "ZScoreNormFilterParams",
    "ZScoreNormFilter",
    "ZeroOneNormFilterParams",
    "ZeroOneNormFilter",
    "RescaleIntensityFilterParams",
    "RescaleIntensityFilter",
    "ClipIntensityFilterParams",
    "ClipIntensityFilter",
    "GaussianFilterParams",
    "GaussianFilter",
    "MedianFilterParams",
    "MedianFilter",
    "LaplacianFilterParams",
    "LaplacianFilter",
]


class IntensityFilterParams(FilterParams):
    """A filter parameter class for the :class:`~pyradise.process.intensity.IntensityFilter` base class. In addition to
    the :class:`~pyradise.process.base.FilterParams` class, this class also provides a ``modalities`` parameter to
    specify the images to be processed by the filters.

    Args:
        modalities (Optional[Tuple[Union[Modality, str], ...]]): The modalities associated with the corresponding
         :class:`~pyradise.data.image.IntensityImage` instances that should be processed. If ``None`` is provided,
         all :class:`~pyradise.data.image.IntensityImage` instances will be processed (default: None).
    """

    def __init__(self, modalities: Optional[Tuple[Union[Modality, str], ...]] = None) -> None:
        if modalities is not None:
            self.modalities = seq_to_modalities(modalities)
        else:
            self.modalities: Optional[Tuple[Modality, ...]] = None


class IntensityFilter(Filter):
    """An abstract filter base class for intensity modifying filters which process the whole image content. In contrast
    to the :class:`~pyradise.process.base.Filter` base class, this class provides two additional abstract methods (
    i.e. :meth:`~pyradise.process.intensity.IntensityFilter._process_image` and
    :meth:`~pyradise.process.intensity.IntensityFilter._process_image_inverse`) for processing the image content as
    whole. Thus, this base class is intended to be used for intensity modifying filters which process the whole image
    content at once.

    Note:
        The selection of the :class:`~pyradise.data.image.IntensityImage` instances to be processed is specified by the
        :class:`~pyradise.process.intensity.IntensityFilterParams` instance. If the ``modalities`` parameter is set to
        ``None``, all :class:`~pyradise.data.image.IntensityImage` instances will be processed. Otherwise, only the
        :class:`~pyradise.data.image.IntensityImage` instances with the specified modalities will be processed. If the
        user wants to implement its own intensity modifying filter, the user do not need to implement the
        selection of the images to be processed. The selection mechanism is already provided in the implemented
        :meth:`~pyradise.process.intensity.IntensityFilter.execute` and
        :meth:`~pyradise.process.intensity.IntensityFilter.execute_inverse` methods.

    Example:

        An example implementation of an intensity clippling filter:

        >>> class ClipFilterParams(IntensityFilterParams):
        >>>     def __init__(self,
        >>>                  min_out: float,
        >>>                  max_out: float,
        >>>                  modalities: Optional[Tuple[Union[Modality, str], ...]] = None
        >>>                  ) -> None:
        >>>         super().__init__(modalities)
        >>>
        >>>         if min_out == max_out:
        >>>             raise ValueError('The min and max output intensity '
        >>>                              'values must not be equal because '
        >>>                              'the resulting image '
        >>>                              'will have constant intensity.')
        >>>
        >>>         if min_out > max_out:
        >>>             min_out, max_out = max_out, min_out
        >>>
        >>>         self.min_value: float = min_out
        >>>         self.max_value: float = max_out
        >>>
        >>>
        >>> class ClipFilter(IntensityFilter):
        >>>
        >>>     @staticmethod
        >>>     def is_invertible() -> bool:
        >>>         # return False because the clipping is not invertible
        >>>         return False
        >>>
        >>>     def _process_image(self,
        >>>                        image: IntensityImage,
        >>>                        params: ClipFilterParams
        >>>                        ) -> IntensityImage:
        >>>         # get the image data
        >>>         sitk_image = image.get_image_data()
        >>>
        >>>         # apply the clipping
        >>>         clipped_image_sitk = sitk.Clamp(sitk_image,
        >>>                                         sitk_image.GetPixelIDValue(),
        >>>                                         params.min_value,
        >>>                                         params.max_value)
        >>>
        >>>         # add the clipped SimpleITK image to the image
        >>>         image.set_image_data(clipped_image_sitk)
        >>>
        >>>         # track the necessary information
        >>>         self._register_tracked_data(image, sitk_image,
        >>>                                     clipped_image_sitk, params)
        >>>
        >>>         return image
        >>>
        >>>     def _process_image_inverse(self,
        >>>                                image: IntensityImage,
        >>>                                transform_info: TransformInfo
        >>>                                ) -> IntensityImage:
        >>>         # return the original image because the clipping
        >>>         # is not invertible
        >>>         return image
        >>>
        >>>     def execute(self,
        >>>                 subject: Subject,
        >>>                 params: ClipFilterParams
        >>>                 ) -> Subject:
        >>>         # implement exclusively due to type adaptation for params
        >>>         return super().execute(subject, params)

    """

    @abstractmethod
    def _process_image(self, image: IntensityImage, params: IntensityFilterParams) -> IntensityImage:
        """Process the content of an image.

        Args:
            image (IntensityImage): The :class:`~pyradise.data.image.IntensityImage` instance to be processed.
            params (IntensityFilterParams): The filter parameters.

        Returns:
            IntensityImage: The processed :class:`~pyradise.data.image.IntensityImage` instance.
        """
        raise NotImplementedError()

    @abstractmethod
    def _process_image_inverse(self, image: IntensityImage, transform_info: TransformInfo) -> IntensityImage:
        """Process the content of an image inversely.

        Args:
            image (IntensityImage): The :class:`~pyradise.data.image.IntensityImage` instance to be processed.
            transform_info (TransformInfo): The transform information.

        Returns:
            IntensityImage: The inversely processed :class:`~pyradise.data.image.IntensityImage` instance.
        """
        raise NotImplementedError()

    def execute(self, subject: Subject, params: IntensityFilterParams) -> Subject:
        """Execute the intensity modifying procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            params (IntensityFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with processed
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        for image in subject.get_images():
            if isinstance(image, IntensityImage):
                # check if the image is specified for processing
                if params.modalities is not None and image.modality not in params.modalities:
                    image_sitk = image.get_image_data()
                    self._register_tracked_data(image, image_sitk, image_sitk, params)

                else:
                    self._process_image(image, params)

        return subject

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Execute the inverse intensity modifying procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            transform_info (TransformInfo): The transform information.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with processed
            :class:`~pyradise.data.image.IntensityImage` instances.
        """

        # potentially warn the user that the operation is not invertible
        if self.warn_on_non_invertible and not self.is_invertible():
            warn(
                "WARNING: "
                f"The {self.__class__.__name__} is called to invert its operation for the following image: \n"
                f"\t{target_image.__str__()} \nHowever, the filter is not invertible. The provided subject "
                "is returned without modification."
            )

        for image in subject.get_images():
            if target_image is not None and image != target_image:
                continue

            if isinstance(image, IntensityImage):
                if (
                    transform_info.params.modalities is not None
                    and image.get_modality() not in transform_info.params.modalities
                ):
                    continue

                self._process_image_inverse(image, transform_info)

        return subject


class IntensityLoopFilterParams(LoopEntryFilterParams):
    """A filter parameter class for the :class:`~pyradise.process.intensity.IntensityLoopFilter` base class.
    In addition to the :class:`~pyradise.process.base.LoopEntryFilterParams` class, this class also provides a
    ``modalities`` parameter to specify the images to be processed by the filters.

    Args:
        loop_axis (Optional[int]): The axis along which the data transformation is performed. If ``None``, the
         transformation is performed on the whole image at once. If a value is given, the transformation is performed
         by looping over the corresponding image dimension.
        modalities (Optional[Tuple[Union[Modality, str], ...]]): The modalities associated with the corresponding
         :class:`~pyradise.data.image.IntensityImage` instances that should be processed. If ``None`` is provided,
         all :class:`~pyradise.data.image.IntensityImage` instances will be processed (default: None).
    """

    def __init__(self, loop_axis: Optional[int], modalities: Optional[Tuple[Union[Modality, str], ...]] = None) -> None:
        super().__init__(loop_axis)

        if modalities is not None:
            self.modalities = seq_to_modalities(modalities)
        else:
            self.modalities = None


class IntensityLoopFilter(LoopEntryFilter):
    """An abstract base class for intensity modifying filters that can process the provided image content by looping
    over a specific axis or by using the whole image extent at once. This base class is intended to be used for
    implementing flexible intensity modifying filters that can iteratively process subsets of the image content such as
    for example normalization filters which may be applied slice-wise to the image content.

    For the implementation of a new :class:`~pyradise.process.intensity.IntensityLoopFilter` subclass implement the
    provided :meth:`~pyradise.process.intensity.IntensityLoopFilter._modify_array` and
    :meth:`~pyradise.process.intensity.IntensityLoopFilter._modify_array_inverse` methods. Both methods are called
    iteratively for each loop axis position. The :meth:`~pyradise.process.intensity.IntensityLoopFilter._modify_array`
    method is called for the forward processing and the
    :meth:`~pyradise.process.intensity.IntensityLoopFilter._modify_array_inverse` method is called for the inverse
    processing.

    Note:
        The selection of the :class:`~pyradise.data.image.IntensityImage` instances to be processed is specified by the
        :class:`~pyradise.process.intensity.IntensityLoopFilterParams` instance. If the ``modalities`` parameter is set
        to ``None``, all :class:`~pyradise.data.image.IntensityImage` instances will be processed. Otherwise, only the
        :class:`~pyradise.data.image.IntensityImage` instances with the specified modalities will be processed. If the
        user wants to implement its own intensity modifying filter, the user do not need to implement the
        selection of the images to be processed. The selection mechanism is already provided in the implemented
        :meth:`~pyradise.process.intensity.IntensityLoopFilter.execute` and
        :meth:`~pyradise.process.intensity.IntensityLoopFilter.execute_inverse` methods.
    """

    def __init__(self):
        super().__init__()

        # provides an index for the position along the loop axis
        self.loop_axis_pos_idx = 0

    @abstractmethod
    def _modify_array(self, array: np.ndarray, params: IntensityLoopFilterParams) -> np.ndarray:
        """The intensity modification function which is applied to the provided array. The provided array can be of
        n-dimensions whereby the dimensionality depend on the provided data and the ``loop_axis`` parameter as
        specified in the appropriate :class:`~pyradise.process.intensity.IntensityFilterParams` instance.

        Args:
            array (np.ndarray): The array to be processed.
            params (IntensityLoopFilterParams): The parameters used for the processing.

        Returns:
            np.ndarray: The processed array.
        """
        raise NotImplementedError()

    @abstractmethod
    def _modify_array_inverse(self, array: np.ndarray, params: TransformInfo) -> np.ndarray:
        """The inverse intensity modification function which is applied to the provided array. The provided array can
        be of n-dimensions whereby the dimensionality depend on the provided data and the ``loop_axis`` parameter as
        specified in the appropriate :class:`~pyradise.process.intensity.IntensityFilterParams` instance which is
        contained in the provided :class:`~pyradise.data.taping.TransformInfo` instance.

        Args:
            array (np.ndarray): The array to be processed.
            params (TransformInfo): The transform information.

        Returns:
            np.ndarray: The processed array.
        """
        raise NotImplementedError()

    def _process_image(self, image: IntensityImage, params: Union[IntensityLoopFilterParams, Any]) -> IntensityImage:
        """Execute the intensity modifying procedure on the provided image by looping over the image accordingly.

        Args:
            image (IntensityImage): The image to be processed.
            params (Union[IntensityLoopFilterParams, Any]): The filter parameters.

        Returns:
            IntensityImage: The processed image.
        """

        # set the loop axis position index to zero because of processing a new image
        self.loop_axis_pos_idx = 0

        # get the image data for computation
        image_sitk = image.get_image_data()
        if "integer" in image_sitk.GetPixelIDTypeAsString():
            image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)

        image_np = sitk.GetArrayFromImage(image_sitk)

        # perform the intensity modifying procedure
        new_image_np = self.loop_entries(image_np, params, self._modify_array, params.loop_axis)

        # construct the new image
        new_image_sitk = sitk.GetImageFromArray(new_image_np)
        new_image_sitk.CopyInformation(image_sitk)

        # set the new image data to the image
        image.set_image_data(new_image_sitk)

        # keep track of the transformation
        self._register_tracked_data(image, image_sitk, new_image_sitk, params)

        return image

    def _process_image_inverse(self, image: IntensityImage, transform_info: TransformInfo) -> IntensityImage:
        """Execute the inverse intensity modifying procedure on the provided image by looping over the image
        accordingly.

        Args:
            image (IntensityImage): The image to be processed.
            transform_info (TransformInfo): The transform information.

        Returns:
            IntensityImage: The processed image.
        """
        # return the image as is if the filter is not invertible
        if not self.is_invertible():
            return image

        # set the loop axis position index to zero because of processing a new image
        self.loop_axis_pos_idx = 0

        # get the image data for inverse processing
        image_sitk = image.get_image_data()
        if "integer" in image_sitk.GetPixelIDTypeAsString():
            image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)
        image_np = sitk.GetArrayFromImage(image_sitk)

        # perform the inverse intensity modifying procedure
        new_image_np = self.loop_entries(
            image_np, transform_info, self._modify_array_inverse, transform_info.params.loop_axis
        )

        # construct the new image
        new_image_sitk = sitk.GetImageFromArray(new_image_np)
        new_image_sitk.CopyInformation(image_sitk)

        # set the new image data to the image
        image.set_image_data(new_image_sitk)

        return image

    def execute(self, subject: Subject, params: IntensityLoopFilterParams) -> Subject:
        """Execute the intensity modifying procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            params (IntensityLoopFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with processed
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        for image in subject.get_images():
            if isinstance(image, IntensityImage):
                # check if the image is specified for processing
                if params.modalities is not None and image.get_modality() not in params.modalities:
                    image_sitk = image.get_image_data()
                    self._register_tracked_data(image, image_sitk, image_sitk, params)

                else:
                    self._process_image(image, params)

        return subject

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Execute the inverse intensity modifying procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            transform_info (TransformInfo): The transform information.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with processed
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        for image in subject.get_images():
            if target_image is not None and image != target_image:
                continue

            if isinstance(image, IntensityImage):
                if (
                    transform_info.params.modalities is not None
                    and image.get_modality() not in transform_info.params.modalities
                ):
                    continue

                self._process_image_inverse(image, transform_info)

        return subject


# pylint: disable=too-few-public-methods
class ZScoreNormFilterParams(IntensityLoopFilterParams):
    """A filter parameter class for the :class:`~pyradise.process.intensity.ZScoreNormFilter` class.

    Args:
        loop_axis (Optional[int]): The axis along which the intensity normalization is performed. If None, the
         intensity normalization is performed on the whole image extent at once. If a value is given, the intensity
         normalization is performed by looping over the corresponding image dimension (default: None).
        modalities (Optional[Tuple[Union[Modality, str], ...]]): The modalities of the images to be rescaled. If
         ``None`` is provided all images of the provided subject are rescaled (default: None).
    """

    def __init__(
        self, loop_axis: Optional[int] = None, modalities: Optional[Tuple[Union[Modality, str], ...]] = None
    ) -> None:
        super().__init__(loop_axis, modalities)


class ZScoreNormFilter(IntensityLoopFilter):
    """A normalization filter class performing an invertible z-score normalization on all
    :class:`~pyradise.data.image.IntensityImage` instances of the provided :class:`~pyradise.data.subject.Subject`
    instance.

    For the normalization the following formula is applied to the image extent or its subsets:

    .. math::
        I_{norm} = \\frac{I_{orig} - \\mu(I_{orig})}{\\sigma(I_{orig})}

    For the inverse normalization the following formula is applied to the image extent or its subsets:

    .. math::
        I_{orig} = I_{norm} \\cdot \\sigma(I_{orig}) + \\mu(I_{orig})

    During the normalization procedure, the intensity mean and standard deviation of the original image or its subsets
    are tracked such that these values are available for the inverse normalization.

    Warning:
        The inverse normalization procedure may not yield the expected results if successive
        :class:`~pyradise.process.base.Filter` s are applied to the same :class:`~pyradise.data.image.Image` instances.
        Thus, it's recommended to use the invertibility feature with appropriate caution.

    Note:
        Due to the limited precision of floating point numbers, the inverse normalization may not be exact.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Check if the filter is invertible.

        Returns:
            bool: True because the z-score normalization procedure is invertible.
        """
        return True

    def _modify_array(self, array: np.ndarray, params: Any) -> np.ndarray:
        """Apply the z-score normalization function to the provided data array.

        Args:
            array (np.ndarray): The array to be normalized.
            params (Any): The parameters used for the normalization.

        Returns:
            np.ndarray: The z-score normalized array.
        """
        # get the mean and standard deviation of the array
        mean = np.mean(array)
        std = np.std(array)

        # track the changes
        self.tracking_data[f"mean_{self.loop_axis_pos_idx}"] = mean
        self.tracking_data[f"std_{self.loop_axis_pos_idx}"] = std

        self.loop_axis_pos_idx += 1

        # compute the normalization function
        return (array - np.mean(array)) / np.std(array)

    def _modify_array_inverse(self, array: np.ndarray, params: TransformInfo) -> np.ndarray:
        """Apply the de-normalization function to the provided data array.

        Args:
            array (np.ndarray): The array to be denormalized.
            params (TransformInfo): The parameters used for the de-normalization.

        Returns:
            np.ndarray: The denormalized array.

        """
        # get the tracked data
        original_mean = params.get_data(f"mean_{self.loop_axis_pos_idx}")
        original_std = params.get_data(f"std_{self.loop_axis_pos_idx}")

        self.loop_axis_pos_idx += 1

        # compute the inverse normalization function
        return array * original_std + original_mean

    def execute(self, subject: Subject, params: ZScoreNormFilterParams) -> Subject:
        """Execute the z-score normalization procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            params (ZScoreNormFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with z-score normalized
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        return super().execute(subject, params)

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Execute the inverse z-score normalization procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            transform_info (TransformInfo): The transform information.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with denormalized
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        return super().execute_inverse(subject, transform_info)


# pylint: disable=too-few-public-methods
class ZeroOneNormFilterParams(IntensityLoopFilterParams):
    """A filter parameter class for the :class:`~pyradise.process.intensity.ZeroOneNormFilter` class.

    Args:
        loop_axis (Optional[int]): The axis along which the intensity normalization is performed. If None, the
         intensity normalization is performed on the whole image extent at once. If a value is given, the intensity
         normalization is performed by looping over the corresponding image dimension (default: None).
        modalities (Optional[Tuple[Union[Modality, str], ...]]): The modalities of the images to be rescaled. If
         ``None`` is provided all images of the provided subject are rescaled (default: None).
    """

    def __init__(
        self, loop_axis: Optional[int] = None, modalities: Optional[Tuple[Union[Modality, str], ...]] = None
    ) -> None:
        super().__init__(loop_axis, modalities)


class ZeroOneNormFilter(IntensityLoopFilter):
    """A normalization filter class performing an invertible zero-one (1-0) normalization on all
    :class:`~pyradise.data.image.IntensityImage` instances of the provided :class:`~pyradise.data.subject.Subject`
    instance.

    For the normalization the following formula is applied to the image extent or its subsets:

    .. math::
        I_{norm} = \\frac{I_{orig} - \\min(I_{orig})}{\\max(I_{orig}) - \\min(I_{orig})}

    For the inverse normalization the following formula is applied to the image extent or its subsets:

    .. math::
        I_{orig} = I_{norm} \\cdot (\\max(I_{orig}) - \\min(I_{orig})) + \\min(I_{orig})

    During the normalization procedure, the min and max intensity values of the image or its subsets are tracked to be
    available for inverse normalization.

    Warning:
        The inverse normalization procedure may not yield the expected results if successive
        :class:`~pyradise.process.base.Filter` s are applied to the same :class:`~pyradise.data.image.Image` instances.
        Thus, it's recommended to use the invertibility feature with appropriate caution.

    Note:
        Due to the limited precision of floating point numbers, the inverse normalization may not be exact.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Check if the filter is invertible.

        Returns:
            bool: True because the zero-one normalization procedure is invertible.
        """
        return True

    def _modify_array(self, array: np.ndarray, params: Any) -> np.ndarray:
        """Apply the zero-one normalization function to the provided data array.

        Args:
            array (np.ndarray): The array to be normalized.
            params (Any): The parameters used for the normalization.

        Returns:
            np.ndarray: The zero-one normalized array.
        """
        # get the min and max of the array
        min_val = np.min(array)
        max_val = np.max(array)

        # track the changes
        self.tracking_data[f"min_{self.loop_axis_pos_idx}"] = min_val
        self.tracking_data[f"max_{self.loop_axis_pos_idx}"] = max_val

        self.loop_axis_pos_idx += 1

        # compute the normalization function
        return (array - min_val) / (max_val - min_val)

    def _modify_array_inverse(self, array: np.ndarray, params: TransformInfo) -> np.ndarray:
        """Apply the de-normalization function to the provided data array.

        Args:
            array (np.ndarray): The array to be denormalized.
            params (Any): The parameters used for the de-normalization.

        Returns:
            np.ndarray: The min-max normalized array.
        """
        # get the tracked data
        original_min = params.get_data(f"min_{self.loop_axis_pos_idx}")
        original_max = params.get_data(f"max_{self.loop_axis_pos_idx}")

        self.loop_axis_pos_idx += 1

        # compute the inverse normalization function
        return array * (original_max - original_min) + original_min

    def execute(self, subject: Subject, params: ZeroOneNormFilterParams) -> Subject:
        """Execute the zero-one normalization procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            params (ZeroOneNormFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with zero-one normalized
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        return super().execute(subject, params)

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Execute the inverse zero-one normalization procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            transform_info (TransformInfo): The transform information.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with denormalized
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        return super().execute_inverse(subject, transform_info)


# pylint: disable=too-few-public-methods
class RescaleIntensityFilterParams(IntensityFilterParams):
    """A filter parameter class for the :class:`~pyradise.process.intensity.RescaleIntensityFilter` class.

    Args:
        min_out (Optional[float]): The minimum value of the rescaled image. If ``None`` is provided the filter takes
         the minimum intensity value of the image.
        max_out (Optional[float]): The maximum value of the rescaled image. If ``None`` is provided the filter takes
         the maximum intensity value of the image.
        modalities (Optional[Tuple[Union[Modality, str], ...]]): The modalities of the images to be rescaled. If
         ``None`` is provided all images of the provided subject are rescaled (default: None).
    """

    def __init__(
        self,
        min_out: Optional[float],
        max_out: Optional[float],
        modalities: Optional[Tuple[Union[Modality, str], ...]] = None,
    ) -> None:
        super().__init__(modalities)

        # check the provided min and max values
        if min_out == max_out:
            raise ValueError(
                "The specified min and max output values are equal. The resulting image would have "
                "constant intensity."
            )

        if min_out > max_out:
            min_out, max_out = max_out, min_out

        self.min_out: Optional[float] = min_out
        self.max_out: Optional[float] = max_out


class RescaleIntensityFilter(IntensityFilter):
    """A filter class performing an invertible intensity rescaling on all selected
    :class:`~pyradise.data.image.IntensityImage` instances of the provided :class:`~pyradise.data.subject.Subject`
    instance.

    For the rescaling the following formula is applied to the image extent or its subsets:

    .. math::
        I_{resc} = \\frac{I_{orig} - \\min(I_{orig})}{\\max(I_{orig}) - \\min(I_{orig})} \\cdot (max_{out} - min_{out})
        + min_{out}

    For the inverse rescaling the following formula is applied to the image extent or its subsets:

    .. math::
        I_{orig} = \\frac{I_{resc} - \\min(I_{resc})}{\\max(I_{resc}) - \\min(I_{resc})} \\cdot (\\max(I_{orig}) -
        \\min(I_{orig})) + \\min(I_{orig})

    During the rescaling procedure, the min and max intensity of the original image or its subsets are tracked to be
    available for inverse rescaling.

    Warning:
        The inverse rescaling procedure may not yield the expected results if successive
        :class:`~pyradise.process.base.Filter` s are applied to the same :class:`~pyradise.data.image.Image` instances.
        Thus, it's recommended to use the invertibility feature with appropriate caution.

    Note:
        Due to the limited precision of floating point numbers, the inverse normalization may not be exact.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Check if the filter is invertible.

        Returns:
            bool: True because the rescaling procedure is invertible.
        """
        return True

    def _process_image(self, image: IntensityImage, params: RescaleIntensityFilterParams) -> IntensityImage:
        """Apply the rescaling function to the provided image.

        Args:
            image (IntensityImage): The image to be rescaled.
            params (RescaleIntensityFilterParams): The filter parameters.

        Returns:
            IntensityImage: The processed image with rescaled intensity values.
        """
        # get the image data as numpy array
        image_sitk = image.get_image_data()
        image_np = image.get_image_data_as_np(False).astype(float)

        # get the min and max values
        min_i_o = np.min(image_np)
        max_i_o = np.max(image_np)
        range_i_o = max_i_o - min_i_o

        # track the min and max intensity
        self.tracking_data["min"] = min_i_o
        self.tracking_data["max"] = max_i_o

        # get the range of the output values
        param_range = params.max_out - params.min_out

        # check if the range of the input array is larger than zero
        if range_i_o < 1e-10:
            warn(
                "The range of the input image or its subset is smaller than 1e-10. The rescaled image or subset"
                "will contain the specified minimum intensity value including the provided noise (input - min(input) "
                "+ min_out)."
            )
            new_image_np = image_np - min_i_o + params.min_out

        # rescale the intensity values
        else:
            new_image_np = (image_np - min_i_o) / range_i_o * param_range + params.min_out

        # add the new SimpleITK image to the PyRaDiSe image
        new_image_sitk = sitk.GetImageFromArray(new_image_np)
        new_image_sitk.CopyInformation(image_sitk)
        image.set_image_data(new_image_sitk)

        # track the necessary data for invertibility
        self._register_tracked_data(image, image_sitk, new_image_sitk, params)

        return image

    def _process_image_inverse(self, image: IntensityImage, transform_info: TransformInfo) -> IntensityImage:
        """Apply the inverse scaling function to the provided data array.

        Args:
            image (IntensityImage): The image to be inversely rescaled.
            transform_info (TransformInfo): The transform information.

        Returns:
            IntensityImage: The inversely processed :class:`~pyradise.data.image.IntensityImage` instance.
        """
        # get the data as numpy array
        image_sitk = image.get_image_data()
        image_np = image.get_image_data_as_np(False).astype(float)

        # get the tracked data
        min_i_o = transform_info.get_data(f"min")
        max_i_o = transform_info.get_data(f"max")
        range_i_o = max_i_o - min_i_o

        # compute the min and max values of the provided array
        min_i_r = np.min(image_np)
        max_i_r = np.max(image_np)
        range_i_r = max_i_r - min_i_r

        # check if the range of the input array is larger than zero
        if range_i_r < 1e-10:
            warn(
                "The range of the input image or its subset is smaller than 1e-10. The rescaled image or subset"
                "will contain the originally provided values (rescaled_input - min(rescaled_input) + min_original)."
            )
            new_image_np = image_np - min_i_r + min_i_o

        # inversely rescale the intensity values
        else:
            new_image_np = (image_np - min_i_r) / range_i_r * range_i_o + min_i_o

        # add the new SimpleITK image to the PyRaDiSe image
        new_image_sitk = sitk.GetImageFromArray(new_image_np)
        new_image_sitk.CopyInformation(image_sitk)
        image.set_image_data(new_image_sitk)

        return image

    def execute(self, subject: Subject, params: RescaleIntensityFilterParams) -> Subject:
        """Execute the rescaling procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be rescaled.
            params (RescaleIntensityFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with rescaled
            :class:`~pyradise.data.image.IntensityImage` entries.
        """
        return super().execute(subject, params)

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Execute the inverse rescaling procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be inversely rescaled.
            transform_info (TransformInfo): The transform information.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with inversely rescaled
            :class:`~pyradise.data.image.IntensityImage` entries.
        """
        return super().execute_inverse(subject, transform_info)


class ClipIntensityFilterParams(IntensityFilterParams):
    """A filter parameter class for the :class:`~pyradise.process.intensity.ClipIntensityFilter` class.

    Args:
        min_out (float): The minimum intensity value of the processed image.
        max_out (float): The maximum intensity value of the processed image.
        modalities (Optional[Tuple[Union[Modality, str], ...]]): The modalities of the images to be clipped. If
         ``None`` is provided all images of the provided subject are clipped (default: None).
    """

    def __init__(
        self, min_out: float, max_out: float, modalities: Optional[Tuple[Union[Modality, str], ...]] = None
    ) -> None:
        super().__init__(modalities)

        # check the provided min and max values
        if min_out == max_out:
            raise ValueError(
                "The min and max output intensity values must not be equal because the resulting image "
                "will have constant intensity."
            )

        if min_out > max_out:
            min_out, max_out = max_out, min_out

        self.min_value: float = min_out
        self.max_value: float = max_out


class ClipIntensityFilter(IntensityFilter):
    """A filter class performing a clipping of intensity values on all selected
    :class:`~pyradise.data.image.IntensityImage` instances of the provided :class:`~pyradise.data.subject.Subject`
    instance. The clipping procedure sets the intensity values outside the specified range to the specified minimum
    and maximum values.

    For the clipping procedure the following formula is applied to the image data:

    .. math::
        I_{out} = \\begin{cases}
            min_{out} & I_{in} < min_{out} \\\\
            max_{out} & I_{in} > max_{out} \\\\
            I_{in} & min_{out} \\leq I_{in} \\leq max_{out}
        \\end{cases}

    Note:
        The clipping procedure causes a loss of information which can not be recovered. Thus, the clipping procedure
        is not invertible.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Check if the filter is invertible.

        Returns:
            bool: False because the clipping procedure is not invertible.
        """
        return False

    def _process_image(self, image: IntensityImage, params: ClipIntensityFilterParams) -> IntensityImage:
        """Apply the clipping to the provided image.

        Args:
            image (IntensityImage): The image to be processed.
            params (ClipIntensityFilterParams): The filter parameters.

        Returns:
            IntensityImage: The processed image.
        """
        # get the image data
        sitk_image = image.get_image_data()

        # apply the clipping
        clipped_image_sitk = sitk.Clamp(sitk_image, sitk_image.GetPixelIDValue(), params.min_value, params.max_value)

        # add the clipped SimpleITK image to the image
        image.set_image_data(clipped_image_sitk)

        # track the necessary information
        self._register_tracked_data(image, sitk_image, clipped_image_sitk, params)

        return image

    def _process_image_inverse(self, image: IntensityImage, transform_info: TransformInfo) -> IntensityImage:
        """Return the provided image without any processing because the clipping procedure is not invertible.

        Args:
            image (IntensityImage): The image to be returned.
            transform_info (TransformInfo): The transform information.

        Returns:
            IntensityImage: The provided image.
        """
        return image

    def execute(self, subject: Subject, params: ClipIntensityFilterParams) -> Subject:
        """Execute the clipping procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            params (ClipIntensityFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with clipped
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        return super().execute(subject, params)

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Return the provided subject without any processing because the clipping procedure is not invertible.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be returned.
            transform_info (TransformInfo): The transform information.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The provided :class:`~pyradise.data.subject.Subject` instance.
        """

        return super().execute_inverse(subject, transform_info)


class GaussianFilterParams(IntensityFilterParams):
    """A filter parameter class for the :class:`~pyradise.process.intensity.GaussianFilter` class.

    Args:
        variance (float): The variance of the Gaussian kernel.
        kernel_size (int): The kernel size of the Gaussian kernel.
        modalities (Optional[Tuple[Union[Modality, str], ...]]): The modalities of the images to be filtered. If
         ``None`` is provided all images of the provided subject are filtered (default: None).
    """

    def __init__(
        self, variance: float, kernel_size: int, modalities: Optional[Tuple[Union[Modality, str], ...]] = None
    ) -> None:
        super().__init__(modalities)

        # check the statistical values
        if variance <= 0:
            raise ValueError("The variance must be greater than zero.")

        if kernel_size <= 0:
            raise ValueError("The kernel size must be greater than zero.")

        self.variance = variance
        self.kernel_size = kernel_size


class GaussianFilter(IntensityFilter):
    """A filter class performing a Gaussian smoothing on all :class:`~pyradise.data.image.IntensityImage` instances of
    the provided :class:`~pyradise.data.subject.Subject` instance.

    Reference:
        The implementation is based on the SimpleITK implementation of the `SimpleITK DiscreteGaussianImageFilter
        <https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1DiscreteGaussianImageFilter.html>`_.

    Note:
        The Gaussian smoothing procedure is not invertible.
    """

    def is_invertible(self) -> bool:
        """Check if the filter is invertible.

        Returns:
            bool: False because the Gaussian filter is not invertible.
        """
        return False

    def _process_image(self, image: IntensityImage, params: GaussianFilterParams) -> IntensityImage:
        """Apply the Gaussian filter to the provided image.

        Args:
            image (IntensityImage): The image to be filtered.
            params (GaussianFilterParams): The filter parameters.

        Returns:
            IntensityImage: The Gaussian filtered image.
        """
        # get the image data as sitk image
        image_sitk = image.get_image_data()

        # cast the image if necessary
        if "integer" in image_sitk.GetPixelIDTypeAsString():
            image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)

        # apply the gaussian filter
        gaussian_filter = sitk.DiscreteGaussianImageFilter()
        gaussian_filter.SetVariance(params.variance)
        gaussian_filter.SetMaximumKernelWidth(params.kernel_size)
        gaussian_filter.SetUseImageSpacing(True)
        new_image_sitk = gaussian_filter.Execute(image_sitk)

        # add the new SimpleITK image to the PyRaDiSe image
        image.set_image_data(new_image_sitk)

        # track the necessary data
        self._register_tracked_data(image, image_sitk, new_image_sitk, params)

        return image

    def _process_image_inverse(self, image: IntensityImage, transform_info: TransformInfo) -> IntensityImage:
        """Return the provided image because the Gaussian filter is not invertible.

        Args:
            image (IntensityImage): The image to be returned.
            transform_info (TransformInfo): The transform information.

        Returns:
            IntensityImage: The provided image.
        """
        return image

    def execute(self, subject: Subject, params: GaussianFilterParams) -> Subject:
        """Execute the Gaussian filter.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be filtered.
            params (GaussianFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with Gaussian filtered
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        for image in subject.get_images():
            if isinstance(image, IntensityImage):
                self._process_image(image, params)

        return subject

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Return the provided subject without any processing because the Gaussian filtering procedure is not
        invertible.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be returned.
            transform_info (TransformInfo): The transform information.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The provided :class:`~pyradise.data.subject.Subject` instance.
        """
        return super().execute_inverse(subject, transform_info)


class MedianFilterParams(IntensityFilterParams):
    """A filter parameter class for the :class:`~pyradise.process.intensity.MedianFilter` class.

    Args:
        radius (int): The radius of the median filter.
        modalities (Optional[Tuple[Union[Modality, str], ...]]): The modalities of the images to be filtered. If
         ``None`` is provided all images of the provided subject are filtered (default: None).
    """

    def __init__(self, radius: int, modalities: Optional[Tuple[Union[Modality, str], ...]] = None) -> None:
        super().__init__(modalities)

        self.radius = radius


class MedianFilter(IntensityFilter):
    """A filter class performing a median filtering on all :class:`~pyradise.data.image.IntensityImage` instances of
    the provided :class:`~pyradise.data.subject.Subject` instance.

    Reference:
        The implementation is based on the SimpleITK implementation of the `SimpleITK MedianImageFilter
        <https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1MedianImageFilter.html>`_.

    Note:
        The median filter is not invertible.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Check if the filter is invertible.

        Returns:
            bool: False because the median filter is not invertible.
        """
        return False

    def _process_image(self, image: IntensityImage, params: MedianFilterParams) -> IntensityImage:
        """Apply the median filter to the provided image.

        Args:
            image (IntensityImage): The image to be filtered.
            params (MedianFilterParams): The filter parameters.

        Returns:
            IntensityImage: The median filtered image.
        """
        # get the image data as sitk image
        image_sitk = image.get_image_data()

        # cast the image if necessary
        if "integer" in image_sitk.GetPixelIDTypeAsString():
            image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)

        # apply the median filter
        median_filter = sitk.MedianImageFilter()
        median_filter.SetRadius(params.radius)
        new_image_sitk = median_filter.Execute(image.get_image_data())

        # add the new SimpleITK image to the PyRaDiSe image
        image.set_image_data(new_image_sitk)

        # track the necessary data
        self._register_tracked_data(image, image_sitk, new_image_sitk, params)

        return image

    def _process_image_inverse(self, image: IntensityImage, transform_info: TransformInfo) -> IntensityImage:
        """Return the provided image because the median filter is not invertible.

        Args:
            image (IntensityImage): The image to be returned.
            transform_info (TransformInfo): The transform information.

        Returns:
            IntensityImage: The provided image.
        """
        return image

    def execute(self, subject: Subject, params: MedianFilterParams) -> Subject:
        """Execute the median filter.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be filtered.
            params (MedianFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with filtered
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        return super().execute(subject, params)

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Return the provided subject without any processing because the median filtering procedure is not
        invertible.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be returned.
            transform_info (TransformInfo): The transform information.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The provided :class:`~pyradise.data.subject.Subject` instance.
        """
        return super().execute_inverse(subject, transform_info)


class LaplacianFilterParams(IntensityFilterParams):
    """A filter parameter class for the :class:`~pyradise.process.intensity.LaplacianFilter` class.

    Args:
        modalities (Optional[Tuple[Union[Modality, str], ...]]): The modalities of the images to be filtered. If
         ``None`` is provided all images of the provided subject are filtered (default: None).
    """

    def __init__(self, modalities: Optional[Tuple[Union[Modality, str], ...]] = None) -> None:
        super().__init__(modalities)


class LaplacianFilter(IntensityFilter):
    """A filter class performing a Laplacian sharpening on all :class:`~pyradise.data.image.IntensityImage` instances of
    the provided :class:`~pyradise.data.subject.Subject` instance.

    Reference:
        The implementation is based on the SimpleITK implementation of the `SimpleITK LaplacianSharpeningImageFilter
        <https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1LaplacianSharpeningImageFilter.html>`_.

    Note:
        The Laplacian filter is not invertible.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Check if the filter is invertible.

        Returns:
            bool: False because the Laplacian filter is not invertible.
        """
        return False

    def _process_image(self, image: IntensityImage, params: LaplacianFilterParams) -> IntensityImage:
        """Apply the Laplacian sharpening filter to the provided image.

        Args:
            image (IntensityImage): The image to be filtered.
            params (LaplacianFilterParams): The filter parameters.

        Returns:
            IntensityImage: The Laplacian filtered image.
        """
        # get the image data as sitk image
        image_sitk = image.get_image_data()

        # cast the image if necessary
        if "integer" in image_sitk.GetPixelIDTypeAsString():
            image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)

        # apply the laplace filter
        laplacian_filter = sitk.LaplacianSharpeningImageFilter()
        laplacian_filter.SetUseImageSpacing(True)
        new_image_sitk = laplacian_filter.Execute(image_sitk)

        # add the new SimpleITK image to the PyRaDiSe image
        image.set_image_data(new_image_sitk)

        # track the necessary data
        self._register_tracked_data(image, image_sitk, new_image_sitk, params)

        return image

    def _process_image_inverse(self, image: IntensityImage, transform_info: TransformInfo) -> IntensityImage:
        """Return the provided image because the Laplacian filter is not invertible.

        Args:
            image (IntensityImage): The image to be returned.
            transform_info (TransformInfo): The transform information.

        Returns:
            IntensityImage: The provided image.
        """
        return image

    def execute(self, subject: Subject, params: Optional[LaplacianFilterParams] = None) -> Subject:
        """Execute the Laplacian sharpening filter.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be filtered.
            params (Optional[LaplacianFilterParams]): The unused filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with Laplace filtered
            :class:`~pyradise.data.image.IntensityImage` instances.

        Note:
            The Laplacian filter does not need any parameters.
        """
        return super().execute(subject, params)

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Return the provided subject without any processing because the Laplace filtering procedure is not
        invertible.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be returned.
            transform_info (TransformInfo): The transform information.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The provided :class:`~pyradise.data.subject.Subject` instance.
        """
        return super().execute_inverse(subject, transform_info)
