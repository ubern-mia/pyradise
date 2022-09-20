from abc import abstractmethod
from typing import (
    Any,
    Union,
    Optional)

import numpy as np
import SimpleITK as sitk

from pyradise.data import (
    Subject,
    IntensityImage,
    TransformInfo)
from .base import (
    Filter,
    LoopEntryFilter,
    LoopEntryFilterParams,
    FilterParams)


__all__ = ['IntensityFilter', 'IntensityLoopFilter', 'IntensityLoopFilterParams', 'ZScoreNormFilter',
           'ZeroOneNormFilter', 'RescaleIntensityFilter', 'RescaleIntensityFilterParams',
           'ClipIntensityFilter', 'ClipIntensityFilterParams', 'GaussianFilter', 'GaussianFilterParams',
           'MedianFilter', 'MedianFilterParams', 'LaplacianFilter', 'LaplacianFilterParams']

# TODO CONTINUE HERE
# pylint: disable=too-few-public-methods
class IntensityLoopFilterParams(LoopEntryFilterParams):
    """A filter parameter class for CONTINUE HERE

    A basic intensity filter parameter class.

    Args:
        loop_axis (Optional[int]): The axis along which the intensity change is performed. If None, the intensity change
         is performed on the whole image at once. If a value is given, the intensity change is performed by looping
         over the corresponding image dimension.
    """

    def __init__(self,
                 loop_axis: Optional[int]
                 ) -> None:
        super().__init__(loop_axis)


class IntensityFilter(Filter):
    """An abstract base class for intensity modifying filters processing the whole image content directly.
    """

    @abstractmethod
    def _process_image(self,
                       image: IntensityImage,
                       params: FilterParams
                       ) -> IntensityImage:
        raise NotImplementedError()

    @abstractmethod
    def _process_image_inverse(self,
                               image: IntensityImage,
                               transform_info: TransformInfo) -> IntensityImage:
        raise NotImplementedError()

    def execute(self,
                subject: Subject,
                params: IntensityLoopFilterParams
                ) -> Subject:
        """Execute the intensity modifying procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            params (FilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with processed
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        for image in subject.get_images():
            if isinstance(image, IntensityImage):
                self._process_image(image, params)

        return subject

    def execute_inverse(self,
                        subject: Subject,
                        transform_info: TransformInfo
                        ) -> Subject:
        """Execute the inverse intensity modifying procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            transform_info (TransformInfo): The transform information.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with processed
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        for image in subject.get_images():
            if isinstance(image, IntensityImage):
                self._process_image_inverse(image, transform_info)

        return subject


class IntensityLoopFilter(LoopEntryFilter):
    """An abstract base class for intensity modifying filters that may process the image content using looping over an
    axis. """

    def __init__(self):
        super().__init__()

        # provides an index for the position along the loop axis
        self.loop_axis_pos_idx = 0

    @abstractmethod
    def _modify_array(self,
                      array: np.ndarray,
                      params: IntensityLoopFilterParams
                      ) -> np.ndarray:
        """The intensity modification function which is applied to the provided array. The provided array can be of
        n-dimensions whereby the dimensionality depend on the provided data and the ``loop_axis`` parameter as
        specified in the appropriate :class:`IntensityFilterParams` instance.

        Args:
            array (np.ndarray): The array to be processed.
            params (IntensityLoopFilterParams): The parameters used for the processing.

        Returns:
            np.ndarray: The processed array.
        """
        raise NotImplementedError()

    @abstractmethod
    def _modify_array_inverse(self,
                              array: np.ndarray,
                              params: TransformInfo
                              ) -> np.ndarray:
        raise NotImplementedError()

    def _process_image(self,
                       image: IntensityImage,
                       params: Union[IntensityLoopFilterParams, Any]
                       ) -> IntensityImage:
        """Execute the intensity modifying procedure on the provided image by looping over the image accordingly.

        Args:
            image (IntensityImage): The image to be processed.
            params (IntensityLoopFilterParams): The filter parameters.

        Returns:
            IntensityImage: The processed image.
        """

        # set the loop axis position index to zero because of processing a new image
        self.loop_axis_pos_idx = 0

        # get the image data for computation
        image_sitk = image.get_image_data(as_sitk=True)
        if 'integer' in image_sitk.GetPixelIDTypeAsString():
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
        transform_info = self._create_transform_info(image_sitk, new_image_sitk, params, self.filter_args,
                                                     self.tracking_data)
        image.add_transform_info(transform_info)
        self.tracking_data.clear()

        return image

    def _process_image_inverse(self,
                               image: IntensityImage,
                               transform_info: TransformInfo) -> IntensityImage:
        # return the image as is if the filter is not invertible
        if not self.is_invertible():
            return image

        # set the loop axis position index to zero because of processing a new image
        self.loop_axis_pos_idx = 0

        # get the image data for inverse processing
        image_sitk = image.get_image_data(as_sitk=True)
        if 'integer' in image_sitk.GetPixelIDTypeAsString():
            image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)
        image_np = sitk.GetArrayFromImage(image_sitk)

        # perform the inverse intensity modifying procedure
        new_image_np = self.loop_entries(image_np, transform_info, self._modify_array_inverse,
                                         transform_info.params.loop_axis)

        # construct the new image
        new_image_sitk = sitk.GetImageFromArray(new_image_np)
        new_image_sitk.CopyInformation(image_sitk)

        # set the new image data to the image
        image.set_image_data(new_image_sitk)

        return image

    def execute(self,
                subject: Subject,
                params: IntensityLoopFilterParams
                ) -> Subject:
        """Execute the intensity modifying procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            params (FilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with processed
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        for image in subject.get_images():
            if isinstance(image, IntensityImage):
                self._process_image(image, params)

        return subject

    def execute_inverse(self,
                        subject: Subject,
                        transform_info: TransformInfo
                        ) -> Subject:
        """Execute the inverse intensity modifying procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            transform_info (TransformInfo): The transform information.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with processed
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        for image in subject.get_images():
            if isinstance(image, IntensityImage):
                self._process_image_inverse(image, transform_info)

        return subject


class ZScoreNormFilter(IntensityLoopFilter):
    """A normalization filter class performing a z-score normalization on all
    :class:`~pyradise.data.image.IntensityImage` instances of the provided :class:`~pyradise.data.subject.Subject`
    instance."""

    @staticmethod
    def is_invertible() -> bool:
        """Check if the filter is invertible.

        Returns:
            bool: False because the z-score normalization procedure is not invertible.
        """
        return False

    def _modify_array(self, array: np.ndarray, params: Any) -> np.ndarray:
        """The z-score normalization function which will be applied to the provided array.

        Args:
            array (np.ndarray): The array to be normalized.
            params (Any): The parameters used for the normalization.

        Returns:
            np.ndarray: The z-score normalized array.
        """
        return (array - np.mean(array)) / np.std(array)

    def _modify_array_inverse(self,
                              array: np.ndarray,
                              params: IntensityLoopFilterParams
                              ) -> np.ndarray:
        return array


class ZeroOneNormFilter(IntensityLoopFilter):

    @staticmethod
    def is_invertible() -> bool:
        """Check if the filter is invertible.

        Returns:
            bool: True because the zero to one normalization procedure is invertible.
        """
        return True

    def _modify_array(self, array: np.ndarray, params: Any) -> np.ndarray:
        """The 0-1 normalization function which will be applied to the provided array.

        Args:
            array (np.ndarray): The array to be normalized.
            params (Any): The parameters used for the normalization.

        Returns:
            np.ndarray: The min-max normalized array.
        """

        self.tracking_data[f'min_{self.loop_axis_pos_idx}'] = np.min(array)
        self.tracking_data[f'max_{self.loop_axis_pos_idx}'] = np.max(array)

        self.loop_axis_pos_idx += 1

        return (array - np.min(array)) / (np.max(array) - np.min(array))

    def _modify_array_inverse(self, array: np.ndarray, params: TransformInfo) -> np.ndarray:
        """The inverse 0-1 normalization function which will be applied to the provided array.

        Args:
            array (np.ndarray): The array to be normalized.
            params (Any): The parameters used for the normalization.

        Returns:
            np.ndarray: The min-max normalized array.
        """
        original_min = params.get_data(f'min_{self.loop_axis_pos_idx}')
        original_max = params.get_data(f'max_{self.loop_axis_pos_idx}')

        self.loop_axis_pos_idx += 1

        return array * (original_max - original_min) + original_min


# pylint: disable=too-few-public-methods
class RescaleIntensityFilterParams(FilterParams):
    """A rescaling normalization filter parameter class.

    Args:
        min_value (Optional[float]): The minimum value of the rescaled image. If ``None`` is provided the filter takes
         the minimum intensity value of the image.
        max_value (Optional[float]): The maximum value of the rescaled image. If ``None`` is provided the filter takes
         the maximum intensity value of the image.
        loop_axis (Optional[int]): The axis along which the rescaling is performed. If None, the rescaling is
         performed on the whole image at once. If a value is given, the rescaling is performed by looping over the
         corresponding image dimension (default: None).
    """

    def __init__(self,
                 min_value: Optional[float],
                 max_value: Optional[float],
                 loop_axis: Optional[int] = None
                 ) -> None:
        super().__init__()

        self.loop_axis: Optional[int] = loop_axis
        self.min_value: Optional[float] = min_value
        self.max_value: Optional[float] = max_value


class RescaleIntensityFilter(IntensityLoopFilter):
    """An intensity rescaling filter class performing a rescaling on all
    :class:`~pyradise.data.image.IntensityImage` instances of the provided :class:`~pyradise.data.subject.Subject`
    instance.

    """

    @staticmethod
    def is_invertible() -> bool:
        """Check if the filter is invertible.

        Returns:
            bool: True because the rescaling procedure is invertible.
        """
        return True

    def _modify_array(self,
                      array: np.ndarray,
                      params: RescaleIntensityFilterParams
                      ) -> np.ndarray:
        """The rescaling function which will be applied to the provided array.

        Args:
            array (np.ndarray): The array to be rescaled.
            params (Any): The parameters used for the rescaling.

        Returns:
            np.ndarray: The array with rescaled intensity values.
        """
        if params.min_value == params.max_value:
            raise ValueError("The specified min and max value is equal. The resulting image would have constant "
                             "intensity.")

        if params.min_value > params.max_value:
            raise ValueError("The min value is larger than the max value. Please adjust the parameters accordingly.")

        param_range = params.max_value - params.min_value

        min_arr_value = np.min(array)
        max_arr_value = np.max(array)
        arr_range_ = max_arr_value - min_arr_value

        self.tracking_data[f'min_{self.loop_axis_pos_idx}'] = min_arr_value
        self.tracking_data[f'max_{self.loop_axis_pos_idx}'] = max_arr_value
        self.loop_axis_pos_idx += 1

        if arr_range_ < 1e-10:
            raise ValueError("The range of the image intensity values is too small to perform a rescaling.")

        return (array - min_arr_value) / arr_range_ * param_range + params.min_value

    def _modify_array_inverse(self,
                              array: np.ndarray,
                              params: TransformInfo
                              ) -> np.ndarray:
        """The inverse rescaling function which will be applied to the provided array.

        Args:
            array (np.ndarray): The array to be rescaled.
            params (Any): The parameters used for the rescaling.

        Returns:
            np.ndarray: The array with rescaled intensity values.
        """
        min_arr_value = np.min(array)
        max_arr_value = np.max(array)
        arr_range_ = max_arr_value - min_arr_value

        min_out_value = params.get_data(f'min_{self.loop_axis_pos_idx}')
        max_out_value = params.get_data(f'max_{self.loop_axis_pos_idx}')
        out_range = max_out_value - min_out_value

        self.loop_axis_pos_idx += 1

        return (array - min_arr_value) / arr_range_ * out_range + min_out_value

    def execute(self,
                subject: Subject,
                params: RescaleIntensityFilterParams
                ) -> Subject:
        """Execute the rescaling procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be rescaled.
            params (RescaleIntensityFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with rescaled
            :class:`~pyradise.data.image.IntensityImage` entries.
        """
        for image in subject.get_images():
            if isinstance(image, IntensityImage):
                self._process_image(image, params)

        return subject


class ClipIntensityFilterParams(FilterParams):
    """An intensity clipping filter parameter class.

    Args:
        min_value (float): The minimum intensity value of the processed image.
        max_value (float): The maximum intensity value of the processed image.
    """

    def __init__(self,
                 min_value: float,
                 max_value: float
                 ) -> None:
        super().__init__()

        self.loop_axis = None
        self.min_value: float = min_value
        self.max_value: float = max_value


class ClipIntensityFilter(IntensityLoopFilter):
    """An intensity clipping filter class performing a clipping on all :class:`~pyradise.data.image.IntensityImage`
    instances of the provided :class:`~pyradise.data.subject.Subject` instance. This filter clips the intensity values
    of each image to a specified range of values and sets all values outside the range to the specified range limits.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Check if the filter is invertible.

        Returns:
            bool: False because the clipping procedure is not invertible.
        """
        return False

    def _modify_array(self, array: np.ndarray, params: ClipIntensityFilterParams) -> np.ndarray:
        """The intensity clipping function which will be applied to the provided array.

        Args:
            array (np.ndarray): The array to be clipped.
            params (Any): The parameters used for the clipping.

        Returns:
            np.ndarray: The array with clipped intensity values.
        """
        if params.min_value == params.max_value:
            raise ValueError("The specified min and max value is equal. The resulting image would have constant "
                             "intensity.")

        if params.min_value > params.max_value:
            raise ValueError("The min value is larger than the max value. Please adjust the parameters accordingly.")

        return np.clip(array, params.min_value, params.max_value)

    def _modify_array_inverse(self,
                              array: np.ndarray,
                              params: TransformInfo
                              ) -> np.ndarray:
        """Returns the provided array because the clipping procedure is not invertible.

        Args:
            array (np.ndarray): The array to be returned.
            params (TransformInfo): The parameters used for the clipping.

        Returns:
            np.ndarray: The provided array.
        """
        return array

    def execute(self,
                subject: Subject,
                params: ClipIntensityFilterParams
                ) -> Subject:
        """Execute the clipping procedure on the intensity images.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be clipped.
            params (ClipIntensityFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with clipped
            :class:`~pyradise.data.image.IntensityImage` entries.
        """
        for image in subject.get_images():
            if isinstance(image, IntensityImage):
                self._process_image(image, params)

        return subject


class GaussianFilterParams(FilterParams):

    def __init__(self,
                 variance: float,
                 kernel_size: int
                 ) -> None:
        super().__init__()

        self.variance = variance
        self.kernel_size = kernel_size


class GaussianFilter(IntensityFilter):

    def is_invertible(self) -> bool:
        """Check if the filter is invertible.

        Returns:
            bool: False because the Gaussian filter is not invertible.
        """
        return False

    def _process_image(self,
                       image: IntensityImage,
                       params: Union[GaussianFilterParams, Any]
                       ) -> IntensityImage:
        """Apply the Gaussian filter to the provided image.

        Args:
            image (IntensityImage): The image to be filtered.
            params (Union[GaussianFilterParams, Any]): The filter parameters.

        Returns:
            IntensityImage: The gaussian filtered image.
        """
        image_sitk = image.get_image_data(True)
        if 'integer' in image_sitk.GetPixelIDTypeAsString():
            image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)

        gaussian_filter = sitk.DiscreteGaussianImageFilter()
        gaussian_filter.SetVariance(params.variance)
        gaussian_filter.SetMaximumKernelWidth(params.kernel_size)
        gaussian_filter.SetUseImageSpacing(True)
        new_image_sitk = gaussian_filter.Execute(image_sitk)

        image.set_image_data(new_image_sitk)

        transform_info = self._create_transform_info(image_sitk, new_image_sitk, params, self.filter_args,
                                                     self.tracking_data)
        image.add_transform_info(transform_info)
        self.tracking_data.clear()

        return image

    def _process_image_inverse(self,
                               image: IntensityImage,
                               transform_info: TransformInfo
                               ) -> IntensityImage:
        """Returns the provided image because the Gaussian filter is not invertible.

        Args:
            image (IntensityImage): The image to be returned.
            transform_info (TransformInfo): The transform information used for the Gaussian filter.

        Returns:
            IntensityImage: The provided image.
        """
        return image

    def execute(self,
                subject: Subject,
                params: GaussianFilterParams
                ) -> Subject:
        """Execute the Gaussian filter on the intensity images.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be filtered.
            params (GaussianFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with filtered
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        for image in subject.get_images():
            if isinstance(image, IntensityImage):
                self._process_image(image, params)

        return subject


class MedianFilterParams(FilterParams):

    def __init__(self, radius: int) -> None:
        super().__init__()

        self.radius = radius


class MedianFilter(IntensityFilter):

    @staticmethod
    def is_invertible() -> bool:
        """Check if the filter is invertible.

        Returns:
            bool: False because the median filter is not invertible.
        """
        return False

    def _process_image(self,
                       image: IntensityImage,
                       params: MedianFilterParams
                       ) -> IntensityImage:
        image_sitk = image.get_image_data(True)
        if 'integer' in image_sitk.GetPixelIDTypeAsString():
            image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)

        median_filter = sitk.MedianImageFilter()
        median_filter.SetRadius(params.radius)
        new_image_sitk = median_filter.Execute(image.get_image_data(True))

        image.set_image_data(new_image_sitk)
        transform_info = self._create_transform_info(image_sitk, new_image_sitk, params, self.filter_args,
                                                     self.tracking_data)
        image.add_transform_info(transform_info)
        self.tracking_data.clear()

        return image


    def _process_image_inverse(self,
                               image: IntensityImage,
                               transform_info: TransformInfo
                               ) -> IntensityImage:
        """Returns the provided image because the median filter is not invertible.

        Args:
            image (IntensityImage): The image to be returned.
            transform_info (TransformInfo): The transform information used for the median filter.

        Returns:
            IntensityImage: The provided image.
        """
        return image

    def execute(self,
                subject: Subject,
                params: MedianFilterParams) -> Subject:
        """Execute the median filter on the intensity images.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be filtered.
            params (MedianFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with filtered
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        for image in subject.get_images():
            if isinstance(image, IntensityImage):
                self._process_image(image, params)

        return subject


class LaplacianFilterParams(FilterParams):
    """The parameters for the Laplacian filter.

    Note:
        The Laplacian filter does not need any parameters.
    """

    def __init__(self) -> None:
        super().__init__()


class LaplacianFilter(IntensityFilter):

    @staticmethod
    def is_invertible() -> bool:
        """Check if the filter is invertible.

        Returns:
            bool: False because the Laplacian filter is not invertible.
        """
        return False

    def _process_image(self,
                       image: IntensityImage,
                       params: LaplacianFilterParams
                       ) -> IntensityImage:
        """Apply the Laplacian filter to the provided image.

        Args:
            image (IntensityImage): The image to be filtered.
            params (LaplacianFilterParams): The filter parameters.

        Returns:
            IntensityImage: The Laplacian filtered image.
        """
        image_sitk = image.get_image_data(True)
        if 'integer' in image_sitk.GetPixelIDTypeAsString():
            image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)

        laplacian_filter = sitk.LaplacianImageFilter()
        laplacian_filter.SetUseImageSpacing(True)
        new_image_sitk = laplacian_filter.Execute(image_sitk)

        image.set_image_data(new_image_sitk)

        transform_info = self._create_transform_info(image_sitk, new_image_sitk, params, self.filter_args,
                                                     self.tracking_data)
        image.add_transform_info(transform_info)
        self.tracking_data.clear()

        return image

    def _process_image_inverse(self,
                               image: IntensityImage,
                               transform_info: TransformInfo
                               ) -> IntensityImage:
        """Returns the provided image because the Laplacian filter is not invertible.

        Args:
            image (IntensityImage): The image to be returned.
            transform_info (TransformInfo): The transform information used for the Laplacian filter.

        Returns:
            IntensityImage: The provided image.
        """
        return image

    def execute(self, subject: Subject, params: Optional[LaplacianFilterParams] = None) -> Subject:
        """Execute the median filter on the intensity images.

        Note:
            The Laplacian filter does not need any parameters.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be filtered.
            params (MedianFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with filtered
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        for image in subject.get_images():
            if isinstance(image, IntensityImage):
                self._process_image(image, params)

        return subject
