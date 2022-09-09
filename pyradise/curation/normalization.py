from abc import abstractmethod
from typing import (
    Any,
    Optional)

import numpy as np
import SimpleITK as sitk

from pyradise.data import (
    Subject,
    IntensityImage)
from .base import (
    LoopEntryFilter,
    FilterParameters)


__all__ = ['NormalizationFilter', 'NormalizationFilterParameters', 'ZScoreNormalizationFilter',
           'MinMaxNormalizationFilter']


# pylint: disable=too-few-public-methods
class NormalizationFilterParameters(FilterParameters):
    """An abstract normalization filter parameter class.

    Args:
        loop_axis (Optional[int]): The axis along which the normalization is performed. If None, the normalization is
         performed on the whole image at once. If a value is given, the normalization is performed by looping over the
         corresponding image dimension.
    """

    def __init__(self,
                 loop_axis: Optional[int]
                 ) -> None:
        super().__init__()

        self.loop_axis: Optional[int] = loop_axis


class NormalizationFilter(LoopEntryFilter):
    """An abstract normalization filter class for building intensity image normalization filters."""

    @staticmethod
    @abstractmethod
    def _normalize_array(array: np.ndarray,
                         _params: Any
                         ) -> np.ndarray:
        """The normalization function which is applied to the provided array. The provided array can be of n-dimensions
        whereby the dimensionality depend on the provided data and the ``loop_axis`` parameter as specified in the
        appropriate :class:`NormalizationFilterParameters` instance.

        Args:
            array (np.ndarray): The array to be normalized.
            _params (Any): The parameters used for the normalization.

        Returns:
            np.ndarray: The normalized array.
        """
        raise NotImplementedError()

    def _normalize_image(self,
                         image: IntensityImage,
                         params: NormalizationFilterParameters
                         ) -> IntensityImage:
        """Execute the normalization procedure on the provided image by looping over the image accordingly.

        Args:
            image (IntensityImage): The image to be normalized.
            params (NormalizationFilterParameters): The filter parameters.

        Returns:
            IntensityImage: The normalized image.
        """
        image_sitk = image.get_image(as_sitk=True)
        image_np = sitk.GetArrayFromImage(image_sitk)

        new_image_np = self.loop_entries(image_np, None, self._normalize_array, params.loop_axis)

        new_image_sitk = sitk.GetImageFromArray(new_image_np)
        new_image_sitk.CopyInformation(image_sitk)

        image.set_image(new_image_sitk)

        return image

    def execute(self,
                subject: Subject,
                params: NormalizationFilterParameters
                ) -> Subject:
        """Execute the normalization procedure.

        Args:
            subject (Subject): The :class:`Subject` instance to be normalized.
            params (Optional[FilterParameters]): The filter parameters.

        Returns:
            Subject: The :class:`Subject` instance with normalized :class:`IntensityImage` entries.
        """
        for image in subject.intensity_images:
            new_image = self._normalize_image(image, params)
            subject.replace_image(image, new_image)

        return subject


class ZScoreNormalizationFilter(NormalizationFilter):
    """A normalization filter class performing a z-score normalization on all :class:`IntensityImage` entries of the
    provided :class:`Subject`."""

    @staticmethod
    def _normalize_array(array: np.ndarray, _params: Any) -> np.ndarray:
        """The z-score normalization function which will be applied to the provided array.

        Args:
            array (np.ndarray): The array to be normalized.
            _params (Any): The parameters used for the normalization.

        Returns:
            np.ndarray: The z-score normalized array.
        """
        return (array - np.mean(array)) / np.std(array)


class MinMaxNormalizationFilter(NormalizationFilter):
    """A normalization filter class performing a min-max (0-1) normalization on all :class:`IntensityImage` entries of
    the provided :class:`Subject`."""

    @staticmethod
    def _normalize_array(array: np.ndarray, _params: Any) -> np.ndarray:
        """The min-max normalization function which will be applied to the provided array.

        Args:
            array (np.ndarray): The array to be normalized.
            _params (Any): The parameters used for the normalization.

        Returns:
            np.ndarray: The min-max normalized array.
        """
        return (array - np.min(array)) / (np.max(array) - np.min(array))
