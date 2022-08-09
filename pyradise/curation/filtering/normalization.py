from abc import abstractmethod
from typing import (
    Any,
    Optional)

import numpy as np
import SimpleITK as sitk

from pyradise.curation.data import (
    Subject,
    IntensityImage)
from .base import (
    LoopEntryFilter,
    FilterParameters)


# pylint: disable=too-few-public-methods

class NormalizationFilterParameters(FilterParameters):
    """A class representing the filter parameters for a NormalizationFilter."""

    def __init__(self,
                 loop_axis: Optional[int]
                 ) -> None:
        super().__init__()

        self.loop_axis = loop_axis


class NormalizationFilter(LoopEntryFilter):
    """A class representing a filter for normalizing intensity images."""

    @staticmethod
    @abstractmethod
    def _normalize_array(array: np.ndarray,
                         _params: Any
                         ) -> np.ndarray:
        raise NotImplementedError()

    def _normalize_image(self,
                         image: IntensityImage,
                         params: NormalizationFilterParameters
                         ) -> IntensityImage:
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
        """Executes the normalization procedure.

        Args:
            subject (Subject): The subject to be normalized.
            params (Optional[FilterParameters]): The filter parameters.

        Returns:
            Subject: The subject containing the normalized intensity images.
        """
        for image in subject.intensity_images:
            new_image = self._normalize_image(image, params)
            subject.replace_image(image, new_image)

        return subject


class ZScoreNormalizationFilter(NormalizationFilter):
    """A filter class performing a z-score normalization."""

    @staticmethod
    def _normalize_array(array: np.ndarray, _params: Any) -> np.ndarray:
        return (array - np.mean(array)) / np.std(array)


class MinMaxNormalizationFilter(NormalizationFilter):
    """A filter class performing a min-max (0-1) normalization."""

    @staticmethod
    def _normalize_array(array: np.ndarray, _params: Any) -> np.ndarray:
        return (array - np.min(array)) / (np.max(array) - np.min(array))
