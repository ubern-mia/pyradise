import warnings
from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk

from pyradise.data import (IntensityImage, Organ, SegmentationImage, Subject,
                           TransformInfo)

from .base import Filter, FilterParams

__all__ = [
    "SingleConnectedComponentFilterParams",
    "SingleConnectedComponentFilter",
    "AlphabeticOrganSortingFilterParams",
    "AlphabeticOrganSortingFilter",
]


# pylint: disable = too-few-public-methods
class SingleConnectedComponentFilterParams(FilterParams):
    """A filter parameter class for the :class:`~pyradise.process.postprocess.SingleConnectedComponentFilter` class.

    Args:
        excluded_organs (Optional[Union[Organ, Tuple[Organ, ...]]]): The organs to be excluded from the connected
         component filtering. If ``None`` all :class:`~pyradise.data.image.SegmentationImage` instances will be
         filtered.
    """

    def __init__(self, excluded_organs: Optional[Union[Organ, Tuple[Organ, ...]]] = None) -> None:
        if isinstance(excluded_organs, Organ):
            self.excluded_organs = (excluded_organs,)
        elif excluded_organs is None:
            self.excluded_organs = tuple()
        else:
            self.excluded_organs = excluded_organs


class SingleConnectedComponentFilter(Filter):
    """A filter class for removing all but the largest connected component from the specified
    :class:`~pyradise.data.image.SegmentationImage` instances in the provided :class:`~pyradise.data.subject.Subject`
    instance.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Return whether the filter is invertible or not.

        Returns:
            bool: False because the :class:`~pyradise.process.postprocess.SingleConnectedComponentFilter` is
            not invertible.
        """
        return False

    def _get_single_label_images(
        self,
        image: SegmentationImage,
    ) -> Tuple[Tuple[sitk.Image, ...], Tuple[int, ...]]:
        """Splits a label image into multiple single/binary label images.

        Args:
            image (SegmentationImage): The image to separate into binary images.

        Returns:
            Tuple[Tuple[sitk.Image, ...], Tuple[int, ...]]: Returns the binary images and the original label indexes.
        """
        sitk_image = image.get_image_data()
        np_image = sitk.GetArrayFromImage(sitk_image)

        labels = self._get_unique_labels(sitk_image)

        unique_images = []
        unique_labels = []
        for label in labels:
            np_image_ = deepcopy(np_image)
            np_image_[np_image_ != label] = 0
            np_image_[np_image_ == label] = 1

            sitk_image_ = sitk.GetImageFromArray(np_image_)
            sitk_image_.CopyInformation(sitk_image)
            unique_images.append(sitk_image_)

            unique_labels.append(label)

        return tuple(unique_images), tuple(unique_labels)

    @staticmethod
    def _combine_images(images: Tuple[sitk.Image, ...]) -> sitk.Image:
        """Combines multiple label images to one image.

        Args:
            images (Tuple[sitk.Image, ...]): The images to combine.

        Returns:
            sitk.Image: The combined image.
        """
        if len(images) == 1:
            return images[0]

        final = sitk.GetArrayFromImage(images[0])
        for i in range(1, len(images)):
            np_image = sitk.GetArrayFromImage(images[i])
            np.putmask(final, np_image != 0, np_image)

        combined = sitk.GetImageFromArray(final)
        combined.CopyInformation(images[0])

        return combined

    @staticmethod
    def _get_single_connected_component_image(image: SegmentationImage, label: int) -> sitk.Image:
        """Removes all connected components except for the largest and adjusts the label index.

        Args:
            image (sitk.Image): The image to process.
            label (int): The label index of the output image.

        Returns:
            sitk.Image: The image with only one single connected component.
        """
        original_image = image.get_image_data()
        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_filter.SetFullyConnected(True)
        sitk_image = cc_filter.Execute(original_image)
        sitk_image = sitk.RelabelComponent(sitk_image, sortByObjectSize=True)
        sitk_image = sitk.BinaryThreshold(sitk_image, 1, 1)

        np_image = sitk.GetArrayFromImage(sitk_image)
        np_image[np_image == 1] = label
        sitk_image = sitk.GetImageFromArray(np_image)
        sitk_image.CopyInformation(original_image)

        return sitk_image

    @staticmethod
    def _get_unique_labels(image: sitk.Image, exclude_bg: bool = True) -> Tuple[int, ...]:
        image_np = sitk.GetArrayFromImage(image)
        unique_labels = np.unique(image_np)

        if exclude_bg:
            unique_labels = unique_labels[unique_labels != 0]

        return tuple(unique_labels)

    def execute(self, subject: Subject, params: SingleConnectedComponentFilterParams) -> Subject:
        """Execute the single connected component filter procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed.
            params (SingleConnectedComponentFilterParams): The filters parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with filtered
            :class:`~pyradise.data.image.SegmentationImage` instances.
        """
        for image in subject.segmentation_images:
            if image.get_organ() in params.excluded_organs:
                continue

            image_sitk = image.get_image_data()
            if image.is_binary():
                single_label_images = (image,)
                labels = self._get_unique_labels(image_sitk)
            else:
                single_label_images, labels = self._get_single_label_images(image)

            cc_images = []
            for single_label_image, label in zip(single_label_images, labels):
                single_cc_image = self._get_single_connected_component_image(single_label_image, label)
                cc_images.append(single_cc_image)

            if cc_images:
                cc_image = self._combine_images(tuple(cc_images))
                cc_image = sitk.Cast(cc_image, sitk.sitkUInt8)

                image.set_image_data(cc_image)

                self._register_tracked_data(image, image_sitk, image.get_image_data(), params)

        return subject

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Return the provided :class:`~pyradise.data.subject.Subject` instance without any processing because
        the single connected component filtering procedure is not invertible.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be returned.
            transform_info (TransformInfo): The transform information.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The provided :class:`~pyradise.data.subject.Subject` instance.
        """

        # potentially warn the user that the operation is not invertible
        if self.warn_on_non_invertible and not self.is_invertible():
            warnings.warn(
                "WARNING: "
                f"The {self.__class__.__name__} is called to invert its operation for the following image: \n"
                f"\t{target_image.__str__()} \nHowever, the filter is not invertible. The provided subject "
                "is returned without modification."
            )

        return subject


class AlphabeticOrganSortingFilterParams(FilterParams):
    """A filter parameter class for the :class:`~pyradise.process.postprocess.AlphabeticOrganSortingFilter` class.

    Args:
        ascending (bool): If the organs should be sorted in ascending order or not (default: True).
    """

    def __init__(self, ascending: bool = True) -> None:
        self.ascending = ascending


class AlphabeticOrganSortingFilter(Filter):
    """A filter class performing an alphabetic sorting of the :class:`~pyradise.data.image.SegmentationImage` instances
    according to their assigned :class:`~pyradise.data.organ.Organ` names.

    Note:
        This filter is helpful when ordering of the output matters such as for example if constructing a DICOM-RTSS
        :class:`~pydicom.dataset.Dataset` instance.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Return whether the filter is invertible or not.

        Returns:
            bool: False because the :class:`~pyradise.process.postprocess.AlphabeticOrganSortingFilter` is
            not invertible.
        """
        return False

    def execute(self, subject: Subject, params: AlphabeticOrganSortingFilterParams) -> Subject:
        """Execute the alphabetical sorting of the :class:`~pyradise.data.image.SegmentationImage` instances according
        to their associated :class:`~pyradise.data.organ.Organ` instances.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be sorted.
            params (AlphabeticOrganSortingFilterParams): The filter parameters

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with the alphabetically sorted
            :class:`~pyradise.data.image.SegmentationImage` instances.
        """
        subject.segmentation_images = sorted(subject.segmentation_images, key=lambda x: x.get_organ(as_str=True))

        if not params.ascending:
            subject.segmentation_images = subject.segmentation_images[::-1]

        return subject

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Return the provided image without any processing because the alphabetical sorting procedure is not
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

        # potentially warn the user that the operation is not invertible
        if self.warn_on_non_invertible and not self.is_invertible():
            warnings.warn(
                "WARNING: "
                f"The {self.__class__.__name__} is called to invert its operation for the following image: \n"
                f"\t{target_image.__str__()} \nHowever, the filter is not invertible. The provided subject "
                "is returned without modification."
            )

        return subject
