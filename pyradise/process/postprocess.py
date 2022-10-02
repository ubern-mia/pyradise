from typing import (
    Tuple,
    Union,
    Optional)
from copy import deepcopy

import itk
import numpy as np
import SimpleITK as sitk

from pyradise.data import (
    Subject,
    Image,
    Organ,
    SegmentationImage, TransformInfo)
from .base import (
    Filter,
    FilterParams)

__all__ = ['SingleConnectedComponentFilterParams', 'SingleConnectedComponentFilter',
           'AlphabeticOrganSortingFilterParams', 'AlphabeticOrganSortingFilter']


# pylint: disable = too-few-public-methods
class SingleConnectedComponentFilterParams(FilterParams):
    """A filter parameter class for the :class:`~pyradise.process.postprocess.SingleConnectedComponentFilter` class.

    Args:
        excluded_organs (Optional[Union[Organ, Tuple[Organ, ...]]]): The organs to be excluded from the connected
         component filtering. If ``None`` all :class:`~pyradise.data.image.SegmentationImage` instances will be
         filtered.
    """

    def __init__(self,
                 excluded_organs: Optional[Union[Organ, Tuple[Organ, ...]]] = None
                 ) -> None:
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

    @staticmethod
    def _is_binary_image(image: Union[itk.Image, SegmentationImage]) -> bool:
        """Checks if an image is binary.

        Args:
            image (Union[itk.Image, SegmentationImage]): The image to analyze.

        Returns:
            bool: True if the image is binary otherwise False.
        """
        if isinstance(image, SegmentationImage):
            itk_image = image.get_image_data(as_sitk=False)
        else:
            itk_image = image

        np_image = itk.GetArrayFromImage(itk_image)
        label_indexes = np.unique(np_image)

        if len(label_indexes) > 2:
            return False
        return True

    @staticmethod
    def _get_single_label_images(image: Union[itk.Image, SegmentationImage]
                                 ) -> Tuple[Tuple[itk.Image, ...], Tuple[int, ...]]:
        """Splits a label image into multiple single/binary label images.

        Args:
            image (Union[itk.Image, SegmentationImage]): The image to separate into binary images.

        Returns:
            Tuple[Tuple[itk.Image, ...], Tuple[int, ...]]: Returns the binary images and the original label indexes.
        """
        if isinstance(image, SegmentationImage):
            itk_image = image.get_image_data(as_sitk=False)
        else:
            itk_image = image

        np_image = itk.GetArrayFromImage(itk_image)
        contained_labels = list(np.unique(np_image))
        contained_labels.remove(0)

        single_label_images = []
        single_label_labels = []

        for label_index in contained_labels:
            np_image_label = deepcopy(np_image)

            np_image_label[np_image_label != label_index] = 0
            itk_image_label = itk.GetImageFromArray(np_image_label)
            itk_image_label.CopyInformation(itk_image)

            single_label_images.append(itk_image_label)
            single_label_labels.append(label_index)

        return tuple(single_label_images), tuple(single_label_labels)

    @staticmethod
    def _combine_images(images: Tuple[itk.Image, ...]) -> itk.Image:
        """Combines multiple label images to one image.

        Args:
            images (Tuple[itk.Image, ...]): The images to combine.

        Returns:
            itk.Image: The combined image.
        """
        if len(images) == 1:
            return images[0]

        base_np_image = itk.GetArrayFromImage(images[0])

        for i in range(1, len(images)):
            np_image = itk.GetArrayFromImage(images[i])
            np.putmask(base_np_image, np_image != 0, np_image)

        combined_itk_image = itk.GetImageFromArray(base_np_image)
        combined_itk_image.CopyInformation(images[0])

        return combined_itk_image

    @staticmethod
    def _get_single_connected_component_image(image: Union[itk.Image, SegmentationImage],
                                              organ: Union[Organ, int]
                                              ) -> itk.Image:
        """Removes all connected components except for the largest.

        Args:
            image (Union[itk.Image, SegmentationImage]): The image to process.
            organ (Union[Organ, int]): The label index of the output image.

        Returns:
            itk.Image: The image with only one single connected component.
        """

        if isinstance(image, SegmentationImage):
            itk_image = image.get_image_data(as_sitk=False)
            itk_image_type = image.get_image_itk_type()
        else:
            itk_image = image
            itk_image_type = itk.template(image)[1]

        cc_image_type = itk.Image[itk.UL, itk_image.GetImageDimension()]

        cc_filter = itk.ConnectedComponentImageFilter[itk_image_type, cc_image_type].New()
        cc_filter.SetInput(itk_image)
        cc_filter.Update()

        if cc_filter.GetObjectCount() == 1:
            return itk_image

        cc_itk_image = cc_filter.GetOutput()
        casted_itk_image = Image.cast(cc_itk_image, itk.template(itk_image_type)[1][0], as_sitk=False)

        ko_filter = itk.LabelShapeKeepNObjectsImageFilter[itk_image_type].New()
        ko_filter.SetInput(casted_itk_image)
        ko_filter.SetBackgroundValue(0)
        ko_filter.SetNumberOfObjects(1)
        ko_filter.Update()
        single_cc_itk_image = ko_filter.GetOutput()

        if isinstance(organ, Organ):
            organ_id = organ.index
        else:
            organ_id = organ

        single_cc_np_image = itk.GetArrayFromImage(single_cc_itk_image)
        single_cc_np_image[single_cc_np_image != 0] = organ_id

        correct_label_itk_image = itk.GetImageFromArray(single_cc_np_image)
        correct_label_itk_image.CopyInformation(single_cc_itk_image)

        return correct_label_itk_image

    def execute(self,
                subject: Subject,
                params: SingleConnectedComponentFilterParams
                ) -> Subject:
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
            if self._is_binary_image(image):
                single_label_images = (image,)

                image_np = sitk.GetArrayFromImage(image_sitk)
                unique_labels = list(np.unique(image_np))

                if 0 in unique_labels:
                    unique_labels.remove(0)

                if not unique_labels:
                    labels = tuple()
                else:
                    labels = (unique_labels[0],)

            else:
                single_label_images, labels = self._get_single_label_images(image)

            cc_images = []
            for single_label_image, label in zip(single_label_images, labels):
                single_cc_image = self._get_single_connected_component_image(single_label_image, label)
                cc_images.append(single_cc_image)

            if cc_images:
                cc_image = self._combine_images(tuple(cc_images))

                image.set_image_data(cc_image)

                self._register_tracked_data(image, image_sitk, image.get_image_data(), params)

        return subject

    def execute_inverse(self,
                        subject: Subject,
                        transform_info: TransformInfo
                        ) -> Subject:
        """Return the provided :class:`~pyradise.data.subject.Subject` instance without any processing because
        the single connected component filtering procedure is not invertible.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be returned.
            transform_info (TransformInfo): The transform information.

        Returns:
            Subject: The provided :class:`~pyradise.data.subject.Subject` instance.
        """
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

    def execute(self,
                subject: Subject,
                params: AlphabeticOrganSortingFilterParams
                ) -> Subject:
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

    def execute_inverse(self,
                        subject: Subject,
                        transform_info: TransformInfo
                        ) -> Subject:
        """Return the provided image without any processing because the alphabetical sorting procedure is not
        invertible.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be returned.
            transform_info (TransformInfo): The transform information.

        Returns:
            Subject: The provided :class:`~pyradise.data.subject.Subject` instance.
        """
        return subject
