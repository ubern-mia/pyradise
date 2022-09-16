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
    SegmentationImage)
from .base import (
    Filter,
    FilterParameters)


# pylint: disable = too-few-public-methods
class SingleConnectedComponentFilterParameters(FilterParameters):
    """A class representing the parameters of a SingleConnectedComponentFilter."""

    def __init__(self,
                 excluded_organs: Optional[Union[Organ, Tuple[Organ, ...]]] = None
                 ) -> None:
        """Constructs the filter parameters for a SingleConnectedComponentFilter.

        Args:
            excluded_organs (Optional[Union[Organ, Tuple[Organ, ...]]]): The organs to be excluded from the computation.
        """
        super().__init__()
        if isinstance(excluded_organs, Organ):
            self.excluded_organs = (excluded_organs,)
        elif excluded_organs is None:
            self.excluded_organs = tuple()
        else:
            self.excluded_organs = excluded_organs


class SingleConnectedComponentFilter(Filter):
    """A class for keeping only one connected component per label."""

    @staticmethod
    def _is_binary_image(image: Union[itk.Image, SegmentationImage]) -> bool:
        """Checks if an image is binary.

        Args:
            image (Union[itk.Image, SegmentationImage]): The image to analyze.

        Returns:
            bool: True if the image is binary otherwise False.
        """
        if isinstance(image, SegmentationImage):
            itk_image = image.get_image()
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
            itk_image = image.get_image()
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
            itk_image = image.get_image()
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
        casted_itk_image = Image.cast(cc_itk_image, itk.template(itk_image_type)[1][0])

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
                params: SingleConnectedComponentFilterParameters
                ) -> Subject:
        """Executes the single connected component keeping procedure.

        Args:
            subject (Subject): The subject to process.
            params (SingleConnectedComponentFilterParameters): The filters parameters.

        Returns:
            Subject: The processed subject.
        """
        for image in subject.segmentation_images:

            if image.get_organ() in params.excluded_organs:
                continue

            if self._is_binary_image(image):
                single_label_images = (image,)

                image_np = sitk.GetArrayFromImage(image.get_image(as_sitk=True))
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
                image.set_image(cc_image)

        return subject


class AlphabeticOrganSortingFilter(Filter):
    """A class sorting a subject's segmentations alphabetical by their organ name."""

    def execute(self,
                subject: Subject,
                params: Optional[FilterParameters] = None
                ) -> Subject:
        """Execute the alphabetical sorting of the segmentation image organs.

        Args:
            subject (Subject): The subject to apply the filter on.
            params (FilterParameters): Unused.

        Returns:
            Subject: The subject with the alphabetically sorted segmentation images.
        """
        subject.segmentation_images = sorted(subject.segmentation_images, key=lambda x: x.get_organ(as_str=True))
        return subject