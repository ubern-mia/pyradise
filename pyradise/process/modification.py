import warnings
from copy import deepcopy
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk

from pyradise.data import (Annotator, IntensityImage, Modality, Organ,
                           SegmentationImage, Subject, TransformInfo,
                           seq_to_annotators, seq_to_modalities, seq_to_organs,
                           str_to_annotator, str_to_organ)

from .base import Filter, FilterParams
from .orientation import SpatialOrientation

__all__ = [
    "AddImageFilterParams",
    "AddImageFilter",
    "RemoveImageByOrganFilterParams",
    "RemoveImageByOrganFilter",
    "RemoveImageByAnnotatorFilterParams",
    "RemoveImageByAnnotatorFilter",
    "RemoveImageByModalityFilterParams",
    "RemoveImageByModalityFilter",
    "MergeSegmentationFilterParams",
    "MergeSegmentationFilter",
]


class AddImageFilterParams(FilterParams):
    """A filter parameter class for the :class:`~pyradise.process.modification.AddImageFilter` class.

    Args:
        images (Union[IntensityImage, SegmentationImage, Tuple[Union[IntensityImage, SegmentationImage], ...]): The
         :class:`~pyradise.data.image.Image` instances to add to the provided :class:`~pyradise.data.subject.Subject`
         instance.
    """

    def __init__(
        self, images: Union[IntensityImage, SegmentationImage, Tuple[Union[IntensityImage, SegmentationImage], ...]]
    ) -> None:
        if isinstance(images, tuple):
            self.images: Tuple[Union[IntensityImage, SegmentationImage], ...] = images
        else:
            self.images: Tuple[Union[IntensityImage, SegmentationImage], ...] = (images,)


class AddImageFilter(Filter):
    """A filter class to add :class:`~pyradise.data.image.Image` instances to the provided
    :class:`~pyradise.data.subject.Subject` instance.

    Note:
        This filter currently does not support the inverse operation (the removal of the added images). This feature
        will be added in the future.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Return whether the filter is invertible or not.

        Note:
        This filter currently does not support the inverse operation (the removal of the added images). This feature
        will be added in the future.

        Returns:
            bool: False because the addition of :class:`~pyradise.data.image.Image` instances is currently not
            supported.
        """
        return False

    def execute(self, subject: Subject, params: AddImageFilterParams) -> Subject:
        """Execute the addition procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to add the appropriate
             :class:`~pyradise.data.image.Image` instances to.
            params (AddImageFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance including the added
            :class:`~pyradise.data.image.Image` instances.
        """
        for image in params.images:
            # add the image to the subject
            subject.add_image(image)

            # track the necessary information
            image_sitk = image.get_image_data()
            self.tracking_data["organ"] = deepcopy(image.get_organ())
            self.tracking_data["annotator"] = deepcopy(image.get_annotator())
            self._register_tracked_data(image, image_sitk, image_sitk, params)

        return subject

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Return the provided subject without any processing because the inverse addition procedure (the removal)
        is currently not supported.

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


class RemoveImageByOrganFilterParams(FilterParams):
    """A filter parameter class for the :class:`~pyradise.process.modification.RemoveImageByOrganFilter` class.

    Args:
        organs (Sequence[Union[Organ, str]]): The organs to remove from the provided
         :class:`~pyradise.data.subject.Subject` instance.
    """

    def __init__(self, organs: Sequence[Union[Organ, str]]) -> None:
        self.organs: Tuple[Organ, ...] = seq_to_organs(organs)


class RemoveImageByOrganFilter(Filter):
    """A filter class to remove :class:`~pyradise.data.image.SegmentationImage` instances from the provided
    :class:`~pyradise.data.subject.Subject` instance. The :class:`~pyradise.data.image.SegmentationImage` instances
    are identified by their :class:`~pyradise.data.organ.Organ` instance.

    Note:
        If multiple :class:`~pyradise.data.image.SegmentationImage` instances exist with the same
        :class:`~pyradise.data.organ.Organ` instance all of them will be removed.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Return whether the filter is invertible or not.

        Returns:
            bool: False because the removal of :class:`~pyradise.data.image.SegmentationImage` instances is not
            invertible.
        """
        return False

    def execute(self, subject: Subject, params: RemoveImageByOrganFilterParams) -> Subject:
        """Execute the removal procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to remove the appropriate
             :class:`~pyradise.data.image.SegmentationImage` instances from.
            params (RemoveImageByOrganFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance excluding the removed
            :class:`~pyradise.data.image.SegmentationImage` instances.
        """
        for organ in params.organs:
            subject.remove_image_by_organ(organ)

            # track the necessary information
            # --> do not track the removal of entities
        return subject

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Return the provided subject without any processing because the removal procedure is not invertible.

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


class RemoveImageByAnnotatorFilterParams(FilterParams):
    """A filter parameter class for the :class:`~pyradise.process.modification.RemoveImageByAnnotatorFilter` class.

    Args:
        annotators (Sequence[Union[Annotator, str]]): The annotators identifying the
         :class:`~pyradise.data.image.SegmentationImage` instances to remove from the provided
         :class:`~pyradise.data.subject.Subject` instance.
    """

    def __init__(self, annotators: Sequence[Union[Annotator, str]]) -> None:
        self.annotators: Tuple[Annotator, ...] = seq_to_annotators(annotators)


class RemoveImageByAnnotatorFilter(Filter):
    """A filter class to remove :class:`~pyradise.data.image.SegmentationImage` instances from the provided
    :class:`~pyradise.data.subject.Subject` instance. The :class:`~pyradise.data.image.SegmentationImage` instances
    are identified by their :class:`~pyradise.data.annotator.Annotator` instance.

    Note:
        If multiple :class:`~pyradise.data.image.SegmentationImage` instances exist with the same
        :class:`~pyradise.data.annotator.Annotator` instance all of them will be removed.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Return whether the filter is invertible or not.

        Returns:
            bool: False because the removal of :class:`~pyradise.data.image.SegmentationImage` instances is not
            invertible.
        """
        return False

    def execute(self, subject: Subject, params: RemoveImageByAnnotatorFilterParams) -> Subject:
        """Execute the removal procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to remove the appropriate
             :class:`~pyradise.data.image.SegmentationImage` instances from.
            params (RemoveImageByAnnotatorFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance excluding the removed
            :class:`~pyradise.data.image.SegmentationImage` instances.
        """
        for annotator in params.annotators:
            subject.remove_image_by_annotator(annotator)

            # track the necessary information
            # --> do not track the removal of entities
        return subject

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Return the provided subject without any processing because the removal procedure is not invertible.

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


class RemoveImageByModalityFilterParams(FilterParams):
    """A filter parameter class for the :class:`~pyradise.process.modification.RemoveImageByModalityFilter` class.

    Args:
        modalities (Sequence[Union[Modality, str]]): The modalities identifying the
         :class:`~pyradise.data.image.IntensityImage` instances to remove from the provided
         :class:`~pyradise.data.subject.Subject` instance.
    """

    def __init__(self, modalities: Sequence[Union[Modality, str]]) -> None:
        self.modalities: Tuple[Modality, ...] = seq_to_modalities(modalities)


class RemoveImageByModalityFilter(Filter):
    """A filter class to remove :class:`~pyradise.data.image.IntensityImage` instances from the provided
    :class:`~pyradise.data.subject.Subject` instance. The :class:`~pyradise.data.image.IntensityImage` instances
    are identified by their :class:`~pyradise.data.modality.Modality` instance.

    Note:
        If multiple :class:`~pyradise.data.image.SegmentationImage` instances exist with the same
        :class:`~pyradise.data.modality.Modality` instance all of them will be removed.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Return whether the filter is invertible or not.

        Returns:
            bool: False because the removal of :class:`~pyradise.data.image.IntensityImage` instances is not
            invertible.
        """
        return False

    def execute(self, subject: Subject, params: RemoveImageByModalityFilterParams) -> Subject:
        """Execute the removal procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to remove the appropriate
             :class:`~pyradise.data.image.IntensityImage` instances from.
            params (RemoveImageByModalityFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance excluding the removed
            :class:`~pyradise.data.image.IntensityImage` instances.
        """
        for modality in params.modalities:
            subject.remove_image_by_modality(modality)

            # track the necessary information
            # --> do not track the removal of entities
        return subject

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Return the provided subject without any processing because the removal procedure is not invertible.

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


class MergeSegmentationFilterParams(FilterParams):
    """A filter parameter class for the :class:`~pyradise.process.modification.MergeSegmentationFilter` class.

    Note:
        The order of the provided organs determines the merging order. The first organ will be inserted into the
        resulting segmentation first, the second one second, etc. Therefore, if the segmentations overlap the
        segmentation of the first organ will be overwritten by the segmentation of the second organ, etc.

    Args:
        organs (Sequence[Union[Organ, str]]): The :class:`~pyradise.data.organ.Organ` instances to merge.
        output_organ_indexes (Sequence[int]): The indexes of the organs at the output (must be of equal length as
         organs). If ```None`` is provided the organs will be enumerated from 1 to n (default: None).
        output_organ (Union[Organ, str]): The :class:`~pyradise.data.organ.Organ` instance of the resulting
         segmentation.
        output_annotator (Union[Annotator, str]): The :class:`~pyradise.data.annotator.Annotator` instance of the
         resulting segmentation.
        output_orientation (Union[SpatialOrientation, str]): The orientation of the output segmentation (default: LPS).
    """

    def __init__(
        self,
        organs: Sequence[Union[Organ, str]],
        output_organ_indexes: Optional[Sequence[int]],
        output_organ: Union[Organ, str],
        output_annotator: Union[Annotator, str],
        output_orientation: Union[SpatialOrientation, str] = SpatialOrientation.LPS,
    ) -> None:
        self.organs: Tuple[Organ, ...] = seq_to_organs(organs)

        if output_organ_indexes is None:
            self.organ_indexes: Tuple[int, ...] = tuple(range(1, len(self.organs) + 1))
        else:
            if len(output_organ_indexes) != len(self.organs):
                raise ValueError("The length of the provided organ indexes must be equal to the number of organs.")

            if len(set(output_organ_indexes)) != len(output_organ_indexes):
                raise ValueError("The provided organ indexes must be unique.")

            self.organ_indexes: Tuple[int, ...] = tuple(output_organ_indexes)

        if isinstance(output_orientation, str):
            try:
                self.output_orientation: SpatialOrientation = SpatialOrientation[output_orientation]
            except KeyError:
                raise ValueError(f"Invalid output orientation: {output_orientation}")
        else:
            self.output_orientation: SpatialOrientation = output_orientation

        self.output_organ: Organ = str_to_organ(output_organ)
        self.output_annotator: Annotator = str_to_annotator(output_annotator)


class MergeSegmentationFilter(Filter):
    """A filter class for merging multiple :class:`~pyradise.data.image.SegmentationImage` instances into
    a new :class:`~pyradise.data.image.SegmentationImage` instance assigned to the provided
    :class:`~pyradise.data.subject.Subject` instance.

    Note:
        If the provided :class:`~pyradise.data.image.SegmentationImage` instances are non-binary all non-zero label
        values will be set to the provided organ index of the corresponding organ. Thus, non-binary segmentations will
        be treated as binary ones with all non-zero values being considered as foreground.

        Note that the merging order is defined by the order of the provided organs. However, the resulting segmentation
        will contain the organ indexes associated with the provided organs.

        The separation of segmentations is technically feasible to some extent. However, in radiotherapy the
        separation of merged segmentations can have adverse effects because it may lead to corrupted segmentations if
        segmentations overlap. Therefore, this filter does not provide the inverse procedure.

    """

    @staticmethod
    def is_invertible() -> bool:
        """Returns whether the filter is invertible or not.

        Note:
            The separation of segmentations is technically feasible to some extent. However, in radiotherapy the
            separation of merged segmentations can have adverse effects because it may lead to corrupted segmentations
            if segmentations overlap. Therefore, this filter does not provide the inverse procedure.

        Returns:
            bool: False because the merging of segmentations is not fully invertible.
        """
        return False

    def _merge_segmentations(
        self,
        target_image: sitk.Image,
        images: Tuple[SegmentationImage, ...],
        images_sitk: Tuple[sitk.Image, ...],
        organs: Tuple[Organ, ...],
        params: MergeSegmentationFilterParams,
    ) -> sitk.Image:
        """Merges the given segmentations into one segmentation.

        Args:
            target_image (sitk.Image): An empty image for the merging.
            images (Tuple[SegmentationImage, ...]): The :class:`~pyradise.data.segmentation_image.SegmentationImage`
             instances associated with the ``images_sitk`` and the ``organs``.
            images_sitk (Tuple[sitk.Image, ...]): The SimpleITK segmentation images to merge.
            organs (Tuple[Organ, ...]): The :class:`~pyradise.data.organ.Organ` instances.
            params (MergeSegmentationFilterParams): The filter parameters.

        Returns:
            sitk.Image: The merged segmentation.
        """
        target_image_np = sitk.GetArrayFromImage(target_image)

        for image, image_sitk, organ, organ_idx in zip(images, images_sitk, organs, params.organ_indexes):
            # resample the image to the empty image
            resampled_image_sitk = sitk.Resample(
                image_sitk, target_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8
            )

            # merge the resampled image into the empty image
            resampled_image_np = sitk.GetArrayFromImage(resampled_image_sitk)
            target_image_np[resampled_image_np > 0] = int(organ_idx)

            # track the necessary information
            self._register_tracked_data(image, image_sitk, resampled_image_sitk, params)

        # restore the target SimpleITK image
        new_target_image = sitk.GetImageFromArray(target_image_np.astype(np.uint8))
        new_target_image.CopyInformation(target_image)

        return new_target_image

    @staticmethod
    def _get_empty_image(images: Tuple[sitk.Image, ...]) -> sitk.Image:
        """Returns an empty image with the same size and spacing as the provided images.

        Args:
            images (Tuple[sitk.Image, ...]): The images to get the size and spacing from.

        Returns:
            sitk.Image: The empty image.
        """
        # get the physical properties of the images
        origins = np.array([image.GetOrigin() for image in images])
        spacings = np.array([image.GetSpacing() for image in images])
        sizes = np.array([image.GetSize() for image in images])
        max_coords = origins + sizes * spacings

        # get the physical limits
        limits = np.stack((np.min(origins, axis=0), np.max(max_coords, axis=0)))
        min_physical_coord = np.min(limits, axis=0)
        max_physical_coord = np.max(limits, axis=0)

        # generate the empty numpy image
        min_spacing = np.min(spacings, axis=0)
        shape = np.ceil((max_physical_coord - min_physical_coord) / min_spacing).astype(int)
        shape_np = shape[2], shape[0], shape[1]
        empty_image_np = np.zeros(shape_np, dtype=np.uint8)

        # generate the empty sitk image
        empty_image_sitk = sitk.GetImageFromArray(empty_image_np)
        empty_image_sitk.SetOrigin(min_physical_coord)
        empty_image_sitk.SetSpacing(min_spacing)
        empty_image_sitk.SetDirection(images[0].GetDirection())

        return empty_image_sitk

    def execute(self, subject: Subject, params: MergeSegmentationFilterParams) -> Subject:
        """Execute the merging procedure.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance containing the segmentations to
             merge.
            params (MergeSegmentationFilterParams): The filter parameters.

        Returns:
            Subject: The subject with the merged :class:`~pyradise.data.image.SegmentationImage` instance added.
        """
        # adjust the images accordingly
        images = []
        images_sitk = []
        organs = []
        for image in subject.segmentation_images:
            if image.get_organ() not in params.organs:
                continue

            image_sitk = deepcopy(image.get_image_data())
            image_sitk = sitk.DICOMOrient(image_sitk, str(params.output_orientation.name))

            # make sure that the image is binary
            image_np = sitk.GetArrayFromImage(image_sitk)
            image_np[image_np > 0] = 1
            image_np[image_np < 1] = 0
            binary_image_sitk = sitk.GetImageFromArray(image_np)
            binary_image_sitk.CopyInformation(image_sitk)

            images.append(image)
            images_sitk.append(binary_image_sitk)
            organs.append(image.get_organ())

        # sort the images according to the provided organs
        sorted_organs = []
        sorted_images = []
        sorted_images_sitk = []
        for param_organ in params.organs:
            if param_organ not in organs:
                continue

            local_idx = organs.index(param_organ)
            sorted_organs.append(organs[local_idx])
            sorted_images.append(images[local_idx])
            sorted_images_sitk.append(images_sitk[local_idx])

        # get an empty image
        empty_image = self._get_empty_image(tuple(images_sitk))

        # merge the segmentations
        merged_image = self._merge_segmentations(
            empty_image, tuple(sorted_images), tuple(sorted_images_sitk), tuple(sorted_organs), params
        )

        # create the new segmentation image
        merged_segmentation = SegmentationImage(merged_image, params.output_organ, params.output_annotator)
        subject.add_image(merged_segmentation)

        return subject

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Return the provided subject without any processing because the merging procedure is not invertible.

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
