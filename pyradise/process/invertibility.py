import warnings
from copy import deepcopy
from typing import Optional, Tuple, Union

from pyradise.data import (IntensityImage, Modality, OrganAnnotatorCombination,
                           SegmentationImage, Subject, TransformInfo,
                           seq_to_modalities,
                           seq_to_organ_annotator_combinations)
from pyradise.process import Filter, FilterParams

__all__ = ["PlaybackTransformTapeFilterParams", "PlaybackTransformTapeFilter"]


class PlaybackTransformTapeFilterParams(FilterParams):
    """A filter parameter class for the :class:`~pyradise.process.invertibility.PlaybackTransformTapeFilter` class.

    Args:
        modalities (Optional[Tuple[Union[str, Modality], ...]]): A tuple of modalities for which the transform tape
         should be played back. If None, the transform tape will be played back for all modalities (default: None).
        organ_annotator_combinations (Optional[Tuple[Union[Tuple[str, str], OrganRaterCombination], ...]]): A tuple of
         organ-annotator combinations for which the transform tape should be played back. If None, the transform tape
         will be played back for all organ-annotator combinations (default: None).
    """

    def __init__(
        self,
        modalities: Optional[Tuple[Union[str, Modality], ...]] = None,
        organ_annotator_combinations: Optional[Tuple[Union[Tuple[str, str], OrganAnnotatorCombination], ...]] = None,
    ) -> None:
        super().__init__()

        if modalities is not None:
            self.modalities = seq_to_modalities(modalities)
        else:
            self.modalities = None

        if organ_annotator_combinations is not None:
            self.organ_annotator_combinations = seq_to_organ_annotator_combinations(organ_annotator_combinations)
        else:
            self.organ_annotator_combinations = organ_annotator_combinations


class PlaybackTransformTapeFilter(Filter):
    """A filter class for playing back the transform tape of specific or all :class:`~pyradise.data.image.Image`
    instances of the provided :class:`~pyradise.data.subject.Subject` instance.

    This filter is helpful for restoring the spatial properties of the loaded data such that the output data of the
    processing pipeline has identical spatial properties as the input data.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Return False because the filter is not invertible.

        Returns:
            bool: False.
        """
        return False

    def execute(self, subject: Subject, params: Optional[PlaybackTransformTapeFilterParams] = None) -> Subject:
        """Execute the filter on the provided :class:`~pyradise.data.subject.Subject` instance.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance.
            params (Optional[FilterParams]): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with played back images.
        """
        original_images = []
        changed_images = []

        images = subject.get_images()
        for image in images:
            # exclude images not matching the provided criteria
            if isinstance(image, IntensityImage) and params.modalities is not None:
                if not image.get_modality() in params.modalities:
                    continue

            if isinstance(image, SegmentationImage) and params.organ_annotator_combinations is not None:
                if not image.get_organ_annotator_combination() in params.organ_annotator_combinations:
                    continue

            # copy the original subject
            temp_subject = deepcopy(subject)
            transform_tape = image.get_transform_tape()
            transform_infos = transform_tape.get_recorded_elements(reverse=True)

            # play back the transform tape
            for transform_info in transform_infos:
                filter_ = transform_info.get_filter()

                # activate warnings if the filter is not invertible
                if self.warn_on_non_invertible:
                    filter_.set_warning_on_non_invertible(True)

                temp_subject = filter_.execute_inverse(temp_subject, transform_info, image)

            # collect the modified images
            changed_image_candidates = [img for img in temp_subject.get_images() if img == image]
            changed_images.append(changed_image_candidates[0])
            original_images.append(image)

        # replace the original images with the modified images and reset the transform tape
        for original_image, changed_image in zip(original_images, changed_images):
            original_image.set_image_data(changed_image.get_image_data())
            original_image.get_transform_tape().reset()

        return subject

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Return the provided :class:`~pyradise.data.subject.Subject` instance without any processing because
        :class:`~pyradise.data.taping.TransformTape` playback is not invertible.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance.
            transform_info (TransformInfo): The :class:`~pyradise.data.taping.TransformInfo` instance.
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
