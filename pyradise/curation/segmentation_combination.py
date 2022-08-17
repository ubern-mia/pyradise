from typing import (
    Tuple,
    Union,
    List,
    Dict,
    Optional)
from copy import deepcopy

import numpy as np
import SimpleITK as sitk

from pyradise.data import (
    Subject,
    Rater,
    Organ,
    SegmentationImage)
from .base import (
    Filter,
    FilterParameters)


# pylint: disable = too-few-public-methods
class SegmentationCombinationFilterParameters(FilterParameters):
    """A class representing the parameters for a SegmentationCombinationFilter."""

    def __init__(self,
                 organs: Union[Tuple[str, ...], Tuple[Organ, ...]],
                 output_organ: Union[str, Organ],
                 output_rater: Union[str, Rater],
                 exclude_terms: Tuple[str, ...] = (),
                 remove_organs: bool = True,
                 matching_method: str = 'exact',
                 check_for_overlap: bool = True
                 ) -> None:
        """

        Args:
            organs (Union[Tuple[str, ...], Tuple[Organ, ...]]): The organs to combine.
            output_organ (Union[str, Organ]): The organ of the combined segmentation image.
            output_rater (Union[str, Rater]): The rater of the combined segmentation image.
            exclude_terms (Tuple[str, ...]): Exclude terms in the organ name.
            remove_organs (bool): Indicates if the segmentation images to combine should be removed after combination
             (default: True).
            matching_method (str): The matching method for the organ (options: 'exact', 'in_case_sensitive', 'in')
             (default: 'exact').
            check_for_overlap (bool): Indicates if a check on overlap should be performed.
        """
        super().__init__()
        if isinstance(organs[0], str):
            self.organs = tuple(Organ(name) for name in organs)
        else:
            self.organs = organs

        if isinstance(output_organ, str):
            self.output_organ = Organ(output_organ)
        else:
            self.output_organ = output_organ

        if isinstance(output_rater, str):
            self.output_rater = Rater(output_rater)
        else:
            self.output_rater = output_rater

        self.exclude_terms = exclude_terms
        self.remove_organs = remove_organs
        self.matching_method = matching_method
        self.check_for_overlap = check_for_overlap


class SegmentationCombinationFilter(Filter):
    """A class which combines multiple organs into one segmentation image."""

    @staticmethod
    def _is_exact_match(organ: Organ,
                        params: SegmentationCombinationFilterParameters
                        ) -> bool:
        """Checks if the provided organ is an exact match with one of the organs in the parameters.

        Args:
            organ (Organ): The organ to check on match.
            params (SegmentationCombinationFilterParameters): The filter parameters containing the organs to combine.

        Returns:
            bool: True if the organ is an exact match otherwise False.
        """
        return organ in params.organs

    @staticmethod
    def _is_in_match(organ: Organ,
                     params: SegmentationCombinationFilterParameters,
                     case_sensitive: bool = False
                     ) -> bool:
        """Checks if the provided organ is an 'in' match with one of the organs in the parameters.

        Args:
            organ (Organ): The organ to check on match.
            params (SegmentationCombinationFilterParameters): The filter parameters containing the organs to combine.
            case_sensitive (bool): Indicates if the matching is case-sensitive or not.

        Returns:
            bool: True if the organ is a match otherwise False.
        """
        organ_name = organ.name if case_sensitive else organ.name.lower()

        for param_organ in params.organs:
            param_organ_name = param_organ.name if case_sensitive else param_organ.name.lower()

            if param_organ_name in organ_name:
                return True

        return False

    @staticmethod
    def _is_valid(segmentations: List[sitk.Image]) -> None:
        """Checks if the image properties are valid / are identical.

        Args:
            segmentations (List[sitk.Image]): The segmentation images to check on the properties.

        Returns:
            None
        """
        directions = [segmentation.GetDirection() for segmentation in segmentations]
        directions_criterion = [directions[0] == direction for direction in directions]
        if not all(directions_criterion):
            raise ValueError('The directions of all selected segmentation masks must be equal!')

        origins = [segmentation.GetOrigin() for segmentation in segmentations]
        origins_criterion = [origins[0] == origin for origin in origins]
        if not all(origins_criterion):
            raise ValueError('The origins of all selected segmentation masks must be equal!')

        sizes = [segmentation.GetSize() for segmentation in segmentations]
        sizes_criterion = [sizes[0] == size for size in sizes]
        if not all(sizes_criterion):
            raise ValueError('The sizes of all selected segmentation masks must be equal!')

    @staticmethod
    def _contains_overlap(segmentations: List[np.ndarray]) -> None:
        """Checks if there is an overlap between at least two segmentation images.

        Args:
            segmentations (List[np.ndarray]): The segmentation images to check on overlap.

        Returns:
            None
        """
        contains_overlap = []

        combinations = [(a, b) for idx, a in enumerate(segmentations) for b in segmentations[idx + 1:]]

        for (segmentation_1, segmentation_2) in combinations:
            indicator = np.logical_and(segmentation_1.astype(np.bool), segmentation_2.astype(np.bool))
            contains_overlap.append(np.sum(indicator) != 0)

        if any(contains_overlap):
            raise ValueError('There is overlap between at least two selected segmentation masks!')

    @staticmethod
    def _combine_segmentations(segmentations: List[SegmentationImage],
                               params: SegmentationCombinationFilterParameters
                               ) -> SegmentationImage:
        """Combines multiple segmentation images into one.

        Args:
            segmentations (List[SegmentationImage]): The segmentation images to combine.
            params (SegmentationCombinationFilterParameters): The filters parameters.

        Returns:
            SegmentationImage: The combined segmentation image.
        """
        segmentations_sitk = [segmentation.get_image(as_sitk=True) for segmentation in segmentations]

        SegmentationCombinationFilter._is_valid(segmentations_sitk)

        segmentations_np = [sitk.GetArrayFromImage(segmentation) for segmentation in segmentations_sitk]

        if params.check_for_overlap:
            SegmentationCombinationFilter._contains_overlap(segmentations_np)

        mask = np.zeros_like(segmentations_np[0]).astype(np.bool)

        for segmentation_np in segmentations_np:
            np.putmask(mask, segmentation_np != 0, 1)

        new_image = sitk.GetImageFromArray(mask.astype(np.uint8))
        new_image.CopyInformation(segmentations_sitk[0])

        new_segmentation = SegmentationImage(new_image, params.output_organ, params.output_rater)
        new_segmentation.transform_tape = deepcopy(segmentations[0].get_transform_tape())

        return new_segmentation

    @staticmethod
    def _contains_exclude_terms(organ: Organ,
                                params: SegmentationCombinationFilterParameters
                                ) -> bool:
        """Checks if the organ name contains exclude names (case-insensitive).

        Args:
            organ (Organ): The organ which name need to be checked.
            params (SegmentationCombinationFilterParameters): The filter parameters containing the exclusion terms.

        Returns:
            bool: True if an exclusion term is contained.
        """
        if not params.exclude_terms:
            return False

        is_contained = False
        for term in params.exclude_terms:
            if term.lower() in organ.name.lower():
                is_contained = True

        return is_contained

    def execute(self,
                subject: Subject,
                params: SegmentationCombinationFilterParameters
                ) -> Subject:
        """Executes the label combination procedure.

        Args:
            subject (Subject): The subject to be processed.
            params (SegmentationCombinationFilterParameters): The filters parameters.

        Returns:
            Subject: The processed subject.
        """
        if params is None:
            raise ValueError('No parameters defined for filter!')

        selected_segmentations = []

        for segmentation in subject.segmentation_images:

            if self._contains_exclude_terms(segmentation.get_organ(), params):
                continue

            if params.matching_method == 'exact':
                matching_result = self._is_exact_match(segmentation.get_organ(), params)

            elif params.matching_method == 'in_case_sensitive':
                matching_result = self._is_in_match(segmentation.get_organ(), params, case_sensitive=True)

            elif params.matching_method == 'in':
                matching_result = self._is_in_match(segmentation.get_organ(), params, case_sensitive=False)

            else:
                raise ValueError('Invalid matching method!')

            if matching_result:
                selected_segmentations.append(segmentation)

        if not selected_segmentations:
            return subject

        new_segmentation = self._combine_segmentations(selected_segmentations, params)

        if params.remove_organs:
            for selected_segmentation in selected_segmentations:
                subject.remove_image(selected_segmentation)

        subject.add_image(new_segmentation, force=True)

        return subject


# pylint: disable = too-few-public-methods
class CombineEnumeratedLabelFilterParameters(FilterParameters):
    """A class representing the parameters for a CombineEnumeratedLabelFilter."""

    def __init__(self,
                 organs: Dict[int, Organ],
                 combination_order: Optional[Tuple[int, ...]],
                 required_rater: Optional[Rater],
                 output_organ: Union[Organ, str],
                 output_rater: Optional[Union[Rater, str]] = None,
                 remove_combined: bool = False
                 ) -> None:
        """Constructs the parameters for a CombineEnumeratedLabelFilter.

        Args:
            organs (Dict[int, Organ]): A dict specifying the organs which should be combined and their output label
             index.
            combination_order (Optional[Tuple[int, ...]]): The order how the organs get combined.
            required_rater (Optional[Rater]): If given adds a criterion for a matching rater.
            output_organ (Union[Organ, str]): Specifies the output organ.
            output_rater (Optional[Union[Rater, str]]): Specifies the output rater.
            remove_combined (bool): Indicates if the combined segmentation masks should be removed from the subject.
        """

        super().__init__()

        self.organs = organs

        if combination_order is not None:
            assert len(organs.keys()) == len(combination_order), 'The number of organs to combine must be equal to ' \
                                                                 'the number of combination order entries!'

            self.combination_order = combination_order
        else:
            combination_indexes = list(self.organs.keys())
            combination_indexes.sort()
            self.combination_order = combination_indexes

        self.required_rater = required_rater

        if isinstance(output_organ, Organ):
            self.output_organ = output_organ
        else:
            self.output_organ = Organ(output_organ)

        if not output_rater:
            self.output_rater = None
        elif isinstance(output_rater, Rater):
            self.output_rater = output_rater
        else:
            self.output_rater = Rater(output_rater)

        self.remove_combined = remove_combined


class CombineEnumeratedLabelFilter(Filter):
    """A class which combines multiple segmentation images into one and enumerates the labels accordingly."""

    @staticmethod
    def is_match(segmentation: SegmentationImage,
                 params: CombineEnumeratedLabelFilterParameters
                 ) -> bool:
        """Checks if the provided segmentation fulfills the matching criterion to be selected for combination.

        Args:
            segmentation (SegmentationImage): The segmentation image to check for match.
            params (CombineEnumeratedLabelFilterParameters): The filter parameters.

        Returns:
            bool: True if the segmentation image is a match otherwise False.
        """
        result = False

        for _, organ in params.organs.items():
            if params.required_rater is None:
                criteria = (segmentation.get_organ() == organ,)
            else:
                criteria = (segmentation.get_organ() == organ,
                            segmentation.get_rater() == params.required_rater)

            if all(criteria):
                result = True

        return result

    @staticmethod
    def _get_index(segmentation: SegmentationImage,
                   params: CombineEnumeratedLabelFilterParameters
                   ) -> int:
        """Gets the output index of the segmentation.

        Args:
            segmentation (SegmentationImage): The segmentation image to get the index for.
            params (CombineEnumeratedLabelFilterParameters): The filter parameter.

        Returns:
            int: The output index of the segmentation in the combined segmentation mask.
        """
        for idx, organ in params.organs.items():
            if organ == segmentation.get_organ():
                return idx

        raise ValueError('The provided segmentation is invalid!')

    @staticmethod
    def _validate_segmentations(segmentations: Tuple[Tuple[SegmentationImage, int]],
                                subject: Subject
                                ) -> None:
        """Validates the segmentations by checking if their physical properties are equal.

        Args:
            segmentations (Tuple[Tuple[SegmentationImage, int]]): The segmentations and their indexes.
            subject (Subject): The subject to be used when an exception is raised.

        Returns:
            None
        """
        reference_image = segmentations[0][0].get_image(as_sitk=True)
        reference_origin = reference_image.GetOrigin()
        reference_direction = reference_image.GetDirection()
        reference_size = reference_image.GetSize()

        for segmentation, _ in segmentations:
            image = segmentation.get_image(as_sitk=True)

            criteria = (reference_origin == image.GetOrigin(),
                        reference_direction == image.GetDirection(),
                        reference_size == image.GetSize())

            if not all(criteria):
                raise Exception(f'The segmentation image {segmentation.get_organ(as_str=True)} of subject '
                                f'{subject.name} is invalid for combination!')

    @staticmethod
    def _combine_labels(segmentations: Tuple[Tuple[SegmentationImage, int]],
                        params: CombineEnumeratedLabelFilterParameters
                        ) -> SegmentationImage:
        """Combines the segmentations into one common segmentation which is enumerated according to the provided
         parameters.

        Args:
            segmentations (Tuple[Tuple[SegmentationImage, int]]): The segmentations to combine and their indexes.
            params (CombineEnumeratedLabelFilterParameters): The filter parameters.

        Returns:
            SegmentationImage: The combined segmentation image.
        """
        combined_mask = np.zeros_like(sitk.GetArrayFromImage(segmentations[0][0].get_image(as_sitk=True)))

        for segmentation, label_idx in segmentations:
            mask = sitk.GetArrayFromImage(segmentation.get_image(as_sitk=True))

            mask[mask != 0] = label_idx

            np.putmask(combined_mask, mask != 0, mask)

        image = sitk.GetImageFromArray(combined_mask)
        image.CopyInformation(segmentations[0][0].get_image(as_sitk=True))

        segmentation_image = SegmentationImage(image, params.output_organ, params.output_rater)

        return segmentation_image

    def execute(self,
                subject: Subject,
                params: CombineEnumeratedLabelFilterParameters
                ) -> Subject:
        """Executes the combination procedure.

        Args:
            subject (Subject): The subject from which to combine the segmentations.
            params (CombineEnumeratedLabelFilterParameters): The filter parameters.

        Returns:
            Subject: The processed subject.
        """
        to_process = []

        for segmentation in subject.segmentation_images:
            if self.is_match(segmentation, params):
                idx = self._get_index(segmentation, params)
                to_process.append((segmentation, idx))

        if not to_process:
            return subject

        self._validate_segmentations(tuple(to_process), subject)

        sorted_to_process = []
        for idx in params.combination_order:
            organ = params.organs.get(idx)
            for segmentation in to_process:
                if segmentation[0].get_organ() == organ:
                    sorted_to_process.append(segmentation)

        combined = self._combine_labels(tuple(sorted_to_process), params)

        if params.remove_combined:
            for processed in sorted_to_process:
                subject.remove_image(processed[0])

        subject.add_image(combined)

        return subject


class CombineSegmentationsFilterParameters(FilterParameters):
    """A class representing the parameters for a CombineSegmentationsFilter."""

    def __init__(self,
                 organs_to_combine: Union[Tuple[Organ, ...], Tuple[str, ...]],
                 new_organ: Organ,
                 new_rater: Optional[Rater] = None,
                 allow_override: bool = False
                 ) -> None:
        """Constructs the parameters for a CombineSegmentationsFilter.

        Args:
            organs_to_combine (Union[Tuple[Organ, ...], Tuple[str, ...]]): The organs identifying the images to be
             combined.
            new_organ (Organ): The organ of the combined segmentation image.
            new_rater (Optional[Rater]): The rater for the combined segmentation image.
            allow_override (bool): If true allows the overriding of a possible existing image with the newly combined,
             otherwise not (Default: False).
        """
        super().__init__()

        assert len(organs_to_combine) > 0, 'No organs selected for combination!'
        if isinstance(organs_to_combine[0], Organ):
            self.organs_to_combine = organs_to_combine
        elif isinstance(organs_to_combine[0], str):
            self.organs_to_combine = tuple(Organ(name) for name in organs_to_combine)
        else:
            raise TypeError('The provided organ to combine information is of an invalid type!')

        self.new_organ = new_organ
        self.new_rater = new_rater

        self.allow_override = allow_override


class CombineSegmentationsFilter(Filter):
    """A class which combines multiple segmentations into one new segmentation."""

    @staticmethod
    def _get_matching_images(subject: Subject,
                             params: CombineSegmentationsFilterParameters
                             ) -> Tuple[SegmentationImage]:
        """Get the matching images according to the specified organs.

        Args:
            subject (Subject): The subject holding the segmentation images.
            params (CombineSegmentationsFilterParameters): The filters parameters.

        Returns:
            Tuple[SegmentationImage]: The segmentation images for combination.
        """
        to_process = []

        for image in subject.segmentation_images:
            if image.get_organ() in params.organs_to_combine:
                to_process.append(image)

        return tuple(to_process)

    @staticmethod
    def _validate_images(images: Tuple[SegmentationImage],
                         subject: Subject
                         ) -> None:
        """Validates if a list of images posses the same properties.

        Args:
            images (Tuple[SegmentationImage]): The images to validate.
            subject (Subject): The subject to which the images belong.

        Returns:
            None
        """
        reference_image = images[0].get_image(as_sitk=True)
        size = reference_image.GetSize()
        spacing = reference_image.GetSpacing()
        direction = reference_image.GetDirection()
        origin = reference_image.GetOrigin()

        for image in images[1:]:
            image_sitk = image.get_image(as_sitk=True)
            result = all((size == image_sitk.GetSize(),
                          spacing == image_sitk.GetSpacing(),
                          direction == image_sitk.GetDirection(),
                          origin == image_sitk.GetOrigin()))

            if not result:
                raise Exception(f'The segmentation image with organ {image.get_organ(as_str=True)} of '
                                f'subject {subject.name} is invalid for combination!')

    @staticmethod
    def _combine_images(images: Tuple[SegmentationImage],
                        params: CombineSegmentationsFilterParameters
                        ) -> SegmentationImage:
        """Combines a list of segmentation images to a new segmentation image.

        Args:
            images (Tuple[SegmentationImage]): The segmentation images to combine.
            params (CombineSegmentationsFilterParameters): The parameters specifying the new organ and rater.

        Returns:
            SegmentationImage: The combined segmentation image.
        """
        new_image_np = sitk.GetArrayFromImage(images[0].get_image(as_sitk=True))
        new_image_np[new_image_np != 0] = 1

        for image in images[1:]:
            image_np = sitk.GetArrayFromImage(image.get_image(as_sitk=True))
            image_np[image_np != 0] = 1

            np.putmask(new_image_np, image_np, 1)

        new_image_sitk = sitk.GetImageFromArray(new_image_np)
        new_image_sitk.CopyInformation(images[0].get_image(as_sitk=True))

        new_image = SegmentationImage(new_image_sitk, params.new_organ, params.new_rater)
        return new_image

    def execute(self,
                subject: Subject,
                params: CombineSegmentationsFilterParameters
                ) -> Subject:
        """Executes the combination of multiple segmentation images as specified by the filters parameters.

        Args:
            subject (Subject): The subject to apply the filter on.
            params (CombineSegmentationsFilterParameters): The filter parameters.

        Returns:
            Subject: The processed subject.
        """
        images_to_process = self._get_matching_images(subject, params)

        # returns the subject if there are no masks to combine
        if not images_to_process:
            return subject

        # validate the images
        self._validate_images(images_to_process, subject)

        # combine the images
        new_image = self._combine_images(images_to_process, params)

        # add or override the new image to the subject
        if params.allow_override:
            same_organ_images = []
            for image in subject.segmentation_images:
                if image.get_organ() == new_image.get_organ():
                    same_organ_images.append(image)

            for same_organ_image in same_organ_images:
                subject.remove_image(same_organ_image)

        subject.add_image(new_image)

        return subject
