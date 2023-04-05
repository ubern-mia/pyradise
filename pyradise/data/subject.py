from collections import abc as col_abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn

from .annotator import Annotator
from .image import Image, IntensityImage, SegmentationImage
from .modality import Modality
from .organ import Organ

__all__ = ["Subject"]


class Subject:
    """The :class:`Subject` is the main data object which holds all :class:`~pyradise.data.image.IntensityImage` s and
    :class:`~pyradise.data.image.SegmentationImage` s associated with the subject. Furthermore, it can hold any type
    of additional data associated with the patient. Currently, the routines implemented in PyRaDiSe do not use this
    mechanism so it can be used freely by the user.

    Args:
        name (str): The name of the subject.
        images (Optional[Union[Image, Sequence[Image]]]): One or multiple images to add to the subject.
        data (Optional[Dict[str, Any]]): Additional data which is associated with the subject.

    Examples:
        The following example demonstrates the manual construction of a :class:`Subject`:

        >>> from argparse import ArgumentParser
        >>> from typing import Tuple
        >>> import os
        >>>
        >>> import SimpleITK as sitk
        >>>
        >>> from pyradise.data import (Subject, IntensityImage, SegmentationImage,
        >>>                            Modality, Organ, Annotator)
        >>> from pyradise.fileio import SubjectWriter, ImageFileFormat
        >>>
        >>>
        >>> def get_segmentation_file_paths(path: str,
        >>>                                 valid_organs: Tuple[Organ, ...]
        >>>                                 ) -> Tuple[str]:
        >>>     file_paths = []
        >>>
        >>>     for file in os.listdir(path):
        >>>         if not file.endswith('.nii.gz'):
        >>>             continue
        >>>
        >>>         if any(entry.name in file for entry in valid_organs):
        >>>             file_paths.append(os.path.join(path, file))
        >>>
        >>>     return tuple(sorted(file_paths))
        >>>
        >>>
        >>> def get_intensity_file_paths(path: str,
        >>>                              valid_modalities: Tuple[Modality, ...]
        >>>                              ) -> Tuple[str]:
        >>>     file_paths = []
        >>>
        >>>     for file in os.listdir(path):
        >>>         if not file.endswith('.nii.gz'):
        >>>             continue
        >>>
        >>>         if any(entry.get_name() in file for entry in valid_modalities):
        >>>             file_paths.append(os.path.join(path, file))
        >>>
        >>>     return tuple(sorted(file_paths))
        >>>
        >>>
        >>> def main(input_dir: str,
        >>>          output_dir: str
        >>>          ) -> None:
        >>>     # Retrieve image file paths
        >>>     organs = (Organ('Brainstem'), Organ('Eyes'),
        >>>               Organ('Hippocampi'), Organ('OpticNerves'))
        >>>     modalities = (Modality('CT'), Modality('T1c'),
        >>>                   Modality('T1w'), Modality('T2w'))
        >>>
        >>>     segmentation_file_paths = get_segmentation_file_paths(input_dir, organs)
        >>>     intensity_file_paths = get_intensity_file_paths(input_dir, modalities)
        >>>
        >>>     # Load the segmentation image files
        >>>     images = []
        >>>     for path, organ in zip(segmentation_file_paths, organs):
        >>>         image = SegmentationImage(sitk.ReadImage(path, sitk.sitkUInt8),
        >>>                                   organ, Annotator.get_default())
        >>>         images.append(image)
        >>>
        >>>     # Load the intensity image files
        >>>     for path, modality in zip(intensity_file_paths, modalities):
        >>>         image = IntensityImage(sitk.ReadImage(path, sitk.sitkFloat32),
        >>>                                modality)
        >>>         images.append(image)
        >>>
        >>>     # Construct the subject
        >>>     subject = Subject(os.path.basename(input_dir), images)
        >>>
        >>>     # Display the subject name and properties of the intensity and
        >>>     # segmentation images
        >>>     print(f'Subject {subject.get_name()} contains the following images:')
        >>>
        >>>     for image in subject.intensity_images:
        >>>         print(f'Intensity image of modality {image.get_modality(True)} '
        >>>               f'with size: {image.get_size()}')
        >>>
        >>>     for image in subject.segmentation_images:
        >>>         print(f'Segmentation image of {image.get_organ(True)} '
        >>>               f'with size: {image.get_size()}')
        >>>
        >>>     # Write the subject to disk
        >>>     SubjectWriter(ImageFileFormat.NRRD).write(output_dir, subject,
        >>>                                               write_transforms=False)
        >>>
        >>>
        >>> if __name__ == '__main__':
        >>>     parser = ArgumentParser()
        >>>     parser.add_argument('-input_dir', type=str)
        >>>     parser.add_argument('-output_dir', type=str)
        >>>     args = parser.parse_args()
        >>>
        >>>     main(args.input_dir, args.output_dir)
        >>>
        >>> # Output:
        >>> # Subject subject_1 contains the following images:
        >>> # Intensity image of modality CT with size: (256, 256, 256)
        >>> # Intensity image of modality T1c with size: (256, 256, 256)
        >>> # Intensity image of modality T1w with size: (256, 256, 256)
        >>> # Intensity image of modality T2w with size: (256, 256, 256)
        >>> # Segmentation image of Brainstem with size: (256, 256, 256)
        >>> # Segmentation image of Eyes with size: (256, 256, 256)
        >>> # Segmentation image of Hippocampi with size: (256, 256, 256)
        >>> # Segmentation image of OpticNerves with size: (256, 256, 256)
    """

    def __init__(
        self, name: str, images: Optional[Union[Image, Sequence[Image]]] = None, data: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()

        self.name = name

        self.intensity_images: List[IntensityImage] = []
        self.segmentation_images: List[SegmentationImage] = []

        if isinstance(images, IntensityImage):
            self.intensity_images.append(images)

        if isinstance(images, SegmentationImage):
            self.segmentation_images.append(images)

        if isinstance(images, col_abc.Sequence):
            for image in images:
                if isinstance(image, IntensityImage):
                    self.intensity_images.append(image)
                elif isinstance(image, SegmentationImage):
                    self.segmentation_images.append(image)
                else:
                    raise ValueError(
                        f"At least one image is not of type {IntensityImage.__class__.__name__} or"
                        f"{SegmentationImage.__class__.__name__}!"
                    )

        # check validity of the additional data
        if data is not None:
            if not isinstance(data, dict):
                raise TypeError(
                    "Additional data must be of type dict with the key providing an identifier for " "data retrieval."
                )
            if not all(isinstance(key, str) for key in data.keys()):
                raise TypeError(
                    "Additional data keys must be of type str because they are used as an identifier for"
                    "data retrieval.!"
                )
        else:
            data = {}

        self.data: Dict[str, Any] = data

    @staticmethod
    def _check_for_single_candidate(candidates: List[Any], entity_name: str, return_first_on_multiple: bool) -> Any:
        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            if return_first_on_multiple:
                warn(
                    f"The search for {entity_name} is ambiguous because there are multiple ({len(candidates)}) "
                    "candidates which are equal. The first found candidate will be returned."
                )
                return candidates[0]
            else:
                raise ValueError(f"There are multiple {entity_name} which fulfil the criterion ({len(candidates)})!")

        return None

    def get_name(self) -> str:
        """Get the name of the subject.

        Returns:
            str: The name of the subject.
        """
        return self.name

    def get_modalities(self) -> Tuple[Optional[Modality], ...]:
        """Get the modalities of the subject-associated intensity images.

        Returns:
            Tuple[Optional[Modality], ...]: The modalities of the intensity images.
        """
        modalities = [img.get_modality() for img in self.intensity_images]
        return tuple(modalities)

    def get_organs(self) -> Tuple[Optional[Organ], ...]:
        """Get the organs of the subject-associated segmentation images.

        Returns:
            Tuple[Optional[Organ], ...]: The organs of the segmentation images.
        """
        organs = [seg.get_organ() for seg in self.segmentation_images]
        return tuple(organs)

    def get_annotators(self) -> Tuple[Optional[Annotator], ...]:
        """Get the annotators of the subject-associated segmentation images.

        Returns:
            Tuple[Optional[Rater], ...]: The annotators of the segmentation images.
        """
        raters = [seg.get_annotator() for seg in self.segmentation_images]
        return tuple(raters)

    def add_image(self, image: Union[IntensityImage, SegmentationImage], force: bool = False) -> None:
        """Add an image to the subject.

        Args:
            image (Union[IntensityImage, SegmentationImage]): The image to add to the subject.
            force (bool): Indicates if addition should be performed even if a similar (same modality for intensity
             images or same organ for segmentation images) image is already contained (default: False).

        Raises:
            ValueError: If an intensity image with similar modality is already contained or a segmentation image with
             similar organ is already contained and ``force`` if False.

        Returns:
            None
        """
        images = self.get_images_by_type(type(image))

        for image_ in images:
            if image_ == image:
                if force:
                    warn(
                        f"An image of type {type(image).__name__} with the same properties is already contained in "
                        f"the subject. The image will be added anyway due to force=True."
                    )
                else:
                    raise ValueError(
                        f"An image of type {type(image).__name__} with the same properties is already "
                        "contained in the subject. No image will be added!"
                    )

        images.append(image)

    def add_images(self, images: Sequence[Union[IntensityImage, SegmentationImage]], force: bool = False) -> None:
        """Add multiple images to the subject.

        Args:
            images (Sequence[Union[IntensityImage, SegmentationImage]]): The images to add to the subject.
            force (bool): Indicates if addition should be performed even if a similar (same modality for intensity
             images or same organ for segmentation images) image is already contained (default: False).

        Returns:
            None
        """
        for image in images:
            self.add_image(image, force)

    def get_images(self) -> List[Union[IntensityImage, SegmentationImage]]:
        """Get all images of the subject.

        Returns:
            List[Union[IntensityImage, SegmentationImage]]: All images of the subject.
        """
        return [*self.intensity_images, *self.segmentation_images]

    def get_image_by_modality(
        self, modality: Union[Modality, str], return_first_on_multiple: bool = False
    ) -> Optional[IntensityImage]:
        """Get one intensity image by its modality.

        Args:
            modality (Union[Modality, str]): The modality of the image to retrieve.
            return_first_on_multiple (bool): Indicates if the first found image should be returned if there are
             multiple candidates, otherwise an error is raised on multiple candidates (default: False).

        Returns:
            Optional[IntensityImage]: The intensity image or None if there are multiple candidates.
        """
        if isinstance(modality, str):
            modality = Modality(modality)

        candidates = [img for img in self.intensity_images if img.get_modality() == modality]

        return self._check_for_single_candidate(candidates, "modalities", return_first_on_multiple)

    def get_image_by_organ(
        self, organ: Union[Organ, str], return_first_on_multiple: bool = False
    ) -> Optional[SegmentationImage]:
        """Get one segmentation image by its organ.

        Args:
            organ (Union[Organ, str]): The organ of the image to retrieve.
            return_first_on_multiple (bool): Indicates if the first found image should be returned if there are
             multiple candidates, otherwise an error is raised on multiple candidates (default: False).

        Returns:
            Optional[SegmentationImage]: The segmentation image or None if there are multiple candidates.
        """
        if isinstance(organ, str):
            organ = Organ(organ, None)

        candidates = [img for img in self.segmentation_images if img.get_organ() == organ]

        return self._check_for_single_candidate(candidates, "organs", return_first_on_multiple)

    def get_images_by_annotator(self, annotator: Union[Annotator, str]) -> Optional[Tuple[SegmentationImage]]:
        """Get one or multiple segmentation images by their annotator.

        Args:
            annotator (Union[Annotator, str]): The annotator of the image to retrieve.

        Returns:
            Optional[Union[SegmentationImage, Tuple[SegmentationImage]]]: The segmentation images or None if there is
            no image with this annotator.
        """
        if isinstance(annotator, str):
            annotator = Annotator(annotator)

        candidates: List[SegmentationImage] = [
            img for img in self.segmentation_images if img.get_annotator() == annotator
        ]

        if not candidates:
            return None

        return tuple(candidates)

    def get_image_by_organ_and_annotator(
        self, organ: Union[Organ, str], annotator: Union[Annotator, str], return_first_on_multiple: bool = False
    ) -> Optional[SegmentationImage]:
        """Get one segmentation image by its organ and annotator.

        Args:
            organ (Union[Organ, str]): The organ of the image to retrieve.
            annotator (Union[Annotator, str]): The annotator of the image to retrieve.
            return_first_on_multiple (bool): Indicates if the first found image should be returned if there are
             multiple candidates, otherwise an error is raised on multiple candidates (default: False).

        Returns:
            Optional[SegmentationImage]: The segmentation image or :data:`None` if there are multiple candidates.
        """
        if isinstance(organ, str):
            organ = Organ(organ, None)
        if isinstance(annotator, str):
            annotator = Annotator(annotator)

        candidates = [
            img for img in self.segmentation_images if img.get_organ() == organ and img.get_annotator() == annotator
        ]

        return self._check_for_single_candidate(candidates, "organs and annotators", return_first_on_multiple)

    def get_images_by_type(self, image_type: type) -> List[Image]:
        """Get all images of a specific type.

        Args:
            image_type: The type of the images to retrieve.

        Returns:
            List[Image]: A list of all images of the specified type.
        """
        if image_type == IntensityImage:
            return self.intensity_images
        elif image_type == SegmentationImage:
            return self.segmentation_images
        else:
            raise ValueError("The given data type is not supported or not contained in the subject!")

    def replace_image(
        self,
        new_image: Union[IntensityImage, SegmentationImage],
        old_image: Optional[Union[IntensityImage, SegmentationImage]] = None,
    ) -> bool:
        """Replace an image in the subject either specified by an old image or by the properties of the new image.

        The following properties are used to identify the image to be replaced:

        - :class:`~pyradise.data.image.IntensityImage`: The :class:`~pyradise.data.modality.Modality` of the image.
        - :class:`~pyradise.data.image.SegmentationImage`: The :class:`~pyradise.data.organ.Organ` and the
          :class:`~pyradise.data.annotator.Rater` of the image.

        Args:
            new_image (Union[IntensityImage, SegmentationImage]): The new image which will be inserted into the subject.
            old_image (Optional[Union[IntensityImage, SegmentationImage]]): The old image which will be replaced by the
             new image. If None, the new image properties are used to find an image to replace (default: None).

        Returns:
            bool: True if the image is replaced successfully, False otherwise.
        """

        def _get_equal_entities(reference: Any, candidates: Sequence[Any]) -> Tuple[Any]:
            candidates_ = [candidate for candidate in candidates if isinstance(candidate, type(reference))]

            if not candidates_:
                return tuple()

            return tuple(candidate for candidate in candidates_ if candidate == reference)

        image_sequence = self.get_images_by_type(type(new_image))

        if old_image is None:
            equal_images = _get_equal_entities(new_image, image_sequence)

            if not equal_images:
                return False

            if len(equal_images) > 1:
                warn(
                    f"More than one image of type {type(new_image).__name__} with the same properties is present "
                    "in the subject. Exclusively the first image found will be replaced!"
                )

            old_image_idx = image_sequence.index(equal_images[0])
            image_sequence[old_image_idx] = new_image
            return True

        else:
            if not isinstance(old_image, type(new_image)):
                raise TypeError(
                    "The new and old image must be of the same type "
                    f"(new image: {type(new_image).__name__}, old image: {type(old_image).__name__})!"
                )

            try:
                old_image_idx = image_sequence.index(old_image)
            except ValueError:
                warn(f"The old image is not contained in the subject. No replacement will be performed.")
                return False

            image_sequence[old_image_idx] = new_image
            return True

    def remove_image_by_modality(self, modality: Union[Modality, str]) -> bool:
        """Remove one or multiple images as specified by the modality.

        Args:
            modality (Union[Modality, str]): The modality of all images to remove.

        Returns:
            bool: True when the removal procedure was successful otherwise False.
        """
        if isinstance(modality, str):
            modality = Modality(modality)

        candidates = [img for img in self.intensity_images if img.get_modality() == modality]

        if not candidates:
            return False

        success = True
        for candidate in candidates:
            try:
                self.intensity_images.remove(candidate)
            except ValueError:
                success = False

        return success

    def remove_image_by_organ(self, organ: Union[Organ, str]) -> bool:
        """Remove one or multiple images as specified by the organ.

        Args:
            organ (Union[Organ, str]): The organ of all images to remove.

        Returns:
            bool: True when the removal procedure was successful otherwise False.
        """
        if isinstance(organ, str):
            organ = Organ(organ, None)

        candidates = [img for img in self.segmentation_images if img.get_organ() == organ]

        if not candidates:
            return False

        success = True
        for candidate in candidates:
            try:
                self.segmentation_images.remove(candidate)
            except ValueError:
                success = False

        return success

    def remove_image_by_annotator(self, annotator: Union[Annotator, str]) -> bool:
        """Remove one or multiple images as specified by the annotator.

        Args:
            annotator (Union[Annotator, str]): The annotator of all images to remove.

        Returns:
            bool: True when the removal procedure was successful, otherwise False.
        """
        if isinstance(annotator, str):
            annotator = Annotator(annotator)

        candidates = [img for img in self.segmentation_images if img.get_annotator() == annotator]

        if not candidates:
            return False

        success = True
        for candidate in candidates:
            try:
                self.segmentation_images.remove(candidate)
            except ValueError:
                success = False

        return success

    def remove_image_by_organ_and_annotator(self, organ: Union[Organ, str], annotator: Union[Annotator, str]) -> bool:
        """Remove one or multiple images as specified by the organ and annotator.

        Args:
            organ (Union[Organ, str]): The organ of all images to remove.
            annotator (Union[Annotator, str]): The annotator of all images to remove.

        Returns:
            bool: True when the removal procedure was successful, otherwise False.
        """
        if isinstance(organ, str):
            organ = Organ(organ, None)
        if isinstance(annotator, str):
            annotator = Annotator(annotator)

        candidates = [
            img for img in self.segmentation_images if img.get_organ() == organ and img.get_annotator() == annotator
        ]

        if not candidates:
            return False

        success = True
        for candidate in candidates:
            try:
                self.segmentation_images.remove(candidate)
            except ValueError:
                success = False

        return success

    def remove_image(self, image: Union[IntensityImage, SegmentationImage]) -> bool:
        """Remove a given image from the subject.

        Args:
            image (Union[IntensityImage, SegmentationImage]): The image to remove from the subject.

        Returns:
            bool: True when the removal procedure was successful otherwise False.
        """
        images = self.get_images_by_type(type(image))
        candidates = [img for img in images if img == image]

        if len(candidates) > 1:
            warn(
                f"The removal of the image is ambiguous because there are multiple ({len(candidates)}) "
                "images which are equal. Only the first found image will be removed"
            )

        if not candidates:
            return False

        images.remove(candidates[0])
        return True

    def add_data(self, data: Dict[str, Any]) -> None:
        """Add additional data to the subject.

        Args:
            data (Dict[str, Any]): The additional datas.

        Returns:
            None
        """
        self.data.update(data)

    def add_data_by_key(self, key: str, data: Any) -> None:
        """Add additional data by key to the subject.

        Args:
            key (str): The key of the additional data.
            data (Any): The additional data.

        Returns:
            None
        """
        self.data[key] = data

    def get_data(self) -> Dict[str, Any]:
        """Get the additional data associated with the subject.

        Returns:
            Dict[str, Any]: The additional data associated with the subject.
        """
        return self.data

    def get_data_by_key(self, key: str) -> Any:
        """Get additional data by key or :data:`None` if the key is not existing.

        Args:
            key (str): The key of the specific additional data.

        Returns:
            Any: The data or :data:`None`.
        """
        return self.data.get(key, None)

    def replace_data(self, key: str, new_data: Any, add_if_missing: bool = False) -> bool:
        """Replace data by a new value.

        Args:
            key (str): The key of the additional data.
            new_data (Any): The new additional data.
            add_if_missing (bool): If True, the additional data will be added if the key is not existing
             (default: False).

        Returns:
            bool: True if the additional data is replaced successfully, False otherwise.
        """
        if key not in self.data.keys() and not add_if_missing:
            warn(f"The key {key} is not contained in the additional data. No replacement will be performed.")
            return False

        self.data[key] = new_data
        return True

    def remove_additional_data(self) -> None:
        """Remove all additional data from the subject.

        Returns:
            None
        """
        self.data.clear()

    def remove_additional_data_by_key(self, key: str) -> bool:
        """Remove additional data by a key from the subject.

        Args:
            key (str): The key of the additional data.

        Returns:
            bool: True when the removal procedure was successful otherwise False.
        """
        return self.data.pop(key, None) is not None

    def playback_transform_tapes(self) -> None:
        """Playback the transform tapes.

        Returns:
            None
        """
        for image in self.get_images():
            image.get_transform_tape().playback(image, subject=self)

    def __str__(self) -> str:
        return (
            f"{self.name} (Intensity Images: {len(self.intensity_images)} / "
            f"Segmentation Images: {len(self.segmentation_images)})"
        )
