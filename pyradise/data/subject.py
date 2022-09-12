from typing import (
    Any,
    List,
    Optional,
    Union,
    Sequence,
    Tuple
)
from collections import abc

from .modality import Modality
from .organ import Organ
from .rater import Rater

from .image import (
    IntensityImage,
    SegmentationImage)


__all__ = ['Subject']

OneOrMultipleImagesOrNone = Optional[Union[IntensityImage, SegmentationImage, Sequence[IntensityImage],
                                           Sequence[SegmentationImage]]]


class Subject:
    """The :class:`Subject` is the main data object which holds all :class:`IntensityImage` s and
    :class:`SegmentationImage` s associated with a subject / patient.

    The :class:`Subject` is constructed either manually by the user or by a converter (see :mod:`pyradise.fileio`
    package) or loader (see :mod:`pyradise.fileio` package). The manual construction of the :class:`Subject` provides
    more flexibility if the :class:`Subject` is used in combination with other libraries such as
    `MONAI <https://monai.io/>`_ or `pymia <https://pymia.readthedocs.io/en/latest/#>`_.

    Args:
        name (str): The name of the subject.
        images (OneOrMultipleImagesOrNone): One or multiple images to add to the subject.

    Examples:
        The following example demonstrates the manual construction of a :class:`Subject`:

        >>> from argparse import ArgumentParser
        >>> from typing import Tuple
        >>> import os
        >>>
        >>> import SimpleITK as sitk
        >>>
        >>> from pyradise.data import Subject, IntensityImage, SegmentationImage, Modality, Organ, Rater
        >>> from pyradise.fileio import SubjectWriter, ImageFileFormat
        >>>
        >>>
        >>> def get_segmentation_file_paths(path: str,
        >>>                                 valid_organs: Tuple[Organ, ...]
        >>>                                 ) -> Tuple[str]:
        >>>     file_paths = []
        >>>
        >>>     for file in os.listdir(path):
        >>>         if any(organ.name in file for organ in valid_organs) and file.endswith('.nii.gz'):
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
        >>>         if any(modality.get_name() in file for modality in valid_modalities) and file.endswith('.nii.gz'):
        >>>             file_paths.append(os.path.join(path, file))
        >>>
        >>>     return tuple(sorted(file_paths))
        >>>
        >>>
        >>> def main(input_dir: str,
        >>>          output_dir: str
        >>>          ) -> None:
        >>>     # Retrieve image file paths
        >>>     organs = (Organ('Brainstem'), Organ('Eyes'), Organ('Hippocampi'), Organ('OpticNerves'))
        >>>     modalities = (Modality('CT'), Modality('T1c'), Modality('T1w'), Modality('T2w'))
        >>>
        >>>     segmentation_file_paths = get_segmentation_file_paths(input_dir, organs)
        >>>     intensity_file_paths = get_intensity_file_paths(input_dir, modalities)
        >>>
        >>>     # Load the segmentation image files
        >>>     images = []
        >>>     for path, organ in zip(segmentation_file_paths, organs):
        >>>         image = SegmentationImage(sitk.ReadImage(path, sitk.sitkUInt8), organ, Rater.get_default())
        >>>         images.append(image)
        >>>
        >>>     # Load the intensity image files
        >>>     for path, modality in zip(intensity_file_paths, modalities):
        >>>         image = IntensityImage(sitk.ReadImage(path, sitk.sitkFloat32), modality)
        >>>         images.append(image)
        >>>
        >>>     # Construct the subject
        >>>     subject = Subject(os.path.basename(input_dir), images)
        >>>
        >>>     # Display the subject name and properties of the intensity and segmentation images
        >>>     print(f'Subject {subject.get_name()} contains the following images:')
        >>>
        >>>     for image in subject.intensity_images:
        >>>         print(f'Intensity image of modality {image.get_modality(True)} with size: {image.get_size()}')
        >>>
        >>>     for image in subject.segmentation_images:
        >>>         print(f'Segmentation image of {image.get_organ(True)} with size: {image.get_size()}')
        >>>
        >>>     # Write the subject to disk
        >>>     SubjectWriter(ImageFileFormat.NRRD).write(output_dir, subject, write_transforms=False)
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

    def __init__(self,
                 name: str,
                 images: OneOrMultipleImagesOrNone = None
                 ) -> None:
        super().__init__()

        self.name = name

        self.intensity_images: List[IntensityImage] = []
        self.segmentation_images: List[SegmentationImage] = []

        if isinstance(images, IntensityImage):
            self.intensity_images.append(images)

        if isinstance(images, SegmentationImage):
            self.segmentation_images.append(images)

        if isinstance(images, abc.Sequence):
            for image in images:
                if isinstance(image, IntensityImage):
                    self.intensity_images.append(image)
                elif isinstance(image, SegmentationImage):
                    self.segmentation_images.append(image)
                else:
                    raise ValueError(f'At least one image is not of type {IntensityImage.__class__.__name__} or'
                                     f'{SegmentationImage.__class__.__name__}!')

    def get_name(self) -> str:
        """Get the name of the subject.

        Returns:
            str: The name of the subject.
        """
        return self.name

    def get_raters(self) -> Tuple[Optional[Rater], ...]:
        """Get the raters of the segmentation images.

        Returns:
            Tuple[Optional[Rater], ...]: The raters of the segmentation images.
        """
        raters = [seg.get_rater() for seg in self.segmentation_images]
        return tuple(raters)

    def replace_image(self,
                      old_image: Union[IntensityImage, SegmentationImage],
                      new_image: Union[IntensityImage, SegmentationImage]
                      ) -> bool:
        """Replace an existing image with a new one.

        Args:
            old_image (Union[IntensityImage, SegmentationImage]): The image to be replaced.
            new_image (Union[IntensityImage, SegmentationImage]): The new image.

        Returns:
            bool: True if the replacement was successful otherwise False.
        """
        if not isinstance(old_image, type(new_image)):
            raise TypeError('The old and the new image must be of the same type!')

        if isinstance(old_image, IntensityImage):

            if old_image.get_modality() != new_image.get_modality():
                if new_image.get_modality() in [img.get_modality() for img in self.intensity_images]:
                    raise ValueError(f'There is already an image existing with the same modality '
                                     f'({new_image.get_modality().name}) for subject {self.name}!')

            index = self.intensity_images.index(old_image)
            self.intensity_images[index] = new_image
            return True

        if isinstance(old_image, SegmentationImage):

            if old_image.get_organ() != new_image.get_organ():
                if new_image.get_organ() in [img.get_organ() for img in self.segmentation_images]:
                    raise ValueError(f'There is already an image existing with the same organ '
                                     f'({new_image.get_organ().name}) for subject {self.name}!')

            index = self.segmentation_images.index(old_image)
            self.segmentation_images[index] = new_image
            return True

        return False

    def add_image(self,
                  image: Union[IntensityImage, SegmentationImage],
                  force: bool = False
                  ) -> None:
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
        if isinstance(image, IntensityImage):

            if image.get_modality() in [img.get_modality() for img in self.intensity_images]:
                raise ValueError(f'There is already an image existing with the same modality '
                                 f'({image.get_modality().name}) for subject {self.name}!')

            self.intensity_images.append(image)

        if isinstance(image, SegmentationImage):
            if image.get_organ() in [img.get_organ() for img in self.segmentation_images]:
                if force:
                    new_organ = image.get_organ()
                    available_organs = [seg_image.get_organ(True) for seg_image in self.segmentation_images]

                    available_identifier_found = False
                    index = 0
                    while not available_identifier_found:
                        new_organ = Organ(image.get_organ(as_str=True) + f'_{index}')

                        if new_organ.name not in available_organs:
                            available_identifier_found = True

                        index += 1

                    image.organ = new_organ

                else:
                    raise ValueError(f'There is already an image existing with the same organ '
                                     f'({image.get_organ().name}) for subject {self.name}!')

            self.segmentation_images.append(image)

    def add_images(self,
                   images: Sequence[Union[IntensityImage, SegmentationImage]],
                   force: bool = False
                   ) -> None:
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

    def _remove_image(self,
                      image: Union[IntensityImage, SegmentationImage]
                      ) -> bool:
        if isinstance(image, IntensityImage):
            self.intensity_images.remove(image)
            return True

        if isinstance(image, SegmentationImage):
            self.segmentation_images.remove(image)
            return True

        return False

    def remove_image_by_modality(self,
                                 modality: Modality
                                 ) -> bool:
        """Remove one or multiple images as specified by the modality.

        Args:
            modality (Modality): The modality of all images to remove.

        Returns:
            bool: True when the removal procedure was successful otherwise False.
        """
        candidates = [img for img in self.intensity_images if img.get_modality() == modality]

        if len(candidates) > 1:
            raise ValueError(f'The removal of the image with the modality {modality.name} is ambiguous because '
                             f'there are multiple ({len(candidates)}) images with this modality!')

        if not candidates:
            return False

        return self._remove_image(candidates[0])

    def remove_image_by_organ(self,
                              organ: Union[Organ, str]
                              ) -> bool:
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

        results = []
        for candidate in candidates:
            result = self._remove_image(candidate)
            results.append(result)

        return all(results)

    def remove_image(self,
                     image: Union[IntensityImage, SegmentationImage]
                     ) -> bool:
        """Remove a given image from the subject.

        Args:
            image (Union[IntensityImage, SegmentationImage]): The image to remove from the subject.

        Returns:
            bool: True when the removal procedure was successful otherwise False.
        """
        if isinstance(image, IntensityImage):
            candidates = [img for img in self.intensity_images if img == image]

        else:
            candidates = [img for img in self.segmentation_images if img == image]

        if len(candidates) > 1:
            raise ValueError(f'The removal of the image is ambiguous because there are multiple ({len(candidates)}) '
                             f'images which are equal!')

        if not candidates:
            return False

        return self._remove_image(candidates[0])

    @staticmethod
    def _check_for_single_candidate(candidates: List[Any]) -> Any:
        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            raise ValueError(f'There are multiple candidates which fulfil the criterion ({len(candidates)})!')

        return None

    def get_image_by_modality(self,
                              modality: Union[Modality, str]
                              ) -> Optional[IntensityImage]:
        """Get one intensity image by its modality.

        Args:
            modality (Union[Modality, str]): The modality of the image to retrieve.

        Returns:
            Optional[IntensityImage]: The intensity image or None if there are multiple candidates.
        """
        if isinstance(modality, str):
            modality = Modality(modality)

        candidates = [img for img in self.intensity_images if img.get_modality() == modality]

        return self._check_for_single_candidate(candidates)

    def get_image_by_rater(self,
                           rater: Union[Rater, str]
                           ) -> Optional[Tuple[SegmentationImage]]:
        """Get one or multiple segmentation images by their rater.

        Args:
            rater (Union[Rater, str]): The rater of the image to retrieve.

        Returns:
            Optional[Union[SegmentationImage, Tuple[SegmentationImage]]]: The segmentation images or None if there is
            no image with this rater.
        """
        if isinstance(rater, str):
            rater = Rater(rater)

        candidates: List[SegmentationImage] = [img for img in self.segmentation_images if img.get_rater() == rater]

        if not candidates:
            return None

        return tuple(candidates)

    def get_image_by_organ(self,
                           organ: Union[Organ, str]
                           ) -> Optional[SegmentationImage]:
        """Get one segmentation image by its organ.

        Args:
            organ (Union[Organ, str]): The organ of the image to retrieve.

        Returns:
            Optional[SegmentationImage]: The segmentation image or None if there are multiple candidates.
        """
        if isinstance(organ, str):
            organ = Organ(organ, None)

        candidates = [img for img in self.segmentation_images if img.get_organ().name == organ.name]

        return self._check_for_single_candidate(candidates)

    def set_image(self,
                  image: Union[IntensityImage, SegmentationImage]
                  ) -> None:
        """Replace an image with either the same modality in case of an intensity image or the same organ and rater
        in case of a segmentation image.

        Args:
            image (Union[IntensityImage, SegmentationImage]): The image to replace the existing one.

        Returns:
            None
        """
        if isinstance(image, IntensityImage):

            candidate_indices = []
            for idx, img in enumerate(self.intensity_images):
                if image.get_modality() == img.get_modality():
                    candidate_indices.append(idx)

            if len(candidate_indices) != 1:
                raise ValueError('The setting of the image is ambiguous because there are multiple images fulfilling'
                                 'the criterion!')

            self.intensity_images.remove(self.intensity_images[candidate_indices[0]])
            self.intensity_images.append(image)

        if isinstance(image, SegmentationImage):

            candidate_indices = []
            for idx, img in enumerate(self.segmentation_images):
                criteria = (image.get_organ() == img.get_organ(),
                            image.get_rater() == img.get_rater())
                if all(criteria):
                    candidate_indices.append(idx)

            if len(candidate_indices) != 1:
                raise ValueError('The setting of the image is ambiguous because there are multiple images fulfilling'
                                 'the criterion!')

            self.segmentation_images.remove(self.segmentation_images[candidate_indices[0]])
            self.segmentation_images.append(image)

    def __str__(self) -> str:
        return f'{self.name} (Intensity Images: {len(self.intensity_images)} / ' \
               f'Segmentation Images: {len(self.segmentation_images)})'
