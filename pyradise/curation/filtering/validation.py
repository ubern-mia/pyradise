from typing import (
    Tuple,
    Union)

import itk

from pyradise.data import (
    Subject,
    IntensityImage,
    Rater,
    Organ,
    SegmentationImage)
from .base import (
    Filter,
    FilterParameters)


__all__ = ['SegmentationCheckingFilterParameters', 'SegmentationCheckingFilter']


# pylint: disable = too-few-public-methods
class SegmentationCheckingFilterParameters(FilterParameters):
    """A class representing parameters for a SegmentationCheckingFilter.

    Args:
        required_organs (Tuple[Organ, ...]): The organs which must be contained in the subject.
        reference_image (Union[IntensityImage, SegmentationImage]): The reference image for newly constructed label
         images.
        default_rater (Union[Rater, str]): The default rater for newly constructed label images.
        strict (bool): If True organs in the subject which are not part of the required organs will be deleted
         (default=False).
    """

    def __init__(self,
                 required_organs: Tuple[Organ, ...],
                 reference_image: Union[IntensityImage, SegmentationImage],
                 default_rater: Union[Rater, str] = 'unknown',
                 strict: bool = False
                 ) -> None:
        super().__init__()
        self.required_organs = required_organs
        self.reference_image = reference_image
        self.strict = strict

        if isinstance(default_rater, str):
            self.default_rater = Rater(default_rater)
        else:
            self.default_rater = default_rater


class SegmentationCheckingFilter(Filter):
    """A class which checks and may correct missing or additional labels within a subject."""

    @staticmethod
    def _add_new_organ_image(subject: Subject,
                             organ: Organ,
                             params: SegmentationCheckingFilterParameters
                             ) -> None:
        """Construct and adds a new organ image to the subject.

        Args:
            subject (Subject): The subject to add the newly constructed image.
            organ (Organ): The organ the newly constructed image should have.
            params (SegmentationCheckingFilterParameters): The parameters specifying the new image properties.

        Returns:
            None
        """
        reference_img_itk = params.reference_image.get_image()

        pixel_type = itk.ctype('unsigned char')
        dimensions = reference_img_itk.GetImageDimension()
        size = reference_img_itk.GetLargestPossibleRegion().GetSize()

        new_image = itk.Image[pixel_type, dimensions].New()

        new_index = itk.Index[dimensions]()
        new_size = itk.Size[dimensions]()

        for i in range(dimensions):
            new_index[i] = 0
            new_size[i] = size[i]

        new_region = itk.ImageRegion[dimensions]()
        new_region.SetSize(new_size)
        new_region.SetIndex(new_index)

        new_image.SetRegions(new_region)
        new_image.Allocate()

        new_image.SetOrigin(reference_img_itk.GetOrigin())
        new_image.SetDirection(reference_img_itk.GetDirection())
        new_image.SetSpacing(reference_img_itk.GetSpacing())

        new_segmentation = SegmentationImage(new_image, organ, params.default_rater)
        subject.add_image(new_segmentation)

    def execute(self,
                subject: Subject,
                params: SegmentationCheckingFilterParameters
                ) -> Subject:
        """Execute the segmentation checking procedure.

        Args:
            subject (Subject): The subject to check and maybe to modify.
            params (SegmentationCheckingFilterParameters): The filters parameters.

        Returns:
            Subject: The processed subject.
        """
        present_organs = [segmentation.get_organ() for segmentation in subject.segmentation_images]

        missing_organs = []
        for organ in params.required_organs:
            if organ not in present_organs:
                missing_organs.append(organ)

        for organ in missing_organs:
            self._add_new_organ_image(subject, organ, params)

        if params.strict:
            for organ in present_organs:
                if organ not in params.required_organs:
                    indexes = [i for (i, segmentation) in enumerate(subject.segmentation_images)
                               if segmentation.get_organ() == organ]
                    for idx in indexes:
                        subject.segmentation_images.remove(subject.segmentation_images[idx])

        return subject
