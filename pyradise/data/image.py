from abc import (
    ABC,
    abstractmethod)
from typing import (
    Tuple,
    Union,
    Optional)

import SimpleITK as sitk
import itk
import numpy as np

from .modality import Modality
from .organ import (
    Organ,
    OrganRaterCombination)
from .rater import Rater
from .taping import TransformTape


__all__ = ['Image', 'IntensityImage', 'SegmentationImage']


class Image(ABC):
    """Abstract base class for images.

    Args:
        image (Union[sitk.Image, itk.Image]): The image data to be stored.
    """

    def __init__(self,
                 image: Union[sitk.Image, itk.Image]
                 ) -> None:
        super().__init__()

        if isinstance(image, sitk.Image):
            self.image: itk.Image = self.convert_to_itk_image(image)

        else:
            self.image: itk.Image = image

        self.transform_tape = TransformTape()

    @abstractmethod
    def is_intensity_image(self) -> bool:
        """Return True if the image is an intensity image otherwise False.

        Returns:
            bool: n/a
        """
        raise NotImplementedError()

    @staticmethod
    def convert_to_sitk_image(image: itk.Image) -> sitk.Image:
        """Convert an ITK image to a SimpleITK image.

        Args:
            image (itk.Image): The ITK image to be converted.

        Returns:
            sitk.Image: The SimpleITK image.
        """
        is_vector_image = image.GetNumberOfComponentsPerPixel() > 1
        image_sitk = sitk.GetImageFromArray(itk.GetArrayFromImage(image), isVector=is_vector_image)
        image_sitk.SetOrigin(tuple(image.GetOrigin()))
        image_sitk.SetSpacing(tuple(image.GetSpacing()))
        image_sitk.SetDirection(itk.GetArrayFromMatrix(image.GetDirection()).flatten())
        return image_sitk

    @staticmethod
    def convert_to_itk_image(image: sitk.Image) -> itk.Image:
        """Convert a SimpleITK image to an ITK image.

        Args:
            image (sitk.Image): The SimpleITK image to be converted.

        Returns:
            itk.Image: The ITK image.
        """
        is_vector_image = image.GetNumberOfComponentsPerPixel() > 1
        image_itk = itk.GetImageFromArray(sitk.GetArrayFromImage(image), is_vector=is_vector_image)
        image_itk.SetOrigin(image.GetOrigin())
        image_itk.SetSpacing(image.GetSpacing())
        image_itk.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(image.GetDirection()), [3] * 2)))
        return image_itk

    def get_image(self, as_sitk: bool = False) -> Union[itk.Image, sitk.Image]:
        """Get the image as the appropriate type.

        Args:
            as_sitk (bool): If True returns the image as a SimpleITK image else as a ITK image.

        Returns:
            Union[itk.Image, sitk.Image]: The image as the appropriate data type.
        """
        if as_sitk:
            return self.convert_to_sitk_image(self.image)

        return self.image

    def set_image(self, image: Union[sitk.Image, itk.Image]) -> None:
        """Set the image.

        Args:
            image (Union[sitk.Image, itk.Image]): The image to be set.

        Returns:
            None
        """
        if isinstance(image, sitk.Image):
            self.image = self.convert_to_itk_image(image)

        else:
            self.image = image

    @staticmethod
    def _return_image_as(image: Union[sitk.Image, itk.Image],
                         as_sitk: bool
                         ) -> Union[sitk.Image, itk.Image]:
        if isinstance(image, sitk.Image) and as_sitk:
            return image

        if isinstance(image, sitk.Image) and not as_sitk:
            return Image.convert_to_itk_image(image)

        if isinstance(image, itk.Image) and as_sitk:
            return Image.convert_to_sitk_image(image)

        return image

    @staticmethod
    def cast(image: Union[sitk.Image, itk.Image],
             pixel_type: Union[itk.support.types.itkCType, int],
             as_sitk: bool = False
             ) -> Union[sitk.Image, itk.Image]:
        """Cast an image to a certain pixel type and return it as a specified data type.

        Args:
            image (Union[sitk.Image, itk.Image]): The image to be casted.
            pixel_type (Union[itk.support.types.itkCType, int]): The output pixel type.
            as_sitk (bool): If True the image gets returned as a SimpleITK image otherwise as an ITK image.

        Returns:
            Union[sitk.Image, itk.Image]: The casted image.
        """
        if isinstance(image, sitk.Image):
            img = sitk.Cast(image, pixel_type)

        else:
            dimensions = itk.template(image)[1][1]
            input_image_type = itk.Image[itk.template(image)[1]]
            output_image_type = itk.Image[pixel_type, dimensions]

            cast_fltr = itk.CastImageFilter[input_image_type, output_image_type].New()
            cast_fltr.SetInput(image)
            cast_fltr.Update()
            img = cast_fltr.GetOutput()

        return Image._return_image_as(img, as_sitk)

    def get_origin(self) -> Tuple[float, ...]:
        """Get the origin of the image.

        Returns:
            Tuple[float, ...]: The origin of the image.
        """
        return tuple(self.image.GetOrigin())

    def get_direction(self) -> np.ndarray:
        """Get the direction of the image.

        Returns:
            np.ndarray: The direction of the image.
        """
        return itk.GetArrayFromMatrix(self.image.GetDirection())

    def get_spacing(self) -> Tuple[float, ...]:
        """Get the spacing of the image.

        Returns:
            Tuple[float, ...]: The spacing of the image
        """
        return tuple(self.image.GetSpacing())

    def get_size(self) -> Tuple[int, ...]:
        """Get the size of the image.

        Returns:
            Tuple[int, ...]: The size of the image.
        """
        return tuple(self.image.GetLargestPossibleRegion().GetSize())

    def ndim(self) -> int:
        """Get the number of image dimensions.

        Returns:
            int: The number of image dimensions.
        """
        return itk.template(self.image)[1][1]

    @staticmethod
    def get_image_type(image: Union[sitk.Image, itk.Image]) -> itk.Image:
        """Get the image type from an image.

        Args:
            image (Union[sitk.Image, itk.Image]): The image to get the image type from.

        Returns:
            itk.Image: The image type.
        """
        if isinstance(image, sitk.Image):
            image = Image.convert_to_itk_image(image)
        return itk.Image[itk.template(image)[1]]

    def get_image_type_(self) -> itk.Image:
        """Get the image type of this image.

        Returns:
            itk.Image: The image type of this image.
        """
        return itk.Image[itk.template(self.image)[1]]

    def get_transform_tape(self) -> TransformTape:
        """Get the transform tape.

        Returns:
            TransformTape: The transform tape of the image.
        """
        return self.transform_tape


class IntensityImage(Image):
    """A class representing an intensity image.

    Args:
        image (Union[sitk.Image, itk.Image]): The image data to add to the image.
        modality (Modality): The image modality.
    """

    def __init__(self,
                 image: Union[sitk.Image, itk.Image],
                 modality: Modality
                 ) -> None:
        super().__init__(image)

        self.modality = modality

    def is_intensity_image(self) -> bool:
        """If the image is an intensity image this function returns True otherwise False.

        Returns:
            bool: True
        """
        return True

    def get_modality(self,
                     as_str: bool = False
                     ) -> Union[Modality, str]:
        """Get the modality of the image.

        Args:
            as_str (bool): If true returns the modality as a string otherwise as type Modality.

        Returns:
            Union[Modality, str]: The modality of the image.
        """
        if as_str:
            return self.modality.name

        return self.modality

    def __eq__(self, other) -> bool:
        return all((self.modality == other.modality, self.image == other.image))

    def __str__(self) -> str:
        return f'Intensity image: {self.modality.name}'


class SegmentationImage(Image):
    """A class representing a segmentation image.

    Args:
        image (Union[sitk.Image, itk.Image]): The segmentation image data.
        organ (Organ): The organ represented by the segmentation image.
        rater (Optional[Rater]): The rater of the segmentation image (default=None).
    """

    def __init__(self,
                 image: Union[sitk.Image, itk.Image],
                 organ: Organ,
                 rater: Optional[Rater] = None
                 ) -> None:
        super().__init__(image)
        self.organ: Organ = organ
        self.rater: Optional[Rater] = rater

    def is_intensity_image(self) -> bool:
        """If the image is an intensity image this function returns True otherwise False.

        Returns:
            bool: False
        """
        return False

    def get_organ(self,
                  as_str: bool = False
                  ) -> Union[Organ, str]:
        """Get the organ.

        Args:
            as_str (bool): It True the organ gets returned as a string otherwise of type Organ.

        Returns:
            Union[Organ, str]: The organ.
        """
        if as_str:
            return self.organ.name

        return self.organ

    def get_rater(self) -> Optional[Rater]:
        """Get the rater if available.

        Returns:
            Optional[Rater]: The rater.
        """
        return self.rater

    def get_organ_rater_combination(self) -> Optional[OrganRaterCombination]:
        """Get the organ rater combination if available.

        Notes:
            Returns None if the rater is not available!

        Returns:
            OrganRaterCombination: The combination of the organ and the rater if available.
        """
        if not self.rater:
            return None

        return OrganRaterCombination(self.organ, self.rater)

    def __eq__(self, other) -> bool:
        return all((self.organ == other.organ, self.rater == other.rater, self.image == other.image))

    def __str__(self) -> str:
        if not self.rater:
            return f'SegmentationImage: {self.organ.name}'

        return f'SegmentationImage: {self.organ.name} / {self.rater.name}'
