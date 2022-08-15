from abc import (
    ABC,
    abstractmethod)
from typing import (
    Any,
    Union,
    TypeVar,
    Tuple,
    List,
    Optional)
from copy import deepcopy

import SimpleITK as sitk
import numpy as np


__all__ = ['Tape', 'TransformTape', 'TransformationInformation']

# pylint: disable=no-member

# Forward declaration of image types
Image = TypeVar('Image')
IntensityImage = TypeVar('IntensityImage')
SegmentationImage = TypeVar('SegmentationImage')


class Tape(ABC):
    """Abstract base class for a recording tape."""

    def __init__(self) -> None:
        super().__init__()
        self.recordings = []

    @abstractmethod
    def record(self, value: Any) -> None:
        """Record a value on the tape.

        Args:
            value (Any): The value to be recorded.

        Returns:
            None
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def playback(data: Any) -> Any:
        """Playback the recorded elements on the tape of the data object.

        Args:
            data (Any): The data on which the playback should take place. This object need to contain also the tape.

        Returns:
            Any: The back played data.
        """
        raise NotImplementedError()

    def get_recorded_elements(self,
                              reverse: bool = False
                              ) -> Tuple[Any, ...]:
        """Get the recorded elements on the tape.

        Args:
            reverse (bool): Indicates if the recordings should be returned in reverse order.

        Returns:
            Tuple[Any, ...]: The recorded elements of the tape.
        """
        if reverse:
            return tuple(reversed(self.recordings))

        return tuple(self.recordings)

    def reset(self) -> None:
        """Reset the transformation tape.

        Returns:
            None
        """
        self.recordings = []


class TransformationInformation:
    """A class holding information about a transformation of an image.

    Args:
        name (str): The name of the operation.
        transform (sitk.Transform): The transformation.
        pre_transform_origin (Tuple[float]): The origin of the pre-transform image.
        pre_transform_direction (Tuple[float]): The direction of the pre-transform image.
        pre_transform_spacing (Tuple[float]): The spacing of the pre-transform image.
        pre_transform_size (Tuple[float]): The size of the pre-transform image.
        post_transform_origin (Tuple[float]): The origin of the transformed image.
        post_transform_direction (Tuple[float]): The direction of the transformed image.
        post_transform_spacing (Tuple[float]): The spacing of the transformed image.
        post_transform_size (Tuple[float]): The size of the transformed image.
        pre_transform_orientation (Optional[str]): The orientation identifier of the pre-transform image (e.g. RAS).
        post_transform_orientation (Optional[str]): The orientation identifier of the transformed image (e.g. RAS).
    """

    def __init__(self,
                 name: str,
                 transform: sitk.Transform,
                 pre_transform_origin: Tuple[float],
                 pre_transform_direction: Tuple[float],
                 pre_transform_spacing: Tuple[float],
                 pre_transform_size: Tuple[float],
                 post_transform_origin: Tuple[float],
                 post_transform_direction: Tuple[float],
                 post_transform_spacing: Tuple[float],
                 post_transform_size: Tuple[float],
                 pre_transform_orientation: Optional[str] = None,
                 post_transform_orientation: Optional[str] = None,
                 ) -> None:
        # pylint: disable=too-many-arguments

        super().__init__()

        assert len(pre_transform_origin) == 3, 'The pre-transform image origin must be of length 3!'
        assert len(pre_transform_direction) == 9, 'The pre-transform image direction must be of length 9!'
        assert len(pre_transform_spacing) == 3, 'The pre-transform image spacing must be of length 3!'
        assert len(pre_transform_size) == 3, 'The pre-transform image size must be of length 3!'

        assert len(post_transform_origin) == 3, 'The origin of the transformed image must be of length 3!'
        assert len(post_transform_direction) == 9, 'The direction of the transformed image must be of length 9!'
        assert len(post_transform_spacing) == 3, 'The spacing of the transformed image must be of length 3!'
        assert len(post_transform_size) == 3, 'The size of the transformed image must be of length 3!'

        self.name = name

        self.transform = transform

        self.pre_transform_origin = pre_transform_origin
        self.pre_transform_direction = pre_transform_direction
        self.pre_transform_spacing = pre_transform_spacing
        self.pre_transform_size = pre_transform_size

        self.post_transform_origin = post_transform_origin
        self.post_transform_direction = post_transform_direction
        self.post_transform_spacing = post_transform_spacing
        self.post_transform_size = post_transform_size

        if isinstance(pre_transform_orientation, str) and len(pre_transform_orientation) != 3:
            raise ValueError(f'The pre-transform image orientation ({pre_transform_orientation}) is invalid!')

        if isinstance(post_transform_orientation, str) and len(post_transform_orientation) != 3:
            raise ValueError(f'The post-transform image orientation ({post_transform_orientation}) is invalid!')

        self.pre_transform_orientation = pre_transform_orientation
        self.post_transform_orientation = post_transform_orientation

    @classmethod
    def from_images(cls,
                    name: str,
                    transform: sitk.Transform,
                    pre_transform_image: sitk.Image,
                    post_transform_image: sitk.Image
                    ) -> "TransformationInformation":
        """Construct a transformation information from the pre- and post-transform images.

        Args:
            name (str): The name
            transform (sitk.Transform): The transformation.
            pre_transform_image (sitk.Image): The pre-transformed image.
            post_transform_image (sitk.Image): The transformed image.

        Returns:
            TransformationInformation: The instance holding the information about the transformation.
        """

        orient_filter = sitk.DICOMOrientImageFilter()
        pre_orientation = orient_filter.GetOrientationFromDirectionCosines(pre_transform_image.GetDirection())
        post_orientation = orient_filter.GetOrientationFromDirectionCosines(post_transform_image.GetDirection())

        return cls(name, transform,
                   pre_transform_image.GetOrigin(),
                   pre_transform_image.GetDirection(),
                   pre_transform_image.GetSpacing(),
                   pre_transform_image.GetSize(),
                   post_transform_image.GetOrigin(),
                   post_transform_image.GetDirection(),
                   post_transform_image.GetSpacing(),
                   post_transform_image.GetSize(),
                   pre_orientation, post_orientation)

    @staticmethod
    def get_matrix_from_direction(direction: Tuple[float]) -> np.ndarray:
        """Reshape a direction tuple into a direction matrix.

        Args:
            direction (Tuple[float]): The direction as a tuple of floats.

        Returns:
            np.ndarray: The 3x3 direction matrix.
        """
        return np.array(direction).reshape(3, 3)

    def get_transform(self,
                      inverse: bool
                      ) -> sitk.Transform:
        """Get the transformation.

        Args:
            inverse (bool): If true the inverse of the transformation is returned.

        Returns:
            sitk.Transform: The transformation.
        """
        if inverse:
            return deepcopy(self.transform).GetInverse()

        return self.transform

    def get_origin(self,
                   pre_transform: bool
                   ) -> Tuple[float]:
        """Get the origin before or after the transformation of the image.

        Args:
            pre_transform (bool): If True the origin of the pre-transformed image is returned.
             Otherwise, the origin from the post-transformed image is returned.

        Returns:
            Tuple[float]: The origin.
        """
        if pre_transform:
            return self.pre_transform_origin

        return self.post_transform_origin

    def get_direction(self,
                      pre_transform: bool,
                      as_matrix: bool = False
                      ) -> Union[Tuple[float], np.ndarray]:
        """Get the direction before or after the transformation of the image.

        Args:
            pre_transform (bool): If True the direction of the pre-transformed image is returned.
             Otherwise, the direction from the post-transformed image is returned.
            as_matrix (bool): If True the direction is returned as a 3x3-matrix.

        Returns:
            Union[Tuple[float], np.ndarray]: The direction.
        """
        if pre_transform:
            direction = self.pre_transform_direction
        else:
            direction = self.post_transform_direction

        if as_matrix:
            direction = self.get_matrix_from_direction(direction)

        return direction

    def get_spacing(self,
                    pre_transform: bool
                    ) -> Tuple[float]:
        """Get the spacing before or after the transformation of the image.

        Args:
            pre_transform (bool): If True the spacing of the pre-transformed image is returned.
             Otherwise, the spacing from the post-transformed image is returned.

        Returns:
            Tuple[float]: The spacing.
        """
        if pre_transform:
            return self.pre_transform_spacing

        return self.post_transform_spacing

    def get_size(self,
                 pre_transform: bool
                 ) -> Tuple[float]:
        """Get the size before or after the transformation of the image.

        Args:
            pre_transform (bool): If True the size of the pre-transformed image is returned.
             Otherwise, the size from the post-transformed image is returned.

        Returns:
            Tuple[float]: The size.
        """
        if pre_transform:
            return self.pre_transform_size

        return self.post_transform_size

    def get_orientation(self,
                        pre_transform: bool
                        ) -> Optional[str]:
        """Get the orientation before or after the transformation of the image.

        Args:
            pre_transform (bool): If True the orientation of the pre-transformed image is returned.
             Otherwise, the orientation from the post-transformed image is returned.

        Returns:
            Optional[str]: The orientation.
        """
        if pre_transform:
            return self.pre_transform_orientation

        return self.post_transform_orientation


class TransformTape(Tape):
    """A class representing a tape for accumulating transformations."""

    def __init__(self) -> None:
        super().__init__()

        self.recordings: List[TransformationInformation] = []

    def record(self, value: TransformationInformation) -> None:
        """Record a transformation information on the tape.

        Args:
            value (TransformationInformation): The transformation information to be recorded.

        Returns:
            None
        """

        self.recordings.append(value)

    @staticmethod
    def is_reorient_only(transform_info: TransformationInformation,
                         invert: bool
                         ) -> bool:
        """Check if the transform info is a re-orientation.

        Args:
            transform_info (TransformationInformation): The transform info to check.
            invert (bool): Indicates if the transform should be inverted.

        Returns:
            bool: True if the transformation info is specifying a reorientation of the image, otherwise False.
        """
        transform = transform_info.get_transform(invert)

        identity_transform = sitk.Transform(transform.GetDimension(), transform.GetTransformEnum())
        identity_transform.SetIdentity()

        return transform.GetParameters() == identity_transform.GetParameters()

    # noinspection DuplicatedCode
    @staticmethod
    def playback(data: Union[IntensityImage, SegmentationImage]) -> Union[IntensityImage, SegmentationImage]:

        image_sitk = data.get_image(as_sitk=True)

        for transform_infos in reversed(data.get_transform_tape().recordings):  # type: TransformationInformation

            if TransformTape.is_reorient_only(transform_infos, invert=True):
                image_sitk = sitk.DICOMOrient(image_sitk, transform_infos.pre_transform_orientation)

            else:
                if data.is_intensity_image():
                    interpolator = sitk.sitkBSpline
                    default_pixel_value = np.min(sitk.GetArrayFromImage(image_sitk))
                else:
                    interpolator = sitk.sitkNearestNeighbor
                    default_pixel_value = 0

                transform = transform_infos.get_transform(inverse=True)

                resample_filter = sitk.ResampleImageFilter()
                resample_filter.SetTransform(transform)
                resample_filter.SetInterpolator(interpolator)
                resample_filter.SetDefaultPixelValue(default_pixel_value)
                resample_filter.SetSize(transform_infos.get_size(True))
                resample_filter.SetOutputOrigin(transform_infos.get_origin(True))
                resample_filter.SetOutputDirection(transform_infos.get_direction(True, False))
                resample_filter.SetOutputSpacing(transform_infos.get_spacing(True))

                image_sitk = resample_filter.Execute(image_sitk)

        data.set_image(image_sitk)
        data.transform_tape.recordings.clear()

        return data

    def get_recorded_elements(self,
                              reverse: bool = False
                              ) -> Tuple[TransformationInformation, ...]:
        """Get the recorded transformation information entries on the tape.

        Args:
            reverse (bool): Indicates if the recordings should be returned in reverse order.

        Returns:
            Tuple[TransformationInformation, ...]: The recorded transformation information entries of the tape.
        """
        # pylint: disable=useless-super-delegation
        return super().get_recorded_elements(reverse)
