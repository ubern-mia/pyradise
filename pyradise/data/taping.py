from abc import (
    ABC,
    abstractmethod)
from typing import (
    Any,
    TypeVar,
    Dict,
    Tuple,
    Optional)

import SimpleITK as sitk
import numpy as np


__all__ = ['Tape', 'TransformTape', 'TransformInfo']

# pylint: disable=no-member

# Forward declaration of image types
Image = TypeVar('Image')
IntensityImage = TypeVar('IntensityImage')
SegmentationImage = TypeVar('SegmentationImage')
Filter = TypeVar('Filter')
FilterParameters = TypeVar('FilterParameters')
ImageProperties = TypeVar('ImageProperties')
Subject = TypeVar('Subject')


class Tape(ABC):
    """An abstract class for a tape which records defined elements and can replay them upon request.
    """

    def __init__(self) -> None:
        super().__init__()
        self.recordings = []

    @abstractmethod
    def record(self, value: Any) -> None:
        """Record a value on the :class:`Tape`.

        Args:
            value (Any): The value to be recorded.

        Returns:
            None
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def playback(data: Any, **kwargs) -> Any:
        """Playback the recorded elements of the :class:`Tape` on the data object.

        Args:
            data (Any): The data on which the playback should take place. This object need to contain also the tape.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The back played data.
        """
        raise NotImplementedError()

    def get_recorded_elements(self,
                              reverse: bool = False
                              ) -> Tuple[Any, ...]:
        """Get the recorded elements on the :class:`Tape`.

        Args:
            reverse (bool): Indicates if the recordings should be returned in reverse order.

        Returns:
            Tuple[Any, ...]: The recorded elements of the :class:`Tape`.
        """
        if reverse:
            return tuple(reversed(self.recordings))

        return tuple(self.recordings)

    def reset(self) -> None:
        """Reset the :class:`Tape`.

        Returns:
            None
        """
        self.recordings = []


# class TransformationInformation:
#     """A class for holding information about the transformation of an image.
#
#     Notes:
#         This class is used in combination with the :class:`TransformTape` to record transformations applied to a
#         certain :class:`~pyradise.data.image.Image`.
#
#     Args:
#         name (str): The name of the operation.
#         transform (sitk.Transform): The transformation.
#         pre_transform_origin (Tuple[float]): The origin of the pre-transform image.
#         pre_transform_direction (Tuple[float]): The direction of the pre-transform image.
#         pre_transform_spacing (Tuple[float]): The spacing of the pre-transform image.
#         pre_transform_size (Tuple[float]): The size of the pre-transform image.
#         post_transform_origin (Tuple[float]): The origin of the transformed image.
#         post_transform_direction (Tuple[float]): The direction of the transformed image.
#         post_transform_spacing (Tuple[float]): The spacing of the transformed image.
#         post_transform_size (Tuple[float]): The size of the transformed image.
#         pre_transform_orientation (Optional[str]): The orientation identifier of the pre-transform image (e.g. RAS).
#         post_transform_orientation (Optional[str]): The orientation identifier of the transformed image (e.g. RAS).
#     """
#
#     def __init__(self,
#                  name: str,
#                  transform: sitk.Transform,
#                  pre_transform_origin: Tuple[float],
#                  pre_transform_direction: Tuple[float],
#                  pre_transform_spacing: Tuple[float],
#                  pre_transform_size: Tuple[float],
#                  post_transform_origin: Tuple[float],
#                  post_transform_direction: Tuple[float],
#                  post_transform_spacing: Tuple[float],
#                  post_transform_size: Tuple[float],
#                  pre_transform_orientation: Optional[str] = None,
#                  post_transform_orientation: Optional[str] = None,
#                  ) -> None:
#         # pylint: disable=too-many-arguments
#
#         super().__init__()
#
#         assert len(pre_transform_origin) == 3, 'The pre-transform image origin must be of length 3!'
#         assert len(pre_transform_direction) == 9, 'The pre-transform image direction must be of length 9!'
#         assert len(pre_transform_spacing) == 3, 'The pre-transform image spacing must be of length 3!'
#         assert len(pre_transform_size) == 3, 'The pre-transform image size must be of length 3!'
#
#         assert len(post_transform_origin) == 3, 'The origin of the transformed image must be of length 3!'
#         assert len(post_transform_direction) == 9, 'The direction of the transformed image must be of length 9!'
#         assert len(post_transform_spacing) == 3, 'The spacing of the transformed image must be of length 3!'
#         assert len(post_transform_size) == 3, 'The size of the transformed image must be of length 3!'
#
#         self.name = name
#
#         self.transform = transform
#
#         self.pre_transform_origin = pre_transform_origin
#         self.pre_transform_direction = pre_transform_direction
#         self.pre_transform_spacing = pre_transform_spacing
#         self.pre_transform_size = pre_transform_size
#
#         self.post_transform_origin = post_transform_origin
#         self.post_transform_direction = post_transform_direction
#         self.post_transform_spacing = post_transform_spacing
#         self.post_transform_size = post_transform_size
#
#         if isinstance(pre_transform_orientation, str) and len(pre_transform_orientation) != 3:
#             raise ValueError(f'The pre-transform image orientation ({pre_transform_orientation}) is invalid!')
#
#         if isinstance(post_transform_orientation, str) and len(post_transform_orientation) != 3:
#             raise ValueError(f'The post-transform image orientation ({post_transform_orientation}) is invalid!')
#
#         self.pre_transform_orientation: Optional[str] = pre_transform_orientation
#         self.post_transform_orientation: Optional[str] = post_transform_orientation
#
#     @classmethod
#     def from_images(cls,
#                     name: str,
#                     transform: sitk.Transform,
#                     pre_transform_image: sitk.Image,
#                     post_transform_image: sitk.Image
#                     ) -> "TransformationInformation":
#         """Construct a :class:`TransformationInformation` from the pre- and post-transform images.
#
#         Args:
#             name (str): The name
#             transform (sitk.Transform): The transformation.
#             pre_transform_image (sitk.Image): The pre-transformed image.
#             post_transform_image (sitk.Image): The transformed image.
#
#         Returns:
#             TransformationInformation: The instance holding the information about the transformation.
#         """
#
#         orient_filter = sitk.DICOMOrientImageFilter()
#         pre_orientation = orient_filter.GetOrientationFromDirectionCosines(pre_transform_image.GetDirection())
#         post_orientation = orient_filter.GetOrientationFromDirectionCosines(post_transform_image.GetDirection())
#
#         return cls(name, transform,
#                    pre_transform_image.GetOrigin(),
#                    pre_transform_image.GetDirection(),
#                    pre_transform_image.GetSpacing(),
#                    pre_transform_image.GetSize(),
#                    post_transform_image.GetOrigin(),
#                    post_transform_image.GetDirection(),
#                    post_transform_image.GetSpacing(),
#                    post_transform_image.GetSize(),
#                    pre_orientation, post_orientation)
#
#     @staticmethod
#     def _get_matrix_from_direction(direction: Tuple[float]) -> np.ndarray:
#         """Reshape a direction tuple into a direction matrix.
#
#         Args:
#             direction (Tuple[float]): The direction as a tuple of floats.
#
#         Returns:
#             np.ndarray: The 3x3 direction matrix.
#         """
#         return np.array(direction).reshape(3, 3)
#
#     def get_transform(self,
#                       inverse: bool
#                       ) -> sitk.Transform:
#         """Get the transformation.
#
#         Args:
#             inverse (bool): If true the inverse of the transformation is returned.
#
#         Returns:
#             sitk.Transform: The transformation.
#         """
#         if inverse:
#             return deepcopy(self.transform).GetInverse()
#
#         return self.transform
#
#     def get_origin(self,
#                    pre_transform: bool
#                    ) -> Tuple[float]:
#         """Get the origin before or after the transformation of the image.
#
#         Args:
#             pre_transform (bool): If True the origin of the pre-transformed image is returned, otherwise the origin
#              from the post-transformed image is returned.
#
#         Returns:
#             Tuple[float]: The origin.
#         """
#         if pre_transform:
#             return self.pre_transform_origin
#
#         return self.post_transform_origin
#
#     def get_direction(self,
#                       pre_transform: bool,
#                       as_matrix: bool = False
#                       ) -> Union[Tuple[float], np.ndarray]:
#         """Get the direction before or after the transformation of the image.
#
#         Args:
#             pre_transform (bool): If True the direction of the pre-transformed image is returned.
#              Otherwise, the direction from the post-transformed image is returned.
#             as_matrix (bool): If True the direction is returned as a 3x3-matrix.
#
#         Returns:
#             Union[Tuple[float], np.ndarray]: The direction.
#         """
#         if pre_transform:
#             direction = self.pre_transform_direction
#         else:
#             direction = self.post_transform_direction
#
#         if as_matrix:
#             direction = self._get_matrix_from_direction(direction)
#
#         return direction
#
#     def get_spacing(self,
#                     pre_transform: bool
#                     ) -> Tuple[float]:
#         """Get the spacing before or after the transformation of the image.
#
#         Args:
#             pre_transform (bool): If True the spacing of the pre-transformed image is returned.
#              Otherwise, the spacing from the post-transformed image is returned.
#
#         Returns:
#             Tuple[float]: The spacing.
#         """
#         if pre_transform:
#             return self.pre_transform_spacing
#
#         return self.post_transform_spacing
#
#     def get_size(self,
#                  pre_transform: bool
#                  ) -> Tuple[float]:
#         """Get the size before or after the transformation of the image.
#
#         Args:
#             pre_transform (bool): If True the size of the pre-transformed image is returned.
#              Otherwise, the size from the post-transformed image is returned.
#
#         Returns:
#             Tuple[float]: The size.
#         """
#         if pre_transform:
#             return self.pre_transform_size
#
#         return self.post_transform_size
#
#     def get_orientation(self,
#                         pre_transform: bool
#                         ) -> Optional[str]:
#         """Get the orientation before or after the transformation of the image.
#
#         Args:
#             pre_transform (bool): If True the orientation of the pre-transformed image is returned.
#              Otherwise, the orientation from the post-transformed image is returned.
#
#         Returns:
#             Optional[str]: The orientation.
#         """
#         if pre_transform:
#             return self.pre_transform_orientation
#
#         return self.post_transform_orientation


# class TransformTape(Tape):
#     """A transformation tape class to record and playback transformations.
#     """
#
#     def __init__(self) -> None:
#         super().__init__()
#
#         self.recordings: List[TransformationInformation] = []
#
#     def record(self, value: TransformationInformation) -> None:
#         """Record a :class:`TransformationInformation` on the tape.
#
#         Args:
#             value (TransformationInformation): The :class:`TransformationInformation` to be recorded.
#
#         Returns:
#             None
#         """
#
#         self.recordings.append(value)
#
#     @staticmethod
#     def is_reorient_only(transform_info: TransformationInformation,
#                          invert: bool
#                          ) -> bool:
#         """Check if the transform info is a re-orientation of the image.
#
#         Args:
#             transform_info (TransformationInformation): The :class:`TransformationInformation` to check.
#             invert (bool): Indicates if the transform should be inverted.
#
#         Returns:
#             bool: True if the :class:`TransformationInformation` is specifying a reorientation of the image,
#             otherwise False.
#         """
#         transform = transform_info.get_transform(invert)
#
#         identity_transform = sitk.Transform(transform.GetDimension(), transform.GetTransformEnum())
#         identity_transform.SetIdentity()
#
#         return transform.GetParameters() == identity_transform.GetParameters()
#
#     # noinspection DuplicatedCode
#     @staticmethod
#     def playback(data: Union[IntensityImage, SegmentationImage], **kwargs) -> Union[IntensityImage, SegmentationImage]:
#         """Play back the transformations of the provided image.
#
#         Args:
#             data (Union[IntensityImage, SegmentationImage]): The image to play back the transformations on.
#             **kwargs: Additional keyword arguments.
#         Returns:
#             Union[IntensityImage, SegmentationImage]: The played back transformed image.
#         """
#
#         image_sitk = data.get_image(as_sitk=True)
#
#         for transform_infos in reversed(data.get_transform_tape().recordings):  # type: TransformationInformation
#
#             if TransformTape.is_reorient_only(transform_infos, invert=True):
#                 image_sitk = sitk.DICOMOrient(image_sitk, transform_infos.pre_transform_orientation)
#
#             else:
#                 if data.is_intensity_image():
#                     interpolator = sitk.sitkBSpline
#                     default_pixel_value = np.min(sitk.GetArrayFromImage(image_sitk))
#                 else:
#                     interpolator = sitk.sitkNearestNeighbor
#                     default_pixel_value = 0
#
#                 transform = transform_infos.get_transform(inverse=True)
#
#                 resample_filter = sitk.ResampleImageFilter()
#                 resample_filter.SetTransform(transform)
#                 resample_filter.SetInterpolator(interpolator)
#                 resample_filter.SetDefaultPixelValue(default_pixel_value)
#                 resample_filter.SetSize(transform_infos.get_size(True))
#                 resample_filter.SetOutputOrigin(transform_infos.get_origin(True))
#                 resample_filter.SetOutputDirection(transform_infos.get_direction(True, False))
#                 resample_filter.SetOutputSpacing(transform_infos.get_spacing(True))
#
#                 image_sitk = resample_filter.Execute(image_sitk)
#
#         data.set_image(image_sitk)
#         data.transform_tape.recordings.clear()
#
#         return data
#
#     def get_recorded_elements(self,
#                               reverse: bool = False
#                               ) -> Tuple[TransformationInformation, ...]:
#         """Get the recorded :class:`TransformationInformation` entries on the tape.
#
#         Args:
#             reverse (bool): Indicates if the recordings should be returned in reverse order.
#
#         Returns:
#             Tuple[TransformationInformation, ...]: The recorded :class:`TransformationInformation` entries on the tape.
#         """
#         # pylint: disable=useless-super-delegation
#         return super().get_recorded_elements(reverse)


class TransformInfo:
    """A class to store information about a data transformation performed via a :class:`~pyradise.process.base.Filter`.
     This class is used in combination with a :class:`~pyradise.data.taping.TransformTape` instance to keep track
     of data transformations and to render invertibility feasible for invertible filters operations.

     Args:
         name (str): The name of the filter which performed the data transformation.
         params (Optional[FilterParameters]): The filter parameters which parameterize the data transformation.
         pre_transform_image_properties (ImageProperties): The image properties before the data transformation.
         post_transform_image_properties (ImageProperties): The image properties after the data transformation.
         filter_args (Optional[Dict[str, Any]]): The filter arguments passed via the constructor of the filter
          (default: None).
         additional_data (Optional[Dict[str, Any]]): Additional data which is required the data transformation or to
          inverse it (default: None).
         transform (Optional[sitk.Transform]): A SimpleITK transform which may be used for the data transformation
          (default: None).
    """

    def __init__(self,
                 name: str,
                 params: Optional[FilterParameters],
                 pre_transform_image_properties: ImageProperties,
                 post_transform_image_properties: ImageProperties,
                 filter_args: Optional[Dict[str, Any]] = None,
                 additional_data: Optional[Dict[str, Any]] = None,
                 transform: Optional[sitk.Transform] = None
                 ) -> None:
        super().__init__()

        self.name = name
        self.params = params
        self.pre_transform_image_properties: ImageProperties = pre_transform_image_properties
        self.post_transform_image_properties: ImageProperties = post_transform_image_properties
        self.filter_args: Dict[str, Any] = filter_args if filter_args is not None else dict()
        self.additional_data: Dict[str, Any] = additional_data if additional_data is not None else dict()
        self.transform: Optional[sitk.Transform] = transform

    def _get_subclasses(self, cls: type) -> Dict[str, type]:
        """Get all subclasses of the provided class.

        Args:
            cls (type): The class to get the subclasses of.

        Returns:
            Dict[str, type]: A dictionary containing the subclasses of the provided class.
        """
        subclasses = {}
        for subclass in cls.__subclasses__():
            subclasses.update({subclass.__name__: subclass})
            if subclass.__subclasses__():
                subclasses.update(self._get_subclasses(subclass))
        return subclasses

    def get_filter(self) -> Filter:
        """Get the :class:`~pyradise.process.base.Filter` instance which performed the data transformation.

        Returns:
            Filter: The filter used for the data transformation.
        """
        from pyradise.process import Filter
        subclasses = self._get_subclasses(Filter)
        return subclasses.get(self.name)(**self.filter_args)

    def get_params(self) -> FilterParameters:
        """Get the :class:`~pyradise.process.base.FilterParams` instance which was used to parameterize the
        data transformation.

        Returns:
            FilterParameters: The filter parameters used for the data transformation.
        """
        return self.params

    def get_image_properties(self, pre_transform: bool) -> ImageProperties:
        """Get the pre-transform or post-transform :class:`~pyradise.data.image.ImageProperties` instance.

        Args:
            pre_transform (bool): If True returns the pre-transform image properties, otherwise the post-transform
             image properties.

        Returns:
            ImageProperties: The pre-transform or post-transform image properties.
        """
        if pre_transform:
            return self.pre_transform_image_properties
        return self.post_transform_image_properties

    def add_data(self, key: str, value: Any) -> None:
        """Add additional data to the :class:`TransformInfo` instance.

        Note:
            If the provided key already exists, the value will be overwritten.

        Args:
            key (str): The key of the additional data.
            value (Any): The value of the additional data.

        Returns:
            None
        """
        self.additional_data[key] = value

    def get_data(self, key: str) -> Any:
        """Get additional data from the :class:`TransformInfo` instance by key.

        Args:
            key (str): The key of the additional data entry to get.

        Returns:
            Any: The value of the additional data entry. If the key is not existing :data:`None` is returned.
        """
        return self.additional_data.get(key, None)

    def get_transform(self, inverse: bool = False) -> sitk.Transform:
        """Get the :class:`SimpleITK.Transform` instance which was used to perform the data transformation.

        Args:
            inverse (bool): Indicates if the inverse transform should be returned (default: False).

        Returns:
            sitk.Transform: The transform used for the data transformation or the identity transform if origin and
            direction did not change during data transformation.
        """
        if self.transform is not None:
            if inverse:
                return self.transform.GetInverse()
            return self.transform

        # check if the image origin and direction have changed
        num_dims = len(self.pre_transform_image_properties.size)
        if self.pre_transform_image_properties.has_equal_origin_direction(self.post_transform_image_properties):
            transform = sitk.AffineTransform(num_dims)
            transform.SetIdentity()
            return transform

        else:
            transform = sitk.AffineTransform(num_dims)
            transform.SetIdentity()

            # compute the translation
            post_origin = self.post_transform_image_properties.origin
            pre_origin = self.pre_transform_image_properties.origin
            translation = list(np.array(post_origin) - np.array(pre_origin))

            # compute the rotation
            post_direction = np.array(self.post_transform_image_properties.direction).reshape(num_dims, num_dims)
            pre_direction = np.array(self.pre_transform_image_properties.direction).reshape(num_dims, num_dims)
            rotation = np.matmul(np.linalg.inv(pre_direction), post_direction)
            rotation = list(rotation.reshape(-1))

            # set the transform parameters
            transform.SetParameters(rotation + translation)

            # return the inverted or the original transform
            if inverse:
                transform = transform.GetInverse()
            return transform


class TransformTape(Tape):
    """A class to keep track of the :class:`~pyradise.data.taping.TransformInfo` instances such that they can be
    played back on appropriate data. This class provides the basic functionality to render invertibility and
    reproducibility feasible.
    """

    def __init__(self):
        super().__init__()

    def record(self, value: TransformInfo) -> None:
        """Record a :class:`~pyradise.data.taping.TransformInfo` instance on the tape.

        Args:
            value (TransformInfo): The :class:`~pyradise.data.taping.TransformInfo` instance to record.

        Returns:
            None
        """
        self.recordings.append(value)

    @staticmethod
    def playback(data: Image, **kwargs) -> Image:
        """Play back the recorded :class:`~pyradise.data.taping.TransformInfo` instances on the provided data.

        Args:
            data (Image): The data to play back the recorded :class:`~pyradise.data.taping.TransformInfo` instances on.
            **kwargs: Additional keyword arguments.

        Returns:
            Image: The :class:`~pyradise.data.image.Image` instance after the playback of the recorded
            :class:`~pyradise.data.taping.TransformInfo` instances.

        """
        from pyradise.data import Subject

        # create a temporary subject to store the image
        subject = Subject('temporary_playback_subject', data)

        # playback the transformations
        for transform_info in data.get_transform_tape().get_recorded_elements(reverse=True):
            filter_ = transform_info.get_filter()

            if not filter_.is_invertible():
                continue

            subject = filter_.execute_inverse(subject, transform_info)

        # set the new image data on the original image
        image = subject.get_images_by_type(type(data))[0]
        data.set_image_data(image.get_image_data())

        # clear the recordings after playback
        data.get_transform_tape().recordings.clear()

        return data
