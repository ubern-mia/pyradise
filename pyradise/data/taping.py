from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, TypeVar

import numpy as np
import SimpleITK as sitk

__all__ = ["Tape", "TransformTape", "TransformInfo"]

# pylint: disable=no-member

# Forward declaration of image types
Image = TypeVar("Image")
IntensityImage = TypeVar("IntensityImage")
SegmentationImage = TypeVar("SegmentationImage")
Filter = TypeVar("Filter")
FilterParameters = TypeVar("FilterParameters")
ImageProperties = TypeVar("ImageProperties")
Subject = TypeVar("Subject")


class Tape(ABC):
    """An abstract class for a tape which records defined elements and can replay them upon request."""

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

    def get_recorded_elements(self, reverse: bool = False) -> Tuple[Any, ...]:
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

    def __init__(
        self,
        name: str,
        params: Optional[FilterParameters],
        pre_transform_image_properties: ImageProperties,
        post_transform_image_properties: ImageProperties,
        filter_args: Optional[Dict[str, Any]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        transform: Optional[sitk.Transform] = None,
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

    def get_recorded_elements(self, reverse: bool = False) -> Tuple[TransformInfo, ...]:
        """Get the recorded :class:`~pyradise.data.taping.TransformInfo` instances.

        Args:
            reverse (bool): Indicates if the recorded elements should be returned in reverse order (default: False).

        Returns:
            Tuple[TransformInfo, ...]: The recorded :class:`~pyradise.data.taping.TransformInfo` instances.
        """
        return super().get_recorded_elements(reverse)

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
        subject = Subject("temporary_playback_subject", data)

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
