import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import itk
import numpy as np
import SimpleITK as sitk

from pyradise.data import (
    Image,
    ImageProperties,
    IntensityImage,
    SegmentationImage,
    Subject,
    TransformInfo,
)

__all__ = [
    "FilterParams",
    "Filter",
    "LoopEntryFilterParams",
    "LoopEntryFilter",
    "FilterPipeline",
]


# pylint: disable=too-few-public-methods
class FilterParams(ABC):
    """An abstract filter parameter class which provides the parameters used for the configuration of a certain
    filter. The derived subclasses can hold any set of parameters and is provided to the corresponding
    :class:`~pyradise.process.base.Filter` via the :meth:`~pyradise.process.base.Filter.execute` method.
    The :class:`~pyradise.process.base.FilterParams` subclasses may incorporate also methods to calculate
    certain parameter values based on the given set of parameters.

    The instances of :class:`~pyradise.process.base.FilterParams` subclasses are stored inside a
    :class:`~pyradise.data.taping.TransformInfo` instance to keep track of the parameters used during the execution
    of a certain :class:`~pyradise.process.base.Filter` such that invertibility can be guaranteed for
    :class:`~pyradise.process.base.Filter` s feasible to be inverted. However, for the reason of reproducibility the
    :class:`~pyradise.process.base.FilterParams` instances should be tracked always.

    Example:

        An example of a :class:`~pyradise.process.base.FilterParams` implementation for an intensity rescaling filter:

        >>> from pyradise.process import FilterParams
        >>>
        >>>
        >>> class ExampleRescaleFilterParams(FilterParams):
        >>>
        >>>     def __init__(self, min_out: float, max_out: float) -> None:
        >>>         super().__init__()
        >>>
        >>>         # reverse the values if min_out > max_out
        >>>         if min_out > max_out:
        >>>             min_out, max_out = max_out, min_out
        >>>
        >>>         # the minimum and maximum output intensity values
        >>>         self.min_out = min_out
        >>>         self.max_out = max_out

    """

    @abstractmethod
    def __init__(self) -> None:
        pass


class Filter(ABC):
    """An abstract filter base class which is used to process a subject and its content. In PyRaDiSe a
    :class:`~pyradise.process.base.Filter` is the main data processing object which is feasible to modify the structure
    and content of a :class:`~pyradise.data.subject.Subject`, the content of the subject-associated
    :class:`~pyradise.data.image.Image` and other subject-associated data. Thus, filters can be used for
    pre-processing, DL-model inference, and post-processing.

    The implemented filter design provides a standardized interface such that filters can be chained together in a
    :class:`~pyradise.process.base.FilterPipeline` to form a processing pipeline. Furthermore, the extensible
    implementation renders the tracking of content changes feasible for the purpose of reproducibility and
    invertibility on invertible :class:`~pyradise.process.base.Filter`.

    The :mod:`~pyradise.process` package provides a set of implemented :class:`~pyradise.process.base.Filter` s and
    associated :class:`~pyradise.process.base.FilterParams`. However, the user may implement its own
    :class:`~pyradise.process.base.Filter` s depending on the task specific needs. We recommend to share the
    user-implemented :class:`~pyradise.process.base.Filter` s with the community via GitHub or by generating pull
    requests to the `PyRaDiSe GitHub repository <https://github.com/ubern-mia/pyradise>`_. We thank all contributors
    in advance for sharing their filter implementations!

    In order to implement a new :class:`~pyradise.process.base.Filter` the following steps are required:

    1. Always derive from the :class:`~pyradise.process.base.Filter` class.

    2. Implement the :meth:`~pyradise.process.base.Filter.execute` method and possible subsequent methods
       which are used to process the :class:`~pyradise.data.subject.Subject`.

    3. Make sure that your implementation tracks the changes and assign it to the
       :class:`~pyradise.data.taping.TransformTape` instance of the corresponding :class:`~pyradise.data.image.Image`
       instance.

    4. Implement the :meth:`~pyradise.process.base.Filter.execute_inverse` and
       :meth:`~pyradise.process.base.Filter.is_invertible` methods if the filter is invertible. Please note that
       the implementation can access all information which was previously recorded on the corresponding
       :class:`~pyradise.data.taping.TransformTape` instance.

    5. Test the new :class:`~pyradise.process.base.Filter` implementation and make sure that it works as expected.


    Example:

        Example implementation of an intensity rescaling filter:

        >>> import SimpleITK as sitk
        >>> import numpy as np
        >>>
        >>> from pyradise.process import Filter, FilterParams
        >>> from pyradise.data import Subject, IntensityImage, TransformInfo
        >>>
        >>>
        >>> class ExampleRescaleFilterParams(FilterParams):
        >>>
        >>>     def __init__(self, min_out: float, max_out: float) -> None:
        >>>         super().__init__()
        >>>
        >>>         # reverse the values if min_out > max_out
        >>>         if min_out > max_out:
        >>>             min_out, max_out = max_out, min_out
        >>>
        >>>         # the minimum and maximum output intensity values
        >>>         self.min_out = min_out
        >>>         self.max_out = max_out
        >>>
        >>>
        >>> class ExampleRescaleFilter(Filter):
        >>>
        >>>     @staticmethod
        >>>     def is_invertible() -> bool:
        >>>         # return True because the filter is invertible
        >>>         return True
        >>>
        >>>     def execute(self,
        >>>                 subject: Subject,
        >>>                 params: ExampleRescaleFilterParams
        >>>                 ) -> Subject:
        >>>         # loop through the images
        >>>         for image in subject.get_images():
        >>>
        >>>             # exclude segmentation images
        >>>             if not isinstance(image, IntensityImage):
        >>>                 continue
        >>>
        >>>             # retrieve the image data
        >>>             original_image_sitk = image.get_image_data()
        >>>
        >>>             # rescale the intensity
        >>>             new_image_sitk = sitk.RescaleIntensity(original_image_sitk,
        >>>                                                    params.min_out,
        >>>                                                    params.max_out)
        >>>
        >>>             # update the image data
        >>>             image.set_image_data(new_image_sitk)
        >>>
        >>>             # track the necessary information
        >>>             original_image_np = sitk.GetArrayFromImage(original_image_sitk)
        >>>             self.tracking_data['min_'] = float(np.min(original_image_np))
        >>>             self.tracking_data['max_'] = float(np.max(original_image_np))
        >>>             self._register_tracked_data(image, original_image_sitk,
        >>>                                         new_image_sitk, params)
        >>>
        >>>         return subject
        >>>
        >>>     def execute_inverse(self,
        >>>                         subject: Subject,
        >>>                         transform_info: TransformInfo,
        >>>                         target_image: Optional[Union[SegmentationImage, IntensityImage]] = None
        >>>                         ) -> Subject:
        >>>         # loop through the images
        >>>         for image in subject.get_images():
        >>>
        >>>             # exclude segmentation images
        >>>             if not isinstance(image, IntensityImage):
        >>>                 continue
        >>>
        >>>             # retrieve the tracked data
        >>>             min_intensity = transform_info.get_data('min_')
        >>>             max_intensity = transform_info.get_data('max_')
        >>>
        >>>             # undo the intensity rescaling
        >>>             original_image_sitk = image.get_image_data()
        >>>             new_image_sitk = sitk.RescaleIntensity(original_image_sitk,
        >>>                                                    min_intensity,
        >>>                                                    max_intensity)
        >>>
        >>>             # update the image data
        >>>             image.set_image_data(new_image_sitk)
        >>>
        >>>             # there is no need to track information because
        >>>             # the operation is inverted
        >>>
        >>>         return subject

    Args:
        warning_on_non_invertible (bool): If True, a warning is printed to the console if a filter is called to
         execute the invertible process but is not invertible (default: False).

    """

    def __init__(self, warning_on_non_invertible: bool = False) -> None:
        super().__init__()

        self.warn_on_non_invertible = warning_on_non_invertible

        self.verbose = False

        # register here all filter arguments such that the filter can be reconstructed
        self.filter_args: Dict[str, Any] = {}

        # data to be tracked for the inverse transformation
        self.tracking_data: Dict[str, Any] = {}

    @staticmethod
    @abstractmethod
    def is_invertible() -> bool:
        """Check if the filter is invertible.

        Returns:
            bool: True if the filter is invertible, otherwise False.
        """
        raise NotImplementedError()

    def set_verbose(self, verbose: bool) -> None:
        """Set the verbose state.

        Args:
            verbose (bool): If True, the filter outputs information to the console, otherwise not.

        Returns:
            None
        """
        self.verbose = verbose

    def set_warning_on_non_invertible(self, warn: bool) -> None:
        """Set the warning state.

        Args:
            warn (bool): If True, the filter outputs a warning if the filter is called and is not invertible,
             otherwise not.

        Returns:
            None
        """
        self.warn_on_non_invertible = warn

    def _register_tracked_data(
        self,
        image: Image,
        pre_transform_image: Union[sitk.Image, itk.Image],
        post_transform_image: Union[sitk.Image, itk.Image],
        params: Optional[FilterParams],
        transform: Optional[sitk.Transform] = None,
    ) -> None:
        """Create the :class:`~pyradise.data.taping.TransformInfo` instance which is used to store the information
        about the performed transformation.

        Args:
            pre_transform_image (Union[sitk.Image, itk.Image]): The image before the transformation.
            post_transform_image (Union[sitk.Image, itk.Image]): The image after the transformation.
            params (Optional[FilterParams]): The filter parameters used for the transformation.
            transform (Optional[sitk.Transform]): The transformation which was applied to the image (default: None).
        """
        filter_args_ = self.filter_args if self.filter_args is not None else {}
        additional_data_ = self.tracking_data if self.tracking_data is not None else {}

        pre_image_props = ImageProperties(pre_transform_image)
        post_image_props = ImageProperties(post_transform_image)

        transform_info = TransformInfo(
            self.__class__.__name__,
            params,
            pre_image_props,
            post_image_props,
            deepcopy(filter_args_),
            deepcopy(additional_data_),
            deepcopy(transform),
        )
        image.add_transform_info(transform_info)

        self.tracking_data.clear()

    @abstractmethod
    def execute(self, subject: Subject, params: Optional[FilterParams]) -> Subject:
        """Execute the filter on the provided :class:`~pyradise.data.subject.Subject` instance.

        Note:
            For the ease of use, the filter provides a private :meth:`_create_transform_info` method which can be used
            to create the :class:`~pyradise.data.taping.TransformInfo` instances.

        Important:
            The filter is responsible to record the transformations applied to each image such that the invertibility
            is ensured. Even if the filter is not invertible, the transformations should be recorded such that the
            order of filter applications can be reconstructed from the transform tapes of the images. In case the
            filter is not invertible, the :meth:`~pyradise.process.base.Filter.is_invertible` must return ``False``.

        Args:
            subject (Subject): The subject to be processed.
            params (Optional[FilterParams]): The filter parameters, if required.

        Returns:
            Subject: The processed subject.
        """
        raise NotImplementedError()

    @abstractmethod
    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Execute the filter inversely if possible. Typically, this method gets a temporary subject which contains
        a single image because the recording of the transformations is image dependent and inappropriate inverse
        transformations would be applied to the other images. However, this method can also be applied to a whole
        subject to apply the inverse transformations to all images. This approach provides a more flexible way to
        handle invertibility of transformations.

        Important:
            If the filter is not invertible, the subject must be returned unchanged and the
            :meth:`~pyradise.process.base.Filter.is_invertible` must return ``False``.

        Args:
            subject (Subject): The subject to be processed.
            transform_info (TransformInfo): The :class:`~pyradise.data.taping.TransformInfo` instance.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The processed subject.
        """
        raise NotImplementedError()


class LoopEntryFilterParams(FilterParams):
    """An abstract filter parameter class which provides in addition to :class:`~pyradise.process.base.FilterParams`
    the ``loop_axis`` parameter which is used to specify the axis to loop over in the
    :class:`~pyradise.process.base.LoopEntryFilter`.

    Args:
        loop_axis (Optional[int]): The axis along which the data transformation is performed. If ``None``, the
         transformation is performed on the whole image at once. If a value is given, the transformation is performed
         by looping over the corresponding image dimension.
    """

    def __init__(self, loop_axis: Optional[int] = None) -> None:
        super().__init__()

        if loop_axis is not None:
            if loop_axis < 0:
                raise ValueError("The loop axis must be a non-negative integer.")
            if loop_axis > 2:
                raise ValueError("The loop axis must be smaller than 3, PyRaDiSe only supports 2D and 3D images.")

        self.loop_axis: Optional[int] = loop_axis


class LoopEntryFilter(Filter):
    """An abstract filter base class which is feasible to process images slice-wise in a loop over a defined
    ``loop_axis``. The ``loop_axis`` must be specified in the appropriate
    :class:`~pyradise.process.base.FilterParams` instance and if it takes a value of ``None``, the filter is
    executed on the whole image extent at once.

    Reference:
        The implementation of this class is inspired by an earlier version of the `pymia package
        <https://pymia.readthedocs.io/en/latest>`_.
    """

    @staticmethod
    @abstractmethod
    def is_invertible() -> bool:
        """Check if the filter is invertible.

        Returns:
            bool: True if the filter is invertible, otherwise False.
        """
        raise NotImplementedError()

    @staticmethod
    def loop_entries(
        data: np.ndarray, params: Any, filter_fn: Callable[[np.ndarray, Any], np.ndarray], loop_axis: Optional[int]
    ) -> np.ndarray:
        """Apply the function :meth:`filter_fn` by looping over the image using the provided parameters
        (i.e. ``params``).

        Args:
            data (np.ndarray): The data to be processed.
            params (Any): The parameters for the filter function.
            filter_fn (Callable[[np.ndarray, Any], np.ndarray]): The filter function.
            loop_axis (Optional[int]): The axis to loop over. If ``None`` the whole image is taken, otherwise the
             respective dimension.

        Returns:
            np.ndarray: The processed data.
        """
        if loop_axis is None:
            new_data = filter_fn(data, params)

        else:
            new_data = np.zeros_like(data)

            slicing: List[Union[slice, int]] = [slice(None) for _ in range(data.ndim)]
            for i in range(data.shape[loop_axis]):
                slicing[loop_axis] = i
                new_data[tuple(slicing)] = filter_fn(data[tuple(slicing)], params)

        return new_data

    @abstractmethod
    def execute(self, subject: Subject, params: Optional[LoopEntryFilterParams]) -> Subject:
        """Execute the filter on the provided :class:`~pyradise.data.subject.Subject` instance.

        Note:
            For the ease of use, the filter provides a private :meth:`_create_transform_info` method which can be used
            to create the :class:`~pyradise.data.taping.TransformInfo` instances.

        Important:
            The filter is responsible to record the transformations applied to each image such that the invertibility
            is ensured. Even if the filter is not invertible, the transformations should be recorded such that the
            order of filter applications can be reconstructed from the transform tapes of the images. In case the
            filter is not invertible, the :meth:`~pyradise.process.base.Filter.is_invertible` must return ``False``.

        Args:
            subject (Subject): The subject to be processed.
            params (Optional[LoopEntryFilterParams]): The filter parameters, if required.

        Returns:
            Subject: The processed subject.
        """
        raise NotImplementedError()

    @abstractmethod
    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Execute the filter inversely if possible. Typically, this method gets a temporary subject which contains
        a single image because the recording of the transformations is image dependent and inappropriate inverse
        transformations would be applied to the other images. However, this method can also be applied to a whole
        subject to apply the inverse transformations to all images. This approach provides a more flexible way to
        handle invertibility of transformations.

        Important:
            If the filter is not invertible, the subject must be returned unchanged and the
            :meth:`~pyradise.process.base.Filter.is_invertible` must return ``False``.

        Args:
            subject (Subject): The subject to be processed.
            transform_info (TransformInfo): The :class:`~pyradise.data.taping.TransformInfo` instance.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The processed subject.
        """
        raise NotImplementedError()


class FilterPipeline:
    """A filter pipeline class which can combine multiple :class:`~pyradise.process.base.Filter` instances into one
    pipeline of sequential filter. This reduces the amount of boilerplate code for the user and provides a nice way to
    chain multiple filters together.

    Args:
        filters (Optional[Tuple[Filter, ...]]): The filters of the pipeline (default: None).
        params (Optional[Tuple[FilterParams, ...]]): The parameters for the filters in the pipeline.
        warning_on_non_invertible (bool): If True, a warning is printed to the console if a filter is called to
         execute the invertible process but is not invertible (default: False).
    """

    def __init__(
        self,
        filters: Optional[Tuple[Filter, ...]] = None,
        params: Optional[Tuple[FilterParams, ...]] = None,
        warning_on_non_invertible: bool = False,
    ) -> None:
        super().__init__()

        self.filters: List[Filter, ...] = []
        self.params: List[FilterParams, ...] = []
        self.warn_on_non_invertible = warning_on_non_invertible

        if filters:
            if not params:
                params = [None] * len(filters)

            else:
                if len(params) != len(filters):
                    raise ValueError(
                        f"The number of filters ({len(filters)}) must be equal "
                        f"to the number of filter parameters ({len(params)})!"
                    )

            for filter_, param in zip(filters, params):
                self.add_filter(filter_, param)

        self.logger: Optional[logging.Logger] = None

    def set_verbose_all(self, verbose: bool) -> None:
        """Set the verbose state for all :class:`~pyradise.process.base.Filter` instances.

        Args:
            verbose (bool): If True the filters print information to the console, otherwise not.

        Returns:
            None
        """
        for filter_ in self.filters:
            filter_.set_verbose(verbose)

    def add_filter(self, filter_: Filter, params: Optional[FilterParams] = None) -> None:
        """Add a :class:`~pyradise.process.base.Filter` instance and its corresponding
        :class:`~pyradise.process.base.FilterParams` to the pipeline.

        Args:
            filter_ (Filter): The :class:`~pyradise.process.base.Filter` instance to add.
            params (Optional[FilterParams]): The :class:`~pyradise.process.base.FilterParams` instance to add,
             if necessary (default: None).

        Returns:
            None
        """
        self.filters.append(filter_)
        self.params.append(params)

    def set_param(self, params: FilterParams, filter_index: int) -> None:
        """Set the :class:`~pyradise.process.base.FilterParams` for a specific
        :class:`~pyradise.process.base.Filter` instance at index ``filter_index``.

        Args:
            params (FilterParams): The :class:`~pyradise.process.base.FilterParams` instance.
            filter_index (int): The index of the :class:`~pyradise.process.base.Filter` to add the parameters to.

        Returns:
            None
        """
        if filter_index >= len(self.filters):
            raise ValueError(
                f"The filter index ({filter_index}) must be smaller than the number of filters ({len(self.filters)})!"
            )

        if filter_index == -1:
            filter_idx = len(self.filters) - 1
        else:
            filter_idx = filter_index

        self.params[filter_idx] = params

    def add_logger(self, logger: logging.Logger) -> None:
        """Add a logger to the filter pipeline.

        Args:
            logger (logging.Logger): The logger to use with the pipeline.

        Returns:
            None
        """
        self.logger = logger

    def execute_iteratively(self, subject: Subject) -> GeneratorExit(Subject, str):
        """Execute iteratively in the filter pipeline on the provided :class:`~pyradise.data.subject.Subject` instance.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed by the pipeline.

        Returns:
            tuple: (Subject: The currently processed Subject iteration, String: Currently executed filter name)
        """
        if len(self.filters) != len(self.params):
            raise ValueError(
                f"The filter pipeline can not be executed due to unequal "
                f"numbers of filters ({len(self.filters)}) and "
                f"parameters ({len(self.params)})!"
            )

        for filter_, param in zip(self.filters, self.params):
            if self.logger:
                self.logger.info(f"{subject.get_name()}: Pipeline executing {filter_.__class__.__name__}...")

            # set the warning on and off
            if self.warn_on_non_invertible:
                filter_.set_warning_on_non_invertible(True)
            else:
                filter_.set_warning_on_non_invertible(False)

            subject = filter_.execute(subject, param)
            yield subject, filter_.__class__.__name__

    def execute(self, subject: Subject) -> Subject:
        """Execute the filter pipeline on the provided :class:`~pyradise.data.subject.Subject` instance.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be processed by the pipeline.

        Returns:
            Subject: The processed subject.
        """
        *_, (subject, _) = self.execute_iteratively(subject)  # iterate over the generator and get the last subject
        return subject
