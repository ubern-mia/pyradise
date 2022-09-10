import logging
from abc import (
    ABC,
    abstractmethod)
from typing import (
    Any,
    Callable,
    Optional,
    Union,
    Tuple,
    List)

import numpy as np

from pyradise.data import Subject

__all__ = ['Filter', 'FilterParameters', 'FilterPipeline', 'LoopEntryFilter']


# pylint: disable=too-few-public-methods
class FilterParameters(ABC):
    """An abstract filter parameter class which provides holds the parameters used for the configuration of a certain
    filter."""


class Filter(ABC):
    """An abstract filter class which is used to process a subject and its content."""

    def __init__(self):
        super().__init__()
        self.verbose = False

    def set_verbose(self, verbose: bool) -> None:
        """Set the verbose state.

        Args:
            verbose (bool): If True, the filter outputs information to the console, otherwise not.

        Returns:
            None
        """
        self.verbose = verbose

    @abstractmethod
    def execute(self,
                subject: Subject,
                params: Optional[FilterParameters]
                ) -> Subject:
        """Execute the filter.

        Args:
            subject (Subject): The subject to be processed.
            params (Optional[FilterParameters]): The filter parameters, if required.

        Returns:
            Subject: The processed subject.
        """
        raise NotImplementedError()


class LoopEntryFilter(Filter):
    """A filter class feasible to process data in a loop over a defined ``loop_axis``."""

    @staticmethod
    def loop_entries(data: np.ndarray,
                     params: Any,
                     filter_fn: Callable[[np.ndarray, Any], np.ndarray],
                     loop_axis: Optional[int]
                     ) -> np.ndarray:
        """Apply the function :func:`fn` by looping over the data using the provided parameters (i.e. ``params``).

        Args:
            data (np.ndarray): The data to be processed.
            params (Any): The parameters for the filter function.
            filter_fn (Callable[[np.ndarray, Any], np.ndarray]): The filter function.
            loop_axis (Optional[int]): The axis to loop over. If None the whole volume is taken, otherwise the
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
    def execute(self,
                subject: Subject,
                params: Optional[FilterParameters]
                ) -> Subject:
        """Execute the filter.

        Args:
            subject (Subject): The subject to be processed.
            params (Optional[FilterParameters]): The filter parameters.

        Returns:
            Subject: The processed subject.
        """
        raise NotImplementedError()


class FilterPipeline:
    """A filter pipeline class which applies multiple sequential filters to the same subject.

    Args:
        filters (Optional[Tuple[Filter, ...]]): The filters of the pipeline (default: None).
        params (Optional[Tuple[FilterParameters, ...]]): The parameters for the filters in the pipeline.
    """

    def __init__(self,
                 filters: Optional[Tuple[Filter, ...]] = None,
                 params: Optional[Tuple[FilterParameters, ...]] = None
                 ) -> None:
        super().__init__()

        self.filters: List[Filter, ...] = []
        self.params: List[FilterParameters, ...] = []

        if filters:
            if not params:
                params = [None] * len(filters)

            else:
                assert len(params) == len(filters), f'The number of filters ({len(filters)}) must be equal ' \
                                                    f'to the number of filter parameters ({len(params)})!'

            for filter_, param in zip(filters, params):
                self.add_filter(filter_, param)

        self.logger: Optional[logging.Logger] = None

    def set_verbose_all(self, verbose: bool) -> None:
        """Set the verbose state for all filter.

        Args:
            verbose (bool): If True the filters print information to the console.

        Returns:
            None
        """
        for filter_ in self.filters:
            filter_.set_verbose(verbose)

    def add_filter(self,
                   filter_: Filter,
                   params: Optional[FilterParameters] = None
                   ) -> None:
        """Add a filter and its parameters to the pipeline.

        Args:
            filter_ (Filter): The filter to add.
            params (Optional[FilterParameters]): The filter parameters if necessary.

        Returns:
            None
        """
        self.filters.append(filter_)
        self.params.append(params)

    def set_param(self,
                  params: FilterParameters,
                  filter_index: int
                  ) -> None:
        """Set the parameters for a specific filter at index ``filter_index``.

        Args:
            params (FilterParameters): The filter parameters.
            filter_index (int): The index of the filter to add the parameters to.

        Returns:
            None
        """
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

    def execute(self, subject: Subject) -> Subject:
        """Execute the filter pipeline.

        Args:
            subject (Subject): The subject to be processed by the pipeline.

        Returns:
            Subject: The processed subject.
        """
        assert len(self.filters) == len(self.params), f'The filter pipeline can not be executed due to unequal ' \
                                                      f'numbers of filters ({len(self.filters)}) and ' \
                                                      f'parameters ({len(self.params)})!'

        for filter_, param in zip(self.filters, self.params):
            if self.logger:
                self.logger.info(f'{subject.get_name()}: Pipeline executing {filter_.__class__.__name__}...')

            subject = filter_.execute(subject, param)

        return subject
