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


# pylint: disable=too-few-public-methods
class FilterParameters(ABC):
    """Represents abstract filter parameters."""


class Filter(ABC):
    """Abstract base class for processing a subject."""

    def __init__(self):
        super().__init__()
        self.verbose = False

    def set_verbose(self, verbose: bool) -> None:
        """Sets the verbose state.

        Args:
            verbose (bool): If true, the filter outputs information to the console.

        Returns:
            None
        """
        self.verbose = verbose

    @abstractmethod
    def execute(self,
                subject: Subject,
                params: Optional[FilterParameters]
                ) -> Subject:
        """Execution of the filter.

        Args:
            subject (Subject): The subject to be filtered.
            params (Optional[FilterParameters]): The filter parameters if required.

        Returns:
            Subject: The processed subject.
        """
        raise NotImplementedError()


class LoopEntryFilter(Filter):
    """A filter class feasible to process data in a loop."""

    @staticmethod
    def loop_entries(data: np.ndarray,
                     params: Any,
                     filter_fn: Callable[[np.ndarray, Any], np.ndarray],
                     loop_axis: Optional[int]
                     ) -> np.ndarray:
        """Applies the function fn by looping over the data using the parameters.

        Args:
            data (np.ndarray): The data to be filtered by the filtering function.
            params (Any): The parameters for the filtering function.
            filter_fn (Callable[[np.ndarray, Any], np.ndarray]): The filtering function.
            loop_axis (Optional[int]): The axis to loop over. If None the whole volume is taken.

        Returns:
            np.ndarray: The filtered data.
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
        """Executes the filter procedure.

        Args:
            subject (Subject): The subject to filter.
            params (Optional[FilterParameters]): The filter parameters.

        Returns:
            Subject: The filtered subject.
        """
        raise NotImplementedError()


class FilterPipeline:
    """Filter pipeline class to construct a pipeline of multiple sequential filters."""

    def __init__(self,
                 filters: Optional[Tuple[Filter, ...]] = None,
                 params: Optional[Tuple[FilterParameters, ...]] = None
                 ) -> None:
        """Constructs a filter pipeline from multiple filters and its parameters.

        Args:
            filters (Optional[Tuple[Filter, ...]]): The filters of the pipeline (default=None).
            params (Optional[Tuple[FilterParameters, ...]]): The parameters for the filters in the pipeline
             (default=None).
        """
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
        """Sets the verbose state for all filter.

        Args:
            verbose (bool): If true the filters print information to the console.

        Returns:
            None
        """
        for filter_ in self.filters:
            filter_.set_verbose(verbose)

    def add_filter(self,
                   filter_: Filter,
                   params: Optional[FilterParameters] = None
                   ) -> None:
        """Adds a filter and its parameters to the pipeline.

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
        """Sets the parameters for a certain filter index.

        Args:
            params (FilterParameters): The filter parameters.
            filter_index (int): The index of the filter to add the parameters.

        Returns:
            None
        """
        if filter_index == -1:
            filter_idx = len(self.filters) - 1
        else:
            filter_idx = filter_index

        self.params[filter_idx] = params

    def add_logger(self, logger: logging.Logger) -> None:
        """Adds a logger to the filter pipeline.

        Args:
            logger (logging.Logger): The logger to use with the pipeline.

        Returns:
            None
        """
        self.logger = logger

    def execute(self, subject: Subject) -> Subject:
        """Executes the filter pipeline.

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
