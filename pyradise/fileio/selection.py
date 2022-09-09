from abc import (ABC, abstractmethod)
from typing import (Tuple, List, Sequence)

from pyradise.data import (
    Modality,
    Organ,
    Rater)
from .series_info import (
    SeriesInfo,
    IntensityFileSeriesInfo,
    SegmentationFileSeriesInfo,
    DicomSeriesInfo,
    DicomSeriesImageInfo,
    DicomSeriesRegistrationInfo,
    DicomSeriesRTSSInfo)

__all__ = ['SeriesInfoSelector', 'SeriesInfoSelectorPipeline', 'ModalityInfoSelector', 'OrganInfoSelector',
           'RaterInfoSelector', 'NoRegistrationInfoSelector']


class SeriesInfoSelector(ABC):
    """An abstract series info selector class."""

    @abstractmethod
    def execute(self, infos: Sequence[SeriesInfo]) -> Tuple[SeriesInfo]:
        raise NotImplementedError()


class SeriesInfoSelectorPipeline:

    def __init__(self, selectors: Sequence[SeriesInfoSelector]) -> None:
        self.selectors = selectors

    def execute(self, infos: Sequence[SeriesInfo]) -> Tuple[SeriesInfo]:
        for selector in self.selectors:
            infos = selector.execute(infos)
        return infos


class ModalityInfoSelector(SeriesInfoSelector):
    """A selector to remove all :class:`IntensityFileSeriesInfo` and :class:`DicomSeriesImageInfo` entries that do not
    contain one of the specified :class:`Modality` s."""

    def __init__(self, keep: Tuple[Modality]) -> None:
        super().__init__()

        assert keep, 'The modalities to keep must not be empty!'
        self.keep = keep

    # noinspection DuplicatedCode
    # pylint: disable=duplicate-code
    @staticmethod
    def _remove_unused_registration_infos(infos: Tuple[SeriesInfo]) -> Tuple[SeriesInfo]:
        """Remove all :class:`DicomSeriesRegistrationInfo` entries that are not used anymore.

        Args:
            infos (List[DicomSeriesInfo]): The :class:`DicomSeriesInfo` entries to analyze.
        """

        registration_infos = [entry for entry in infos if isinstance(entry, DicomSeriesRegistrationInfo)]
        image_infos = [entry for entry in infos if isinstance(entry, DicomSeriesImageInfo)]

        remove_indices = []
        for i, registration_info in enumerate(registration_infos):
            criteria = [entry.series_instance_uid == registration_info.referenced_series_instance_uid_transform
                        for entry in image_infos]

            if not any(criteria):
                remove_indices.append(i)

        for index in reversed(remove_indices):
            registration_infos.pop(index)

        keep = [entry for entry in infos if not isinstance(entry, DicomSeriesRegistrationInfo)]
        keep.extend(registration_infos)
        return tuple(keep)

    # noinspection DuplicatedCode
    # pylint: disable=duplicate-code
    def execute(self, infos: Sequence[SeriesInfo]) -> Tuple[SeriesInfo]:
        """Remove all :class:`IntensityFileSeriesInfo` and :class:`DicomSeriesImageInfo` entries that do not contain
        one of the specified :class:`Modality` s.

        Args:
            infos (Sequence[SeriesInfo]): The :class:`SeriesInfo` entries to analyze.

        Returns:
            Tuple[SeriesInfo]: The filtered :class:`SeriesInfo` entries.
        """
        assert infos, 'The series infos must not be empty!'

        selected: List[SeriesInfo] = []
        for info in infos:
            if isinstance(info, IntensityFileSeriesInfo):
                if info.modality in self.keep:
                    selected.append(info)

            elif isinstance(info, DicomSeriesImageInfo):
                if info.modality in self.keep:
                    selected.append(info)
            else:
                selected.append(info)

        return self._remove_unused_registration_infos(tuple(selected))


class OrganInfoSelector(SeriesInfoSelector):
    """A selector to remove all :class:`SegmentationFileSeriesInfo` entries that do not contain one of the specified
    :class:`Organ` s."""

    def __init__(self, keep: Tuple[Organ]) -> None:
        super().__init__()

        assert keep, 'The organs to keep must not be empty!'
        self.keep = keep

    def execute(self, infos: Tuple[SeriesInfo]) -> Tuple[SeriesInfo]:
        """Remove all :class:`SegmentationFileSeriesInfo` entries that do not contain one of the specified
        :class:`Organ` s.

        Notes:
            :class:`DicomSeriesRTSSInfo` entries will NOT be analyzed if they contain the specified organs because the
            data is not completely loaded yet!

        Args:
            infos (Tuple[SeriesInfo]): The :class:`SeriesInfo` entries to analyze.

        Returns:
            Tuple[SeriesInfo]: The filtered :class:`SeriesInfo` entries.
        """
        assert infos, 'The series infos must not be empty!'

        selected: List[SeriesInfo] = []
        for info in infos:
            if isinstance(info, SegmentationFileSeriesInfo):
                if info.organ in self.keep:
                    selected.append(info)
            else:
                selected.append(info)

        return tuple(selected)


class RaterInfoSelector(SeriesInfoSelector):
    """A selector to remove all :class:`SegmentationFileSeriesInfo` and :class:`DicomSeriesRTSSInfo` entries that do
    not contain one of the specified :class:`Rater` s.
    """

    def __init__(self, keep: Tuple[Rater]) -> None:
        super().__init__()

        assert keep, 'The raters to keep must not be empty!'
        self.keep = keep

    # noinspection DuplicatedCode
    # pylint: disable=duplicate-code
    def execute(self, infos: Tuple[SeriesInfo]) -> Tuple[SeriesInfo]:
        """Remove all :class:`SegmentationFileSeriesInfo` and :class:`DicomSeriesRTSSInfo` entries that do not contain
        one of the specified :class:`Rater` s.

        Args:
            infos (Tuple[SeriesInfo]): The :class:`SeriesInfo` entries to analyze.

        Returns:
            Tuple[SeriesInfo]: The filtered :class:`SeriesInfo` entries.
        """
        assert infos, 'The series infos must not be empty!'

        selected: List[SeriesInfo] = []
        for info in infos:
            if isinstance(info, SegmentationFileSeriesInfo):
                if info.rater in self.keep:
                    selected.append(info)

            elif isinstance(info, DicomSeriesRTSSInfo):
                if info.rater in self.keep:
                    selected.append(info)
            else:
                selected.append(info)

        return tuple(selected)


class NoRegistrationInfoSelector(SeriesInfoSelector):
    """A selector to exclude all :class:`DicomSeriesRegistrationInfo` from the provided :class:`SeriesInfo`."""

    def execute(self, infos: Tuple[SeriesInfo]) -> Tuple[SeriesInfo]:
        """Returns all :class:`SeriesInfo` entries except the :class:`DicomSeriesRegistrationInfo` entries.

        Args:
            infos (Tuple[SeriesInfo]): The :class:`SeriesInfo` entries to analyze.

        Returns:
            Tuple[SeriesInfo]: The filtered :class:`SeriesInfo` entries.
        """
        assert infos, 'The series infos must not be empty!'

        selected: List[SeriesInfo] = []
        for info in infos:
            if not isinstance(info, DicomSeriesRegistrationInfo):
                selected.append(info)

        return tuple(selected)


class NoRTSSInfoSelector(SeriesInfoSelector):
    """A selector to exclude all :class:`DicomSeriesRTSSInfo` from the provided :class:`SeriesInfo`."""

    def execute(self, infos: Tuple[SeriesInfo]) -> Tuple[SeriesInfo]:
        """Returns all :class:`SeriesInfo` entries except the :class:`DicomSeriesRTSSInfo` entries.

        Args:
            infos (Tuple[SeriesInfo]): The :class:`SeriesInfo` entries to analyze.

        Returns:
            Tuple[SeriesInfo]: The filtered :class:`SeriesInfo` entries.
        """
        assert infos, 'The series infos must not be empty!'

        selected: List[SeriesInfo] = []
        for info in infos:
            if not isinstance(info, DicomSeriesRTSSInfo):
                selected.append(info)

        return tuple(selected)
