from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple, Union

from pyradise.data import (Annotator, Modality, Organ, seq_to_annotators,
                           seq_to_modalities, seq_to_organs)

from .series_info import (DicomSeriesImageInfo, DicomSeriesInfo,
                          DicomSeriesRegistrationInfo, DicomSeriesRTSSInfo,
                          IntensityFileSeriesInfo, SegmentationFileSeriesInfo,
                          SeriesInfo)

__all__ = [
    "SeriesInfoSelector",
    "SeriesInfoSelectorPipeline",
    "ModalityInfoSelector",
    "OrganInfoSelector",
    "AnnotatorInfoSelector",
    "NoRegistrationInfoSelector",
    "NoRTSSInfoSelector",
]


class SeriesInfoSelector(ABC):
    """An abstract base class for all :class:`SeriesInfoSelector` classes. A selector is used to select a subset of
    :class:`~pyradise.fileio.series_info.SeriesInfo` entries from a list of
    :class:`~pyradise.fileio.series_info.SeriesInfo` entries such that unused entries will be excluded from the loading
    and probable conversion procedures. The aim of using a selector is to improve speed and reduce memory usage while
    allowing the input directory to contain unused data.
    """

    @abstractmethod
    def execute(self, infos: Sequence[SeriesInfo]) -> Tuple[SeriesInfo, ...]:
        """Perform the selection procedure such that the appropriate :class:`~pyradise.fileio.series_info.SeriesInfo`
        entries are kept.

        Args:
            infos (Sequence[SeriesInfo]): The :class:`~pyradise.fileio.series_info.SeriesInfo` entries to select from.

        Returns:
            Tuple[SeriesInfo, ...]: The selected :class:`~pyradise.fileio.series_info.SeriesInfo` entries.
        """
        raise NotImplementedError()


class SeriesInfoSelectorPipeline:
    """A class for constructing :class:`SeriesInfoSelector` pipelines. A pipeline is a sequence of
    :class:`SeriesInfoSelector` s that are executed sequentially in a given order. The output of one selector is the
    input of the next selector.

    Args:
        selectors (Sequence[SeriesInfoSelector]): The :class:`SeriesInfoSelector` s of the pipeline in a given order.
    """

    def __init__(self, selectors: Sequence[SeriesInfoSelector]) -> None:
        super().__init__()

        self.selectors: List[SeriesInfoSelector] = [selector for selector in selectors]

    def add_selector(self, selector: SeriesInfoSelector) -> None:
        """Add a :class:`SeriesInfoSelector` to the pipeline.

        Args:
            selector (SeriesInfoSelector): The :class:`SeriesInfoSelector` to add.
        """
        self.selectors.append(selector)

    def execute(self, infos: Sequence[SeriesInfo]) -> Tuple[SeriesInfo, ...]:
        """Perform the selection of the :class:`~pyradise.fileio.series_info.SeriesInfo` entries according to
        the :class:`SeriesInfoSelector` s specified.

        Args:
            infos (Sequence[SeriesInfo]): The :class:`~pyradise.fileio.series_info.SeriesInfo` entries to select
             from.

        Returns:
            Tuple[SeriesInfo, ...]: The selected :class:`~pyradise.fileio.series_info.SeriesInfo` entries.
        """
        for selector in self.selectors:
            infos = selector.execute(infos)
        return infos


class ModalityInfoSelector(SeriesInfoSelector):
    """A :class:`SeriesInfoSelector` to remove all :class:`~pyradise.fileio.series_info.IntensityFileSeriesInfo` and
    :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo` entries that do not have a matching
    :class:`~pyradise.data.modality.Modality`.

     Note:
         If a :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo` entry is removed, the associated
         :class:`~pyradise.fileio.series_info.DicomSeriesRegistrationInfo` entry is also removed because a
         registration always requires both referenced registration images.


     Args:
         keep (Tuple[Union[Modality, str], ...]): The :class:`~pyradise.data.modality.Modality` entries of
          the :class:`~pyradise.fileio.series_info.SeriesInfo` entries to keep.
    """

    def __init__(self, keep: Tuple[Union[Modality, str], ...]) -> None:
        super().__init__()

        assert keep, "The modalities to keep must not be empty!"
        self.keep: Tuple[Modality, ...] = seq_to_modalities(keep)

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
            criteria = [
                entry.series_instance_uid == registration_info.referenced_series_instance_uid_transform
                for entry in image_infos
            ]

            if not any(criteria):
                remove_indices.append(i)

        for index in reversed(remove_indices):
            registration_infos.pop(index)

        keep = [entry for entry in infos if not isinstance(entry, DicomSeriesRegistrationInfo)]
        keep.extend(registration_infos)
        return tuple(keep)

    # noinspection DuplicatedCode
    # pylint: disable=duplicate-code
    def execute(self, infos: Sequence[SeriesInfo]) -> Tuple[SeriesInfo, ...]:
        """Remove all :class:`~pyradise.fileio.series_info.IntensityFileSeriesInfo` and
        :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo` entries that do not contain
        one of the specified :class:`~pyradise.data.modality.Modality` entries.

        Args:
            infos (Sequence[SeriesInfo]): The :class:`~pyradise.fileio.series_info.SeriesInfo` entries to select
             from.

        Returns:
            Tuple[SeriesInfo, ...]: The selected :class:`~pyradise.fileio.series_info.SeriesInfo` entries.
        """
        assert infos, "The series infos must not be empty!"

        selected: List[SeriesInfo] = []
        for info in infos:
            if isinstance(info, IntensityFileSeriesInfo):
                if info.modality in self.keep:
                    selected.append(info)

            elif isinstance(info, DicomSeriesImageInfo):
                if info.get_modality() in self.keep:
                    selected.append(info)
            else:
                selected.append(info)

        return self._remove_unused_registration_infos(tuple(selected))


class OrganInfoSelector(SeriesInfoSelector):
    """A :class:`SeriesInfoSelector` to remove all :class:`~pyradise.fileio.series_info.SegmentationFileSeriesInfo`
    entries that do not have a matching :class:`~pyradise.data.organ.Organ`.

    Important:
        This selector does not remove :class:`~pyradise.fileio.series_info.DicomSeriesRTSSInfo` entries
        because a DICOM-RTSS contains multiple organs and the information about the organs is not retrieved before
        loading.

    Args:
        keep (Tuple[Union[Organ, str], ...]): The :class:`~pyradise.data.organ.Organ` entries of the
         :class:`~pyradise.fileio.series_info.SeriesInfo` entries to keep.

    """

    def __init__(self, keep: Tuple[Union[Organ, str], ...]) -> None:
        super().__init__()

        assert keep, "The organs to keep must not be empty!"
        self.keep: Tuple[Organ, ...] = seq_to_organs(keep)

    def execute(self, infos: Sequence[SeriesInfo]) -> Tuple[SeriesInfo, ...]:
        """Remove all :class:`~pyradise.fileio.series_info.SegmentationFileSeriesInfo` entries that do not contain
        one of the specified :class:`~pyradise.data.organ.Organ` entries.

        Args:
            infos (Sequence[SeriesInfo]): The :class:`~pyradise.fileio.series_info.SeriesInfo` entries to select from.

        Returns:
            Tuple[SeriesInfo, ...]: The selected :class:`~pyradise.fileio.series_info.SeriesInfo` entries.
        """
        assert infos, "The series infos must not be empty!"

        selected: List[SeriesInfo] = []
        for info in infos:
            if isinstance(info, SegmentationFileSeriesInfo):
                if info.organ in self.keep:
                    selected.append(info)
            else:
                selected.append(info)

        return tuple(selected)


class AnnotatorInfoSelector(SeriesInfoSelector):
    """A :class:`SeriesInfoSelector` to remove all :class:`~pyradise.fileio.series_info.SegmentationFileSeriesInfo` and
    :class:`~pyradise.fileio.series_info.DicomSeriesRTSSInfo` entries that do not have a matching
    :class:`~pyradise.data.annotator.Annotator`.

    Args:
        keep (Tuple[Union[Annotator, str], ...]): The :class:`~pyradise.data.annotator.Annotator` entries of the
         :class:`~pyradise.fileio.series_info.SeriesInfo` entries to keep.
    """

    def __init__(self, keep: Tuple[Union[Annotator, str], ...]) -> None:
        super().__init__()

        assert keep, "The annotators to keep must not be empty!"
        self.keep: Tuple[Annotator, ...] = seq_to_annotators(keep)

    # noinspection DuplicatedCode
    # pylint: disable=duplicate-code
    def execute(self, infos: Sequence[SeriesInfo]) -> Tuple[SeriesInfo, ...]:
        """Remove all :class:`~pyradise.fileio.series_info.SegmentationFileSeriesInfo` and
        :class:`~pyradise.fileio.series_info.DicomSeriesRTSSInfo` entries that do not contain one of the specified
        :class:`~pyradise.data.annotator.Annotator` entries.

        Args:
            infos (Sequence[SeriesInfo]): The :class:`~pyradise.fileio.series_info.SeriesInfo` entries to select from.

        Returns:
            Tuple[SeriesInfo, ...]: The selected :class:`~pyradise.fileio.series_info.SeriesInfo` entries.
        """
        assert infos, "The series infos must not be empty!"

        selected: List[SeriesInfo] = []
        for info in infos:
            if isinstance(info, SegmentationFileSeriesInfo):
                if info.get_annotator() in self.keep:
                    selected.append(info)

            elif isinstance(info, DicomSeriesRTSSInfo):
                if info.annotator in self.keep:
                    selected.append(info)
            else:
                selected.append(info)

        return tuple(selected)


class NoRegistrationInfoSelector(SeriesInfoSelector):
    """A :class:`SeriesInfoSelector` to remove all :class:`~pyradise.fileio.series_info.DicomSeriesRegistrationInfo`
    entries such that no registration is applied during loading."""

    def execute(self, infos: Sequence[SeriesInfo]) -> Tuple[SeriesInfo, ...]:
        """Remove all :class:`~pyradise.fileio.series_info.DicomSeriesRegistrationInfo` entries from the provided
        :class:`~pyradise.fileio.series_info.SeriesInfo` entries.

        Args:
            infos (Tuple[SeriesInfo, ...]): The :class:`~pyradise.fileio.series_info.SeriesInfo` entries to select from.

        Returns:
            Sequence[SeriesInfo]: The selected :class:`~pyradise.fileio.series_info.SeriesInfo` entries.
        """
        assert infos, "The series infos must not be empty!"

        selected: List[SeriesInfo] = []
        for info in infos:
            if not isinstance(info, DicomSeriesRegistrationInfo):
                selected.append(info)

        return tuple(selected)


class NoRTSSInfoSelector(SeriesInfoSelector):
    """A :class:`SeriesInfoSelector` to remove all :class:`~pyradise.fileio.series_info.DicomSeriesRTSSInfo`
    entries such that all DICOM-RTSS data is excluded from loading."""

    def execute(self, infos: Sequence[SeriesInfo]) -> Tuple[SeriesInfo, ...]:
        """Remove all :class:`~pyradise.fileio.series_info.DicomSeriesRTSSInfo` entries from the provided
        :class:`~pyradise.fileio.series_info.SeriesInfo` entries.

        Args:
            infos (Sequence[SeriesInfo]): The :class:`~pyradise.fileio.series_info.SeriesInfo` entries to select from.

        Returns:
            Tuple[SeriesInfo, ...]: The selected :class:`~pyradise.fileio.series_info.SeriesInfo` entries.
        """
        assert infos, "The series infos must not be empty!"

        selected: List[SeriesInfo] = []
        for info in infos:
            if not isinstance(info, DicomSeriesRTSSInfo):
                selected.append(info)

        return tuple(selected)
