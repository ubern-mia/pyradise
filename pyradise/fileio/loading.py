from abc import (ABC, abstractmethod)
from typing import (Tuple)

import SimpleITK as sitk

from pyradise.data import (
    Subject,
    IntensityImage,
    SegmentationImage)
from .series_info import (
    FileSeriesInfo,
    IntensityFileSeriesInfo,
    SegmentationFileSeriesInfo)

__all__ = ['Loader', 'SubjectLoader', 'IterableSubjectLoader']


class Loader(ABC):
    """An abstract loader class to load common discrete medical image file formats.

    Args:
        intensity_pixel_type (int): The pixel type of the intensity images.
        segmentation_pixel_type (int): The pixel type of the segmentation images.
    """

    def __init__(self,
                 intensity_pixel_type: int = sitk.sitkFloat32,
                 segmentation_pixel_type: int = sitk.sitkUInt8
                 ) -> None:
        super().__init__()

        self.intensity_pixel_type = intensity_pixel_type
        self.segmentation_pixel_type = segmentation_pixel_type

    @staticmethod
    def _clean_infos(infos: Tuple[FileSeriesInfo]) -> Tuple[FileSeriesInfo]:
        """Clean the :class:`FileSeriesInfo` entries from :class:`DicomSeriesInfo` entries.

        Args:
            infos (Tuple[FileSeriesInfo]): The :class:`FileSeriesInfo` entries to check.

        Returns:
            Tuple[FileSeriesInfo]: The :class:`FileSeriesInfo` entries without :class:`DicomSeriesInfo` entries.
        """
        keep = []
        for info in infos:
            if isinstance(info, FileSeriesInfo):
                keep.append(info)
        return tuple(keep)

    @staticmethod
    def _check_subject_name(infos: Tuple[FileSeriesInfo]) -> None:
        """Check if the subject names are equal.

        Args:
            infos (Tuple[FileSeriesInfo]): The :class:`FileSeriesInfo` entries to check.

        Raises:
            ValueError: If the subject names are not equal.
        """
        subject_names = [info.subject_name for info in infos]
        if not all(subject_name == subject_names[0] for subject_name in subject_names):
            raise ValueError('The subject name for at least one provided info is not equal!')

    def _load_intensity_image(self, info: IntensityFileSeriesInfo) -> IntensityImage:
        """Load an intensity image.

        Args:
            info (IntensityFileSeriesInfo): The :class:`IntensityFileSeriesInfo` to load.

        Raises:
            ValueError: If the provided file series contains more than one file.
            NotImplementedError: If the image has more than three dimensions.

        Returns:
            IntensityImage: The loaded :class:`IntensityImage`.
        """
        if len(info.path) > 1:
            raise ValueError('The provided file series contains more than one file!')

        sitk_image = sitk.ReadImage(info.path[0], outputPixelType=self.intensity_pixel_type)

        if sitk_image.GetDimension() > 3:
            raise NotImplementedError(f'Conversion of {sitk_image.GetDimension()}D images is not supported!')

        return IntensityImage(sitk_image, info.modality)

    def _load_segmentation_image(self, info: SegmentationFileSeriesInfo) -> SegmentationImage:
        """Load a segmentation image.

                Args:
                    info (SegmentationFileSeriesInfo): The :class:`SegmentationFileSeriesInfo` to load.

                Raises:
                    ValueError: If the provided file series contains more than one file.
                    NotImplementedError: If the image has more than three dimensions.

                Returns:
                    SegmentationImage: The loaded :class:`SegmentationImage`.
                """
        if len(info.path) > 1:
            raise ValueError('The provided file series contains more than one file!')

        sitk_image = sitk.ReadImage(info.path[0], outputPixelType=self.segmentation_pixel_type)

        if sitk_image.GetDimension() > 3:
            raise NotImplementedError(f'Conversion of {sitk_image.GetDimension()}D images is not supported!')

        return SegmentationImage(sitk_image, info.organ, info.rater)

    @abstractmethod
    def load(self) -> Subject:
        """Load the :class:`Subject`.

        Returns:
            Subject: The loaded :class:`Subject`.
        """
        raise NotImplementedError()


class SubjectLoader(Loader):
    """A loader class to load a :class:`Subject` from a list of :class:`FileSeriesInfo` entries.

    Args:
        infos (Tuple[FileSeriesInfo]): The :class:`FileSeriesInfo` entries to load the subject from.
        intensity_pixel_type (int): The pixel type of the intensity images.
        segmentation_pixel_type (int): The pixel type of the segmentation images.
    """

    def __init__(self,
                 infos: Tuple[FileSeriesInfo],
                 intensity_pixel_type: int = sitk.sitkFloat32,
                 segmentation_pixel_type: int = sitk.sitkUInt8
                 ) -> None:
        super().__init__()

        assert infos, 'The infos must not be empty!'
        self.infos = infos

        self.intensity_pixel_type = intensity_pixel_type
        self.segmentation_pixel_type = segmentation_pixel_type

    def load(self) -> Subject:
        """Load the :class:`Subject` as specified by the :class:`FileSeriesInfo` entries.

        Returns:
            Subject: The loaded :class:`Subject`.
        """
        file_infos = self._clean_infos(self.infos)
        if len(file_infos) != len(self.infos):
            print('The infos contain DICOM infos which will be disregarded for conversion! '
                  'Use the appropriate converter to ingest DICOM data instead')

        self._check_subject_name(file_infos)

        subject = Subject(file_infos[0].subject_name)

        for info in file_infos:
            if isinstance(info, IntensityFileSeriesInfo):
                image = self._load_intensity_image(info)
                subject.add_image(image)

            elif isinstance(info, SegmentationFileSeriesInfo):
                image = self._load_segmentation_image(info)
                subject.add_image(image)

            else:
                raise ValueError('The provided info is not supported!')

        return subject


class IterableSubjectLoader(Loader):
    """An iterable loader to load a sequence of :class:`Subject` s from a list of :class:`FileSeriesInfo` entries.

    Notes:
        The ``infos`` argument must be a list of lists of :class:`FileSeriesInfo` entries. Each list of
        :class:`FileSeriesInfo` entries represents a separate :class:`Subject`.

    Args:
        infos (Tuple[Tuple[FileSeriesInfo]]): The nested :class:`FileSeriesInfo` entries to load the subjects from.
        intensity_pixel_type (int): The pixel type of the intensity images.
        segmentation_pixel_type (int): The pixel type of the segmentation images.
    """

    def __init__(self,
                 infos: Tuple[Tuple[FileSeriesInfo]],
                 intensity_pixel_type: int = sitk.sitkFloat32,
                 segmentation_pixel_type: int = sitk.sitkUInt8
                 ) -> None:
        super().__init__(intensity_pixel_type, segmentation_pixel_type)

        assert infos, 'The infos must not be empty!'
        self.infos = infos

        self.current_idx = 0
        self.num_subjects = len(self.infos)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx < self.num_subjects:
            subject = SubjectLoader(self.infos[self.current_idx]).load()
            self.current_idx += 1
            return subject

        raise StopIteration()

    def __len__(self):
        return self.num_subjects

    def load(self) -> Subject:
        raise NotImplementedError(f'The load method is not implemented for {self.__class__.__name__} because it '
                                  'remains unused!')
