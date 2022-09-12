from abc import (ABC, abstractmethod)
from typing import (
    Tuple,
    List,
    Sequence)

import SimpleITK as sitk

from pyradise.data import (
    Subject,
    IntensityImage,
    SegmentationImage)
from .series_info import (
    SeriesInfo,
    FileSeriesInfo,
    IntensityFileSeriesInfo,
    SegmentationFileSeriesInfo,
    DicomSeriesInfo,
    DicomSeriesImageInfo,
    DicomSeriesRTSSInfo,
    DicomSeriesRegistrationInfo)
from .dicom_conversion import (
    DicomImageSeriesConverter,
    DicomRTSSSeriesConverter)

__all__ = ['DirectBaseLoader', 'SubjectLoader', 'IterableSubjectLoader', 'SubjectLoaderV2', 'IterableSubjectLoaderV2']


class LoaderBase(ABC):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _extract_info_by_type(info: Sequence[SeriesInfo],
                              type_: type
                              ) -> Tuple:
        """Extract all :class:`SeriesInfo` entries of the specified type from the provided sequence.

        Args:
            info (Sequence[SeriesInfo]): The sequence of :class:`SeriesInfo` entries.
            type_ (type): The type of the :class:`SeriesInfo` entries to be extracted.

        Returns:
            Tuple[SeriesInfo]: The extracted :class:`SeriesInfo` entries.
        """
        return tuple(filter(lambda x: isinstance(x, type_), info))


class DirectBaseLoader(LoaderBase, ABC):
    """An abstract loader class to load common discrete medical image file formats.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def load(self,
             info: Tuple[SeriesInfo]
             ) -> Subject:
        """Load the :class:`Subject`.

        Args:
            info (Tuple[SeriesInfo]): The :class:`SeriesInfo` entries to be loaded.

        Returns:
            Subject: The loaded :class:`Subject`.
        """
        raise NotImplementedError()


class SubjectLoader(DirectBaseLoader):
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


class IterableSubjectLoader(LoaderBase):
    """An iterable loader to load a sequence of :class:`Subject` s from a list of :class:`FileSeriesInfo` entries.

    Notes:
        The ``info`` argument must be a list of lists of :class:`FileSeriesInfo` entries. Each list of
        :class:`FileSeriesInfo` entries represents a separate :class:`Subject`.

    Args:
        info (Tuple[Tuple[FileSeriesInfo]]): The nested :class:`FileSeriesInfo` entries to load the subjects from.
        intensity_pixel_type (int): The pixel type of the intensity images.
        segmentation_pixel_type (int): The pixel type of the segmentation images.
    """

    def __init__(self,
                 info: Tuple[Tuple[FileSeriesInfo]],
                 intensity_pixel_type: int = sitk.sitkFloat32,
                 segmentation_pixel_type: int = sitk.sitkUInt8
                 ) -> None:
        super().__init__()

        assert info, 'The infos must not be empty!'
        self.info = info

        self.intensity_pixel_type = intensity_pixel_type
        self.segmentation_pixel_type = segmentation_pixel_type

        self.current_idx = 0
        self.num_subjects = len(self.info)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx < self.num_subjects:
            subject = SubjectLoader(self.info[self.current_idx]).load()
            self.current_idx += 1
            return subject

        raise StopIteration()

    def __len__(self):
        return self.num_subjects


class DicomLoader(DirectBaseLoader):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _get_image_series_info(info: Tuple[SeriesInfo]
                               ) -> Tuple[DicomSeriesImageInfo]:
        """Extract the :class:`DicomSeriesImageInfo` entries from a sequence of :class:`SeriesInfo`.

        Args:
            info (Tuple[SeriesInfo]): The :class:`SeriesInfo` entries to filter.

        Returns:
            Tuple[DicomSeriesImageInfo]: The extracted :class:`DicomSeriesImageInfo` entries.
        """
        return tuple(entry for entry in info if isinstance(entry, DicomSeriesImageInfo))

    @staticmethod
    def _validate_patient_identification(infos: Tuple[DicomSeriesInfo]) -> bool:
        if not infos:
            return False

        return all(info.patient_name == infos[0].patient_name and info.patient_id == infos[0].patient_id
                   for info in infos)

    @staticmethod
    def _validate_registration_infos(reg_info: Tuple[DicomSeriesRegistrationInfo],
                                     image_info: Tuple[DicomSeriesImageInfo]
                                     ) -> bool:

        def is_image_info_available(instance_uids: List[str],
                                    image_info_: Tuple[DicomSeriesImageInfo]
                                    ) -> bool:
            comparison = [[info.series_instance_uid == uid for uid in instance_uids] for info in image_info_]
            return all(any(comparison_) for comparison_ in comparison)

        if not image_info:
            return False

        if not reg_info:
            return True

        identity_uids = []
        transform_uids = []
        for reg_info_entry in reg_info:
            reg_info_entry.update() if not reg_info_entry.is_updated() else None
            identity_uids.append(reg_info_entry.referenced_series_instance_uid_identity)
            transform_uids.append(reg_info_entry.referenced_series_instance_uid_transform)

        if is_image_info_available(identity_uids, image_info) and is_image_info_available(transform_uids, image_info):
            return True

        return False

    # TODO add here a validate_rtss function

    def load(self, info: Tuple[SeriesInfo]) -> Subject:
        # sort the SeriesInfo entries such that only DicomSeriesInfo entries are existing
        image_info: Tuple[DicomSeriesImageInfo] = self._extract_info_by_type(info, DicomSeriesImageInfo)
        reg_info: Tuple[DicomSeriesRegistrationInfo] = self._extract_info_by_type(info, DicomSeriesRegistrationInfo)
        rtss_info: Tuple[DicomSeriesRTSSInfo] = self._extract_info_by_type(info, DicomSeriesRTSSInfo)

        selected_info: Tuple[DicomSeriesInfo, ...] = image_info + reg_info + rtss_info

        # validate the DicomSeriesInfo entries
        if not self._validate_patient_identification(selected_info):
            raise ValueError('The provided image infos do not contain the same patient identification!')

        if not self._validate_registration_infos(reg_info, image_info):
            raise ValueError('The provided registration infos are invalid!')

        # create the subject
        subject = Subject(selected_info[0].patient_name)

        # convert and add the DICOM intensity images to the subject
        intensity_images = DicomImageSeriesConverter(image_info, reg_info).convert()
        subject.add_images(intensity_images, force=True)

        # convert and add the DICOM segmentation images to the subject
        if rtss_info:
            segmentation_images = DicomRTSSSeriesConverter(rtss_info, image_info, reg_info).convert()
            subject.add_images(segmentation_images, force=True)

        return subject


class IterableDicomLoader(LoaderBase):

    def __init__(self, infos: Tuple[Tuple[SeriesInfo]]):
        super().__init__()

        assert infos, 'The infos must not be empty!'
        self.infos = infos

        self.current_idx = 0
        self.num_subjects = len(self.infos)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx < self.num_subjects:
            subject = DicomLoader().load(self.infos[self.current_idx])
            self.current_idx += 1
            return subject

        raise StopIteration()

    def __len__(self):
        return self.num_subjects

    def load(self) -> Subject:
        raise NotImplementedError(f'The load method is not implemented for {self.__class__.__name__} because it '
                                  'remains unused!')


class SubjectLoaderV2(DirectBaseLoader):

    def __init__(self,
                 intensity_pixel_value_type: int = sitk.sitkFloat32,
                 segmentation_pixel_value_type: int = sitk.sitkUInt8
                 ) -> None:
        super().__init__()

        self.intensity_pixel_type = intensity_pixel_value_type
        self.segmentation_pixel_type = segmentation_pixel_value_type

    @staticmethod
    def _load_intensity_images(info: Tuple[IntensityFileSeriesInfo],
                               pixel_value_type: sitk.sitkFloat32
                               ) -> Tuple[IntensityImage]:
        images = []
        for info_entry in info:
            image = sitk.ReadImage(info_entry.get_path()[0], pixel_value_type)
            images.append(IntensityImage(image, info_entry.get_modality()))

        return tuple(images)

    @staticmethod
    def _load_segmentation_images(info: Tuple[SegmentationFileSeriesInfo],
                                  pixel_value_type: sitk.sitkUInt8
                                  ) -> Tuple[SegmentationImage]:
        images = []
        for info_entry in info:
            image = sitk.ReadImage(info_entry.get_path()[0], pixel_value_type)
            images.append(SegmentationImage(image, info_entry.get_organ(), info_entry.get_rater()))

        return tuple(images)

    @staticmethod
    def _validate_patient_identification(info: Tuple[SeriesInfo]) -> bool:
        if not info:
            return False

        patient_names = []
        patient_ids = []
        for info_entry in info:
            if isinstance(info_entry, DicomSeriesInfo):
                patient_names.append(info_entry.patient_name)
                patient_ids.append(info_entry.patient_id)
            elif isinstance(info_entry, FileSeriesInfo):
                patient_names.append(info_entry.subject_name)
            else:
                raise ValueError(f'Unknown type {type(info_entry)}!')

        result_patient_name = all(patient_name == patient_names[0] for patient_name in patient_names)
        result_patient_id = all(patient_id == patient_ids[0] for patient_id in patient_ids)

        return result_patient_name and result_patient_id

    @staticmethod
    def _validate_registration(reg_info: Tuple[DicomSeriesRegistrationInfo],
                               image_info: Tuple[DicomSeriesImageInfo]
                               ) -> bool:

        def is_image_info_available(instance_uids: List[str],
                                    image_info_: Tuple[DicomSeriesImageInfo]
                                    ) -> bool:
            comparison = [[info.series_instance_uid == uid for info in image_info_] for uid in instance_uids]
            return all(any(comparison_) for comparison_ in comparison)

        if not reg_info:
            return True

        if not image_info:
            return False

        identity_uids = []
        transform_uids = []
        for reg_info_entry in reg_info:
            reg_info_entry.update() if not reg_info_entry.is_updated() else None
            identity_uids.append(reg_info_entry.referenced_series_instance_uid_identity)
            transform_uids.append(reg_info_entry.referenced_series_instance_uid_transform)

        if is_image_info_available(identity_uids, image_info) and is_image_info_available(transform_uids, image_info):
            return True

        return False

    @staticmethod
    def _validate_rtss_info(rtss_info: Tuple[DicomSeriesRTSSInfo],
                            image_info: Tuple[DicomSeriesImageInfo]
                            ) -> bool:
        if not rtss_info:
            return True

        if not image_info:
            return False

        comparison = [any(info.series_instance_uid == rtss_info_entry.referenced_instance_uid for info in image_info) 
                      for rtss_info_entry in rtss_info]

        return all(comparison)

    def load(self, info: Tuple[SeriesInfo]) -> Subject:
        # separate the info entries
        dicom_image_info = self._extract_info_by_type(info, DicomSeriesImageInfo)
        dicom_reg_info = self._extract_info_by_type(info, DicomSeriesRegistrationInfo)
        dicom_rtss_info = self._extract_info_by_type(info, DicomSeriesRTSSInfo)
        intensity_image_info = self._extract_info_by_type(info, IntensityFileSeriesInfo)
        segmentation_image_info = self._extract_info_by_type(info, SegmentationFileSeriesInfo)

        # validate the info entries
        if not self._validate_patient_identification(info):
            raise ValueError('The patient identification is not unique!')
        
        if not self._validate_registration(dicom_reg_info, dicom_image_info):
            raise ValueError('The registration information is not valid!')
        
        if not self._validate_rtss_info(dicom_rtss_info, dicom_image_info):
            raise ValueError('The RTSS information is not valid!')

        # create the subject
        if dicom_image_info:
            subject = Subject(dicom_image_info[0].patient_name)
        elif intensity_image_info:
            subject = Subject(intensity_image_info[0].subject_name)
        elif segmentation_image_info:
            subject = Subject(segmentation_image_info[0].subject_name)
        else:
            raise ValueError('Subject can not be constructed because a subject name is missing!')

        
        # load the images and add them to the subject
        if dicom_image_info:
            dicom_images = DicomImageSeriesConverter(dicom_image_info, dicom_reg_info).convert()
            subject.add_images(dicom_images)

        if dicom_rtss_info:
            dicom_segmentations = DicomRTSSSeriesConverter(dicom_rtss_info, dicom_image_info, dicom_reg_info).convert()
            subject.add_images(dicom_segmentations)

        intensity_images = self._load_intensity_images(intensity_image_info, self.intensity_pixel_type)
        segmentation_images = self._load_segmentation_images(segmentation_image_info, self.segmentation_pixel_type)
        subject.add_images(intensity_images + segmentation_images)

        return subject


class IterableSubjectLoaderV2(LoaderBase):

    def __init__(self, infos: Tuple[Tuple[SeriesInfo, ...], ...]):
        super().__init__()

        assert infos, 'The infos must not be empty!'
        self.infos = infos

        self.current_idx = 0
        self.num_subjects = len(self.infos)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> Subject:
        if self.current_idx < self.num_subjects:
            subject = SubjectLoaderV2().load(self.infos[self.current_idx])
            self.current_idx += 1
            return subject

        raise StopIteration()

    def __len__(self):
        return self.num_subjects
