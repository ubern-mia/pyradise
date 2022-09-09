from abc import (
    ABC,
    abstractmethod)
import os
from typing import (
    Any,
    Callable,
    Tuple)

import itk
from pydicom.tag import Tag

from pyradise.data import (
    Modality,
    Organ,
    Rater)
from pyradise.utils import (
    is_dir_and_exists,
    is_dicom_file,
    load_dataset_tag)
from .modality_config import ModalityConfiguration
from .series_info import (
    FileSeriesInfo,
    IntensityFileSeriesInfo,
    SegmentationFileSeriesInfo,
    DicomSeriesInfo,
    DicomSeriesImageInfo,
    DicomSeriesRegistrationInfo,
    DicomSeriesRTSSInfo)


__all__ = ['Crawler', 'SubjectFileCrawler', 'DatasetFileCrawler', 'IterableFileCrawler',
           'SubjectDicomCrawler', 'DatasetDicomCrawler', 'IterableDicomCrawler']


def default_modality_extractor(path: str) -> Modality:
    """Extract the modality from the specified file path, if possible. If the modality cannot be extracted,
    the default modality :class:`Modality.UNKNOWN` is returned.

    Args:
        path (str): The path from which to extract the modality.

    Returns:
        Modality: The extracted modality or the default value (:class:`Modality.UNKNOWN`).
    """

    if os.path.basename(path).startswith('seg'):
        return Modality.UNKNOWN

    try:
        modality = Modality[os.path.basename(path).split('.')[0].split('_')[-1]]
    except KeyError:
        print(f'Could not extract modality from path {path}! Assigned the default value Modality.UNKNOWN!')
        modality = Modality.UNKNOWN

    return modality


def default_organ_extractor(path: str) -> Organ:
    """Extract the organ from the specified file path, if possible. If the organ cannot be extracted, the default
    organ :class:`Organ('NA')` is returned.

    Args:
        path (str): The path from which to extract the organ.

    Returns:
        Organ: The extracted organ or the default value (:class:`Organ('NA')`).
    """
    if os.path.basename(path).startswith('img'):
        return Organ('NA')

    organ_name = os.path.basename(path).split('.')[0].split('_')[-1]

    if organ_name:
        return Organ(organ_name)
    return Organ('NA')


def default_rater_extractor(path: str) -> Rater:
    """Extract the rater from the specified file path, if possible. If the rater cannot be extracted, the default
    rater :class:`Rater('NA')` is returned.

    Args:
        path (str): The path from which to extract the rater.

    Returns:
        Rater: The extracted rater or the default rater (:class:`Rater('NA')`).
    """
    if os.path.basename(path).startswith('img'):
        return Rater('NA')

    rater_name = os.path.basename(path).split('.')[0].split('_')[2]

    if rater_name:
        return Rater(rater_name)
    return Rater('NA')


class Crawler(ABC):
    """An abstract crawler whose subtypes are used for searching a certain type of files in a filesystem hierarchy.

    Args:
        path (str): The directory path for which the crawling will be performed.
    """

    def __init__(self,
                 path: str
                 ) -> None:
        super().__init__()

        self.path = is_dir_and_exists(os.path.normpath(path))

    @abstractmethod
    def execute(self) -> Any:
        """Execute the crawling process.

        Returns:
            Any: The crawled data.
        """
        raise NotImplementedError()


class SubjectFileCrawler(Crawler):
    """A crawler for retrieving :class:`FileSeriesInfo` entries from a subject directory containing image files of a
    specified type (see ``extension`` parameter).

    Notes:
        The :class:`SubjectFileCrawler` is used for reading data from a specific subject directory containing all
        the subject's data. If there are multiple subject directories we recommend using the :class:`DatasetFileCrawler`
        or the :class:`IterableDatasetFileCrawler`.

    Warning:
        The DICOM format is not supported by this crawler! Use the appropriate DICOM variant instead.

    Args:
        path (str): The directory path for which the crawling will be performed.
    """

    def __init__(self,
                 path: str,
                 subject_name: str,
                 extension: str,
                 modality_extractor: Callable[[str], Modality] = default_modality_extractor,
                 organ_extractor: Callable[[str], Organ] = default_organ_extractor,
                 rater_extractor: Callable[[str], Rater] = default_rater_extractor
                 ) -> None:
        super().__init__(path)

        if 'dcm' in extension:
            raise ValueError(f'The DICOM format is not supported by {self.__class__.__name__}! '
                             'Use the appropriate DICOM variant instead.')

        self.extension = extension
        self.subject_name = subject_name
        self.modality_extractor = modality_extractor
        self.organ_extractor = organ_extractor
        self.rater_extractor = rater_extractor

    def execute(self) -> Tuple[FileSeriesInfo, ...]:
        """Execute the crawling process.

        Returns:
            Tuple[FileSeriesInfo, ...]: The crawled data.
        """

        series_infos = []
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(self.extension):
                    file_path = os.path.join(root, file)

                    if (modality := self.modality_extractor(file_path)) == Modality.UNKNOWN:
                        organ = self.organ_extractor(file_path)
                        rater = self.rater_extractor(file_path)

                        series_info = SegmentationFileSeriesInfo(file_path, self.subject_name, organ, rater)
                    else:
                        series_info = IntensityFileSeriesInfo(file_path, self.subject_name, modality)

                    series_infos.append(series_info)

        return tuple(series_infos)


class DatasetFileCrawler(Crawler):
    """A crawler for retrieving :class:`FileSeriesInfo` entries from a dataset directory containing at least
    one subject directory with image files of a specified type (see ``extension`` parameter).

    Notes:
        This crawler is used for reading data from a dataset directory containing multiple separate subject directories.

        If you want to load a large dataset with many subjects, we recommend using the :class:`IterableFileCrawler`
        which loads the subjects iteratively and thus reducing memory requirements.

    Warning:
        The DICOM format is not supported by this crawler! Use the appropriate DICOM variant instead.

    Args:
        path (str): The dataset directory path for which the crawling will be performed.
        extension (str): The file extension of the image files to be crawled.
        modality_extractor (Callable[[str], Modality]): The function used for extracting the :class:`Modality` from
         the file path.
        organ_extractor (Callable[[str], Organ]): The function used for extracting the :class:`Organ` from the
         file path.
        rater_extractor (Callable[[str], Rater]): The function used for extracting the :class:`Rater` from the
         file path.
    """

    def __init__(self,
                 path: str,
                 extension: str,
                 modality_extractor: Callable[[str], Modality] = default_modality_extractor,
                 organ_extractor: Callable[[str], Organ] = default_organ_extractor,
                 rater_extractor: Callable[[str], Rater] = default_rater_extractor
                 ) -> None:
        super().__init__(path)

        if 'dcm' in extension:
            raise ValueError(f'The DICOM format is not supported by {self.__class__.__name__}! '
                             'Use the appropriate DICOM variant instead.')
        self.extension = extension

        self.modality_extractor = modality_extractor
        self.organ_extractor = organ_extractor
        self.rater_extractor = rater_extractor

    @staticmethod
    def _get_subject_dir_paths(path: str, extension: str) -> Tuple[str, ...]:
        """Get the paths of the subject directories containing valid files.

        Args:
            path (str): The directory path for which the crawling will be performed.
            extension (str): The file extension of the files to be considered.

        Returns:
            Tuple[str, ...]: The subject directory paths containing valid files.
        """
        candidate_dir_paths = [entry.path for entry in os.scandir(path) if entry.is_dir()]

        subject_dir_paths = []
        for candidate_dir_path in candidate_dir_paths:
            for root, _, files in os.walk(candidate_dir_path):
                for file in files:
                    if file.endswith(extension):
                        subject_dir_paths.append(candidate_dir_path)
                        break

        return tuple(subject_dir_paths)

    def execute(self) -> Tuple[Tuple[FileSeriesInfo, ...], ...]:
        """Execute the crawling process.

        Returns:
            Tuple[Tuple[FileSeriesInfo, ...], ...]: The crawled data.
        """
        # Get subject folders
        subject_folders = [entry for entry in os.scandir(self.path) if entry.is_dir()]

        # Get subject files
        subject_files = []
        for subject_folder in subject_folders:
            subject_file_crawler = SubjectFileCrawler(subject_folder.path, subject_folder.name, self.extension,
                                                      self.modality_extractor, self.organ_extractor,
                                                      self.rater_extractor)
            subject_files.append(subject_file_crawler.execute())

        return tuple(subject_files)


class IterableFileCrawler(DatasetFileCrawler):
    """An iterable crawler for retrieving :class:`FileSeriesInfo` entries from a dataset directory containing at least
    one subject directory with image files of a specified type (see ``extension`` parameter). In contrast to the
    :class:`DatasetFileCrawler` this crawler loads the subjects iteratively reducing memory requirements.

    Notes:
        This crawler is used for reading data from a dataset directory containing multiple separate subject directories.

    Warning:
        The DICOM format is not supported by this crawler! Use the appropriate DICOM variant instead.

    Args:
        path (str): The dataset directory path for which the crawling will be performed.
        extension (str): The file extension of the image files to be crawled.
        modality_extractor (Callable[[str], Modality]): The function used for extracting the :class:`Modality` from
         the file path.
        organ_extractor (Callable[[str], Organ]): The function used for extracting the :class:`Organ` from the
         file path.
        rater_extractor (Callable[[str], Rater]): The function used for extracting the :class:`Rater` from the
         file path.
    """

    def __init__(self,
                 path: str,
                 extension: str,
                 modality_extractor: Callable[[str], Modality] = default_modality_extractor,
                 organ_extractor: Callable[[str], Organ] = default_organ_extractor,
                 rater_extractor: Callable[[str], Rater] = default_rater_extractor
                 ) -> None:
        super().__init__(path, extension, modality_extractor, organ_extractor, rater_extractor)

        # Get subject directories and subject names
        subject_dir_paths = self._get_subject_dir_paths(self.path, self.extension)
        self.subject_dir_paths = tuple(sorted(subject_dir_paths))
        self.subject_names = tuple(os.path.basename(subject_dir_path) for subject_dir_path in self.subject_dir_paths)

        self.current_idx = 0
        self.num_subjects = len(self.subject_dir_paths)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> Tuple[FileSeriesInfo]:
        if self.current_idx < self.num_subjects:
            subject_info = SubjectFileCrawler(self.subject_dir_paths[self.current_idx],
                                              self.subject_dir_paths[self.current_idx].name,
                                              self.extension,
                                              self.modality_extractor,
                                              self.organ_extractor,
                                              self.rater_extractor).execute()

            self.current_idx += 1

            return subject_info

        raise StopIteration

    def __len__(self) -> int:
        return self.num_subjects


class SubjectDicomCrawler(Crawler):
    """A crawler for retrieving :class:`DicomSeriesInfo` entries from a subject directory containing DICOM files
    (e.g. DICOM images, DICOM registrations, DICOM RTSS).

    Notes:
        The :class:`SubjectDicomCrawler` is used for reading data from a specific subject directory containing all
        the subject's DICOM data. If there are multiple subject directories we recommend using the
        :class:`DatasetDicomCrawler` or the :class:`IterableDicomCrawler`.

        The :class:`SubjectDicomCrawler` can also be used to generate the modality configuration file skeleton for a
        specific subject. In this case set the ``generate_modality_config`` parameter to ``True``.

    Warning:
        This crawler exclusively support the DICOM file format and does not support any other type of file format.

    Args:
        path (str): The subject directory path.
        modality_config_file_name (str): The file name for the modality configuration file within the subject
         directory (default: modality_config.json).
        write_modality_config (bool): If True writes the modality configuration retrieved from the subject
         directory to the subject directory (default: False).
    """

    def __init__(self,
                 path: str,
                 modality_config_file_name: str = 'modality_config.json',
                 write_modality_config: bool = False
                 ) -> None:
        super().__init__(path)
        self.config_file_name = modality_config_file_name
        self.write_config = write_modality_config

    def _get_dicom_files(self) -> Tuple[str, ...]:
        """Get all DICOM files in the subject directory.

        Returns:
            Tuple[str, ...]: The DICOM file paths.
        """
        file_paths = []
        for root, _, files in os.walk(self.path):
            for file in files:
                file_path = os.path.join(root, file)
                if is_dicom_file(file_path):
                    file_paths.append(file_path)

        return tuple(file_paths)

    def _get_image_files(self) -> Tuple[Tuple[str, ...], ...]:
        """Get all DICOM image files in the subject directory.

        Notes:
            The DICOM image files are grouped by their SeriesInstanceUID.

        Returns:
            Tuple[Tuple[str, ...], ...]: The DICOM image file paths separated by SeriesInstanceUID.
        """
        series_extractor = itk.GDCMSeriesFileNames.New()
        series_extractor.SetRecursive(True)
        series_extractor.SetDirectory(self.path)
        series_extractor.Update()

        image_series_paths = []
        for series_uid in series_extractor.GetSeriesUIDs():
            series_paths = [str(os.path.normpath(entry)) for entry in series_extractor.GetFileNames(series_uid)]
            image_series_paths.append(tuple(series_paths))

        return tuple(image_series_paths)

    @staticmethod
    def _get_registration_files(paths: Tuple[str, ...]) -> Tuple[str, ...]:
        """"Get all DICOM registration files in the subject directory.

        Args:
            paths (Tuple[str, ...]): The DICOM file paths to check if they specify a DICOM registration file.

        Returns:
            Tuple[str, ...]: The DICOM registration file paths.
        """
        valid_sop_class_uids = ('1.2.840.10008.5.1.4.1.1.66.1',  # Spatial Registration Storage
                                '1.2.840.10008.5.1.4.1.1.66.3')  # Deformable Spatial Registration Storage

        registration_files = []
        for path in paths:
            dataset = load_dataset_tag(path, (Tag(0x0008, 0x0016),))

            if dataset.get('SOPClassUID', None) in valid_sop_class_uids:
                registration_files.append(path)

        return tuple(registration_files)

    @staticmethod
    def _get_rtss_files(paths: Tuple[str, ...]) -> Tuple[str, ...]:
        """"Get all DICOM RTSS files in the subject directory.

        Args:
            paths (Tuple[str, ...]): The DICOM file paths to check if they specify a DICOM RTSS file.

        Returns:
            Tuple[str, ...]: The DICOM RTSS file paths.
        """
        valid_sop_class_uid = '1.2.840.10008.5.1.4.1.1.481.3'  # RT Structure Set Storage

        rtss_files = []
        for path in paths:
            dataset = load_dataset_tag(path, (Tag(0x0008, 0x0016),))

            if dataset.get('SOPClassUID', None) == valid_sop_class_uid:
                rtss_files.append(path)

        return tuple(rtss_files)

    @staticmethod
    def _generate_image_infos(image_paths: Tuple[Tuple[str, ...], ...]) -> Tuple[DicomSeriesImageInfo]:
        """Generate the :class:`DicomSeriesImageInfos` from the DICOM file paths specified.

        Args:
            image_paths (Tuple[Tuple[str, ...], ...]): The DICOM image file paths provided.

        Returns:
            Tuple[DicomSeriesImageInfo, ...]: The retrieved :class:`DicomSeriesImageInfo` entries.
        """
        infos = []

        for path_set in image_paths:
            image_info = DicomSeriesImageInfo(path_set)
            infos.append(image_info)

        return tuple(infos)

    @staticmethod
    def _generate_registration_infos(registration_paths: Tuple[str, ...],
                                     image_infos: Tuple[DicomSeriesImageInfo, ...]
                                     ) -> Tuple[DicomSeriesRegistrationInfo]:
        """Generate the :class:`DicomSeriesRegistrationInfos` from the DICOM file paths specified.

        Args:
            registration_paths (Tuple[str, ...]): The DICOM registration file paths provided.
            image_infos (Tuple[DicomSeriesImageInfo, ...]): The available :class:`DicomSeriesImageInfo` entries.

        Returns:
            Tuple[DicomSeriesRegistrationInfo, ...]: The retrieved :class:`DicomSeriesRegistrationInfo`.
        """
        infos = []

        for path in registration_paths:
            registration_info = DicomSeriesRegistrationInfo(path, image_infos, persistent_image_infos=False)
            infos.append(registration_info)

        return tuple(infos)

    @staticmethod
    def _generate_rtss_info(rtss_paths: Tuple[str, ...]) -> Tuple[DicomSeriesRTSSInfo]:
        """Generate the :class:`DicomSeriesRTStructureSetInfo` entries from the DICOM file paths specified.

        Args:
            rtss_paths (Tuple[str, ...]): The DICOM RTSS file paths.

        Returns:
            Tuple[DicomSeriesRTStructureSetInfo, ...]: AThe retrieved :class:`DicomSeriesRTStructureSetInfos`.
        """
        infos = []

        for path in rtss_paths:
            rtss_info = DicomSeriesRTSSInfo(path)
            infos.append(rtss_info)

        return tuple(infos)

    def _export_modality_config(self,
                                infos: Tuple[DicomSeriesInfo, ...]
                                ) -> None:
        """Export the retrieved :class:`ModalityConfiguration` to a file.

        Args:
            infos (Tuple[DicomSeriesInfo, ...]): The :class:`DicomSeriesInfo` entries containing the information to
             export.

        Returns:
            None
        """
        config = ModalityConfiguration.from_dicom_series_info(infos)
        config.to_file(os.path.join(self.path, self.config_file_name))

    def _apply_modality_config(self, infos: Tuple[DicomSeriesImageInfo, ...]) -> None:
        """Load the :class:`ModalityConfiguration` from a file and apply it to the specified
        :class:`DicomSeriesImageInfo` entries.

        Args:
            infos (Tuple[DicomSeriesImageInfo, ...]): The available :class:`DicomSeriesImageInfo` entries to which the
             loaded :class:`ModalityConfiguration` can be applied.

        Returns:
            None
        """
        # search for the modality file
        modality_file_path = ''
        for root, _, files in os.walk(self.path):
            for file in files:
                if self.config_file_name in file:
                    modality_file_path = os.path.join(root, file)
                    break

        if not os.path.exists(modality_file_path):
            raise ValueError(f'The modality configuration is not found in the specified path: {self.path}')

        config = ModalityConfiguration.from_file(modality_file_path)
        config.add_modalities_to_info(infos)

    def execute(self) -> Tuple[DicomSeriesInfo, ...]:
        """Execute the crawling process to get the :class:`DicomSeriesInfo` entries.

        Returns:
            Tuple[DicomSeriesInfo, ...]: The retrieved :class:`DicomSeriesInfo` entries.
        """
        # get the dicom file paths and sort them according to the file content
        file_paths = self._get_dicom_files()
        image_paths = self._get_image_files()

        flat_image_paths = [path for paths in image_paths for path in paths]
        remaining_paths = tuple(set(file_paths) - set(flat_image_paths))

        registration_paths = self._get_registration_files(remaining_paths)
        remaining_paths = tuple(set(remaining_paths) - set(registration_paths))

        rtss_paths = self._get_rtss_files(remaining_paths)

        # generate the series infos
        image_infos = self._generate_image_infos(image_paths)
        registration_infos = self._generate_registration_infos(registration_paths, image_infos)
        rtss_infos = self._generate_rtss_info(rtss_paths)

        # apply the modality config
        self._apply_modality_config(image_infos)

        # write the config to a file if necessary
        if self.write_config:
            config = ModalityConfiguration.from_dicom_series_info(image_infos)
            config.to_file(os.path.join(self.path, self.config_file_name))

        return image_infos + registration_infos + rtss_infos


class DatasetDicomCrawler(Crawler):
    """A crawler for retrieving :class:`DicomSeriesInfo` entries from a dataset directory containing at least one
    subject directory with DICOM files (e.g. DICOM images, DICOM registrations, DICOM RTSS).

    Notes:
        The :class:`DatasetDicomCrawler` is used for reading data from a dataset directory containing at least one
        subject folder with DICOM files. If there is only one subject directory we recommend using the
        :class:`SubjectDicomCrawler`.

        If you want to load a large dataset with many subjects, we recommend using the :class:`IterableDicomCrawler`
        which loads the subjects iteratively and thus reducing memory requirements.

        The :class:`DatasetDicomCrawler` can also be used to generate the modality configuration file skeletons for all
        subjects in a dataset. In this case set the ``generate_modality_config`` parameter to ``True``.

    Warning:
        This crawler exclusively support the DICOM file format and does not support any other type of file format.

    Args:
        path (str): The dataset directory path.
        modality_config_file_name (str): The file name for the modality configuration file within the subject
         directory (default: modality_config.json).
        write_modality_config (bool): If True writes the modality configuration retrieved from the subject
         directory to the subject directory (default: False).
    """

    def __init__(self,
                 path: str,
                 modality_config_file_name: str = 'modality_config.json',
                 write_modality_config: bool = False) -> None:
        super().__init__(path)
        self.config_file_name = modality_config_file_name
        self.write_config = write_modality_config

    @staticmethod
    def _get_subject_dir_paths(path: str) -> Tuple[str, ...]:
        """Get the paths of the subject directories containing DICOM files.

        Args:
            path (str): The base directory path which contain the subject directories.

        Returns:
            Tuple[str, ...]: Paths to all subject directories containing DICOM files.
        """
        candidate_dir_paths = [entry.path for entry in os.scandir(path) if entry.is_dir()]

        subject_dir_paths = []
        for candidate_dir_path in candidate_dir_paths:
            for root, _, files in os.walk(candidate_dir_path):
                for file in files:
                    if is_dicom_file(os.path.join(root, file)):
                        subject_dir_paths.append(os.path.normpath(candidate_dir_path))
                        break

        return tuple(subject_dir_paths)

    def execute(self) -> Tuple[Tuple[DicomSeriesInfo, ...], ...]:
        """Execute the crawling process to get the :class:`DicomSeriesInfos` for each subject directory separately.

        Returns:
            Tuple[Tuple[DicomSeriesInfo, ...], ...]: The retrieved :class:`DicomSeriesInfos`.
        """
        subject_dir_paths = self._get_subject_dir_paths(self.path)

        subject_infos = []
        for subject_dir_path in subject_dir_paths:
            subject_info = SubjectDicomCrawler(subject_dir_path, self.config_file_name,
                                               self.write_config).execute()

            if subject_info:
                subject_infos.append(subject_info)

        return tuple(subject_infos)


class IterableDicomCrawler(DatasetDicomCrawler):
    """An iterable crawler for retrieving :class:`DicomSeriesInfo` entries from a dataset directory containing at
    least one subject directory with DICOM files (e.g. DICOM images, DICOM registrations, DICOM RTSS). In contrast to
    the :class:`DatasetDicomCrawler` this crawler loads the subjects iteratively reducing memory requirements.

    Notes:
        The :class:`IterableDicomCrawler` is used for reading data from a dataset directory containing at least one
        subject folder with DICOM files. If there is only one subject directory we recommend using the
        :class:`SubjectDicomCrawler`.

        The :class:`IterableDicomCrawler` can also be used to generate the modality configuration file skeletons for all
        subjects in a dataset. In this case set the ``generate_modality_config`` parameter to ``True``.

    Warning:
        This crawler exclusively support the DICOM file format and does not support any other type of file format.

    Args:
        path (str): The dataset directory path.
        modality_config_file_name (str): The file name for the modality configuration file within the subject
         directory (default: modality_config.json).
        write_modality_config (bool): If True writes the modality configuration retrieved from the subject
         directory to the subject directory (default: False).
    """

    def __init__(self,
                 path: str,
                 modality_config_file_name: str = 'modality_config.json',
                 write_modality_config: bool = False
                 ) -> None:
        super().__init__(path, modality_config_file_name, write_modality_config)

        subject_dir_path = super()._get_subject_dir_paths(self.path)
        self.subject_dir_path = tuple(sorted(subject_dir_path))

        self.current_idx = 0
        self.num_subjects = len(self.subject_dir_path)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> Tuple[DicomSeriesInfo]:
        if self.current_idx < self.num_subjects:
            subject_info = SubjectDicomCrawler(self.subject_dir_path[self.current_idx],
                                               self.config_file_name, self.write_config).execute()
            self.current_idx += 1

            return subject_info

        raise StopIteration

    def __len__(self) -> int:
        return self.num_subjects
