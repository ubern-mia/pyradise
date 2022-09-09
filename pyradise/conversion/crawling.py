from abc import (
    ABC,
    abstractmethod)
from typing import (
    Any,
    Tuple)
import os

from .series_information import (
    DicomSeriesInfo,
    DicomSeriesImageInfo,
    DicomSeriesRegistrationInfo,
    DicomSeriesRTStructureSetInfo)
from .directory_filtering import DicomCombinedDirectoryFilter
from .configuration import ModalityConfiguration
from .utils import check_is_dir_and_existing

__all__ = ['Crawler', 'DicomSubjectDirectoryCrawler', 'DicomDatasetDirectoryCrawler',
           'IterableDicomDatasetDirectoryCrawler']


class Crawler(ABC):
    """An abstract crawler class.

    Args:
        path (str): The directory path for which the crawling will be performed.
    """

    def __init__(self,
                 path: str
                 ) -> None:
        super().__init__()

        self.path = check_is_dir_and_existing(os.path.normpath(path))

    @abstractmethod
    def execute(self) -> Any:
        """Executes the crawling process.

        Returns:
            Any: The crawled data.
        """
        raise NotImplementedError()


class DicomSubjectDirectoryCrawler(Crawler):
    """A crawler class to retrieve the data from a directory containing all DICOM files of a subject.

    Args:
        path (str): The subject directory path.
        modality_config_file_name (str): The file name for the modality configuration file within the subject
         directory (Default: modality_config.json).
        write_modality_config (bool): If True writes the modality configuration retrieved from the subject
         directory to the subject directory.
    """

    def __init__(self,
                 path: str,
                 modality_config_file_name: str = 'modality_config.json',
                 write_modality_config: bool = False) -> None:
        super().__init__(path)
        self.modality_config_file_name = modality_config_file_name
        self.write_modality_config = write_modality_config

    @staticmethod
    def _generate_image_infos(image_paths: Tuple[Tuple[str, ...], ...]) -> Tuple[DicomSeriesImageInfo, ...]:
        """Generates the DicomSeriesImageInfos from the DICOM files specified.

        Args:
            image_paths (Tuple[Tuple[str, ...], ...]): A tuple of tuples specifying the DicomSeriesImageInfos.

        Returns:
            Tuple[DicomSeriesImageInfo, ...]: A tuple with the retrieved DicomSeriesImageInfos.
        """
        infos = []

        for path_set in image_paths:
            image_info = DicomSeriesImageInfo(path_set)
            infos.append(image_info)

        return tuple(infos)

    @staticmethod
    def _generate_registration_infos(registration_paths: Tuple[str, ...],
                                     image_infos: Tuple[DicomSeriesImageInfo, ...]
                                     ) -> Tuple[DicomSeriesRegistrationInfo, ...]:
        """Generates the DicomSeriesRegistrationInfos from the DICOM files specified.

        Args:
            registration_paths (Tuple[str, ...]): A tuple of paths specifying the DicomSeriesRegistrationInfos.
            image_infos (Tuple[DicomSeriesImageInfo, ...]): A tuple of DicomSeriesImageInfos to generate the
             DicomSeriesRegistrationInfos.

        Returns:
            Tuple[DicomSeriesRegistrationInfo, ...]: A tuple with the retrieved DicomSeriesRegistrationInfo.
        """
        infos = []

        for path in registration_paths:
            registration_info = DicomSeriesRegistrationInfo(path, image_infos, persistent_image_infos=False)
            infos.append(registration_info)

        return tuple(infos)

    @staticmethod
    def _generate_rtss_info(rtss_paths: Tuple[str, ...]) -> Tuple[DicomSeriesRTStructureSetInfo, ...]:
        """Generates the DicomSeriesRTStructureSetInfos from the DICOM files specified.

        Args:
            rtss_paths (Tuple[str, ...]): A tuple of paths specifying the DicomSeriesRTStructureSetInfos.

        Returns:
            Tuple[DicomSeriesRTStructureSetInfo, ...]: A tuple with the retrieved DicomSeriesRTStructureSetInfos.
        """
        infos = []

        for path in rtss_paths:
            rtss_info = DicomSeriesRTStructureSetInfo(path)
            infos.append(rtss_info)

        return tuple(infos)

    def _export_modality_config(self,
                                infos: Tuple[DicomSeriesInfo, ...]
                                ) -> None:
        """Exports the retrieved ModalityConfiguration to a file.

        Args:
            infos (Tuple[DicomSeriesInfo, ...]): A tuple of DicomSeriesInfos from which the ModalityConfiguration is
             retrieved.

        Returns:
            None
        """
        config_output_path = os.path.join(self.path, 'modality_config.json')
        config = ModalityConfiguration.from_dicom_series_info(infos)
        config.to_file(config_output_path)

    def _apply_modality_config(self, infos: Tuple[DicomSeriesImageInfo, ...]) -> None:
        """Loads the ModalityConfiguration from a file and applies it to the specified DicomSeriesImageInfos.

        Args:
            infos (Tuple[DicomSeriesImageInfo, ...]): A tuple of DicomSeriesImageInfos to which the loaded
             ModalityConfiguration will be applied.

        Returns:
            None
        """
        modality_file_path = ''
        for root, _, files in os.walk(self.path):
            for file in files:
                if self.modality_config_file_name in file:
                    modality_file_path = os.path.join(root, file)
                    break

        if not os.path.exists(modality_file_path):
            return

        config = ModalityConfiguration.from_file(modality_file_path)
        config.add_modalities_to_info(infos)

    def execute(self) -> Tuple[DicomSeriesInfo, ...]:
        """Executes the crawling process to get the DicomSeriesInfos.

        Returns:
            Tuple[DicomSeriesInfo, ...]: A tuple containing the retrieved DicomSeriesInfos.
        """
        image_paths, registration_paths, rtss_paths = DicomCombinedDirectoryFilter(self.path).filter()

        image_infos = self._generate_image_infos(image_paths)
        registration_infos = self._generate_registration_infos(registration_paths, image_infos)
        rtss_infos = self._generate_rtss_info(rtss_paths)

        self._apply_modality_config(image_infos)

        if self.write_modality_config:
            self._export_modality_config(image_infos)

        return image_infos + registration_infos + rtss_infos


class DicomDatasetDirectoryCrawler(Crawler):
    """A crawler class to retrieve the data from multiple subject directories containing each DICOM files of a
    single subject.

    Args:
        path (str): The path to the base directory of the dataset.
        modality_config_file_name (str): The name of the modality configuration file saved in each subject
         directory if requested.
        write_modality_config (bool): If True the modality configuration files will be writen to each subject
         directory.
    """

    def __init__(self,
                 path: str,
                 modality_config_file_name: str = 'modality_config.json',
                 write_modality_config: bool = False) -> None:
        super().__init__(path)
        self.modality_config_file_name = modality_config_file_name
        self.write_modality_config = write_modality_config

    @staticmethod
    def _get_subject_dir_paths(path: str) -> Tuple[str, ...]:
        """Gets the paths of the subject directories containing DICOM files.

        Args:
            path (str): The base directory path which contain the subject directories.

        Returns:
            Tuple[str, ...]: A tuple containing all the subject directory paths.
        """
        dicom_dir_paths = []

        dir_paths = [entry.path for entry in os.scandir(path) if entry.is_dir()]

        for dir_path in dir_paths:
            if [entry for entry in os.scandir(dir_path) if entry.is_file() and entry.name.endswith('.dcm')]:
                dicom_dir_paths.append(os.path.normpath(dir_path))

        return tuple(dicom_dir_paths)

    def execute(self) -> Tuple[Tuple[DicomSeriesInfo, ...], ...]:
        """Executes the crawling process to get the DicomSeriesInfos for each subject directory separately.

        Returns:
            Tuple[Tuple[DicomSeriesInfo, ...], ...]: A tuple of tuples containing the retrieved DicomSeriesInfos.
        """
        subject_infos = []

        subject_dir_paths = self._get_subject_dir_paths(self.path)

        for subject_dir_path in subject_dir_paths:
            subject_info = DicomSubjectDirectoryCrawler(subject_dir_path,
                                                        self.modality_config_file_name,
                                                        self.write_modality_config).execute()

            if subject_info:
                subject_infos.append(subject_info)

        return tuple(subject_infos)


class IterableDicomDatasetDirectoryCrawler(DicomDatasetDirectoryCrawler):
    """An iterable crawler class to retrieve the data from multiple subject directories containing each DICOM files
    of a single subject.

    Args:
        path (str): The path to the base directory of the dataset.
        modality_config_file_name (str): The name of the modality configuration file saved in each subject
         directory if requested.
        write_modality_config (bool): If True the modality configuration files will be writen to each subject
         directory.
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
            crawler = DicomSubjectDirectoryCrawler(self.subject_dir_path[self.current_idx],
                                                   self.modality_config_file_name,
                                                   self.write_modality_config)
            subject_info = crawler.execute()

            self.current_idx += 1

            return subject_info

        raise StopIteration

    def __len__(self) -> int:
        return self.num_subjects
