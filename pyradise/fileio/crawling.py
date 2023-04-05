import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import itk
from pydicom.tag import Tag

from pyradise.data import Modality
from pyradise.utils import (assume_is_segmentation, is_dicom_file,
                            is_dir_and_exists, load_dataset_tag)

from .extraction import AnnotatorExtractor, ModalityExtractor, OrganExtractor
from .modality_config import ModalityConfiguration
from .series_info import (DicomSeriesImageInfo, DicomSeriesInfo,
                          DicomSeriesRegistrationInfo, DicomSeriesRTSSInfo,
                          FileSeriesInfo, IntensityFileSeriesInfo,
                          SegmentationFileSeriesInfo)

__all__ = ["Crawler", "SubjectFileCrawler", "DatasetFileCrawler", "SubjectDicomCrawler", "DatasetDicomCrawler"]


class Crawler(ABC):
    """An abstract crawler whose subtypes are intended to be used for searching files of a certain type in a specified
    location or within a hierarchy of directories.

    Args:
        path (str): The directory path for which the crawling will be performed.
    """

    def __init__(self, path: str) -> None:
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
    """A crawler for retrieving :class:`~pyradise.fileio.series_info.FileSeriesInfo` entries from a subject directory
    containing discrete image files of a specified type (see ``extension`` parameter).

    The :class:`SubjectFileCrawler` is used for searching appropriate files within a specific subject directory
    containing all the subject's data. If there are multiple subjects in separate directories but within a
    common top-level directory to be crawled we recommend using the :class:`DatasetFileCrawler`.

    Important:
        The DICOM format is not supported by this crawler. Use the appropriate crawler variant instead.

    Raises:
        ValueError: If the ``extension`` parameter specifies the DICOM file extension (i.e. ``.dcm``).

    Args:
        path (str): The directory path to crawl for files.
        subject_name (str): The name of the subject.
        extension (str): The file extension of the files to be searched.
        modality_extractor (ModalityExtractor): The modality extractor.
        organ_extractor (OrganExtractor): The organ extractor.
        annotator_extractor (AnnotatorExtractor): The annotator extractor.

    """

    def __init__(
        self,
        path: str,
        subject_name: str,
        extension: str,
        modality_extractor: ModalityExtractor,
        organ_extractor: OrganExtractor,
        annotator_extractor: AnnotatorExtractor,
    ) -> None:
        super().__init__(path)

        if "dcm" in extension:
            raise ValueError(
                f"The DICOM format is not supported by {self.__class__.__name__}! "
                "Use the appropriate DICOM variant instead."
            )

        self.extension = extension
        self.subject_name = subject_name
        self.modality_extractor = modality_extractor
        self.organ_extractor = organ_extractor
        self.annotator_extractor = annotator_extractor

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
                    modality = self.modality_extractor.extract(file_path)

                    if self.modality_extractor.is_enumerated_default_modality(modality):
                        is_segmentation = assume_is_segmentation(file_path)
                    else:
                        is_segmentation = True if modality is None else False

                    if is_segmentation:
                        organ = self.organ_extractor.extract(file_path)
                        annotator = self.annotator_extractor.extract(file_path)

                        series_info = SegmentationFileSeriesInfo(file_path, self.subject_name, organ, annotator)
                    else:
                        series_info = IntensityFileSeriesInfo(file_path, self.subject_name, modality)

                    series_infos.append(series_info)

        return tuple(series_infos)


class DatasetFileCrawler(Crawler):
    """An iterable crawler for retrieving :class:`~pyradise.fileio.series_info.FileSeriesInfo` entries from a dataset
    directory containing at least one subject directory with image files of a specified type (see ``extension``
    parameter).

    If you want to load a large dataset with many subjects, we recommend using the iterative crawling approach instead
    of crawling the data via :meth:`execute` to reduce memory consumption (see example below).

    Important:
        The DICOM format is not supported by this crawler. Use the appropriate crawler variant instead.

    Example:

        Demonstration of the iterative and the non-iterative loading approach:

        >>> from pyradise.data import (Modality, Organ, Annotator)
        >>> from pyradise.fileio import (DatasetFileCrawler, ModalityExtractor,
        >>>                              OrganExtractor, AnnotatorExtractor, SubjectLoader)
        >>>
        >>>
        >>> # An example modality extractor
        >>> class MyModalityExtractor(ModalityExtractor):
        >>>
        >>>     def extract_from_dicom(self, path: str) -> Optional[Modality]:
        >>>         return None
        >>>
        >>>     def extract_from_path(self, path: str) -> Optional[Modality]:
        >>>         file_name = os.path.basename(path)
        >>>         if 't1' in file_name:
        >>>             return Modality('T1')
        >>>         elif 't2' in file_name:
        >>>             return Modality('T2')
        >>>         else:
        >>>             return None
        >>>
        >>>
        >>> # An example organ extractor
        >>> class MyOrganExtractor(OrganExtractor):
        >>>
        >>>     def extract(self, path: str) -> Optional[Organ]:
        >>>         file_name = os.path.basename(path).lower()
        >>>         if 'brainstem' in file_name:
        >>>             return Organ('Brainstem')
        >>>         elif 'tumor' in file_name:
        >>>             return Organ('Tumor')
        >>>         else:
        >>>             return None
        >>>
        >>>
        >>> # An example annotator extractor
        >>> class MyAnnotatorExtractor(AnnotatorExtractor):
        >>>
        >>>     def extract(self, path: str) -> Optional[Annotator]:
        >>>         file_name = os.path.basename(path).lower()
        >>>         if 'example_expert' in file_name:
        >>>             return Annotator('ExampleExpert')
        >>>         return None
        >>>
        >>>
        >>> def main_iterative_crawling(dataset_path: str) -> None:
        >>>     extension = '.nii.gz'
        >>>
        >>>     # Create the crawler
        >>>     crawler = DatasetFileCrawler(dataset_path, extension, MyModalityExtractor(),
        >>>                                  MyOrganExtractor(), MyAnnotatorExtractor())
        >>>
        >>>     # Use the crawler iteratively (more memory efficient)
        >>>     for series_info in crawler:
        >>>         subject = SubjectLoader().load(series_info)
        >>>         # Do something with the subject
        >>>         print(subject.get_name())
        >>>
        >>>
        >>> def main_crawling_using_execute_fn(dataset_path: str) -> None:
        >>>     extension = '.nii.gz'
        >>>
        >>>     # Create the crawler
        >>>     crawler = DatasetFileCrawler(dataset_path, extension, MyModalityExtractor(),
        >>>                                  MyOrganExtractor(), MyAnnotatorExtractor())
        >>>
        >>>     # Use the crawler with the execute function
        >>>     # (all series info entries are loaded in one step)
        >>>     series_infos = crawler.execute()
        >>>
        >>>     # Iterate over the series infos
        >>>     for series_info in series_infos:
        >>>         subject = SubjectLoader().load(series_info)
        >>>         # Do something with the subject
        >>>         print(subject.get_name())

    Raises:
        ValueError: If the ``extension`` parameter specifies the DICOM file extension (i.e. ``.dcm``).

    Args:
        path (str): The dataset directory path to crawl for data.
        extension (str): The file extension of the image files to be crawled.
        modality_extractor (ModalityExtractor): The modality extractor.
        organ_extractor (OrganExtractor): The organ extractor.
        annotator_extractor (AnnotatorExtractor): The annotator extractor.
    """

    def __init__(
        self,
        path: str,
        extension: str,
        modality_extractor: ModalityExtractor,
        organ_extractor: OrganExtractor,
        annotator_extractor: AnnotatorExtractor,
    ) -> None:
        super().__init__(path)

        if "dcm" in extension:
            raise ValueError(
                f"The DICOM format is not supported by {self.__class__.__name__}! "
                "Use the appropriate DICOM variant instead."
            )
        self.extension = extension

        self.modality_extractor = modality_extractor
        self.organ_extractor = organ_extractor
        self.annotator_extractor = annotator_extractor

        subject_dir_paths = self._get_subject_dir_paths(self.path, self.extension)
        self.subject_dir_path = tuple(sorted(subject_dir_paths))
        self.subject_names = tuple(os.path.basename(path) for path in self.subject_dir_path)

        self.current_idx = 0
        self.num_subjects = len(self.subject_dir_path)

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
        # Get subject files
        subject_files = []
        for subject_dir, subject_name in zip(self.subject_dir_path, self.subject_names):
            subject_file_crawler = SubjectFileCrawler(
                subject_dir,
                subject_name,
                self.extension,
                self.modality_extractor,
                self.organ_extractor,
                self.annotator_extractor,
            )
            subject_files.append(subject_file_crawler.execute())

        return tuple(subject_files)

    def __iter__(self) -> "DatasetFileCrawler":
        self.current_idx = 0
        return self

    def __next__(self) -> Tuple[FileSeriesInfo, ...]:
        if self.current_idx < self.num_subjects:
            subject_info = SubjectFileCrawler(
                self.subject_dir_path[self.current_idx],
                self.subject_names[self.current_idx],
                self.extension,
                self.modality_extractor,
                self.organ_extractor,
                self.annotator_extractor,
            ).execute()
            self.current_idx += 1
            return subject_info
        else:
            raise StopIteration

    def __len__(self) -> int:
        return self.num_subjects


class SubjectDicomCrawler(Crawler):
    """A crawler for retrieving :class:`~pyradise.fileio.series_info.DicomSeriesInfo` entries from a subject directory
    containing DICOM files (e.g. DICOM images, DICOM registrations, DICOM RTSS). Files of other formats then DICOM will
    be ignored and can not be crawled with this type of crawler.

    The :class:`SubjectDicomCrawler` is used for searching appropriate files within a specific subject directory
    containing all the subject's data. If there are multiple subjects in separate directories but within a common
    top-level directory to be crawled we recommend using the :class:`DatasetDicomCrawler`.

    The prioritized method to extract the :class:`~pyradise.data.modality.Modality` for the retrieved images is the
    usage of a modality configuration file. If no modality configuration file is available the
    :class:`SubjectDicomCrawler` will try to extract the :class:`~pyradise.data.modality.Modality` from the retrieved
    images using the class:`ModalityExtractor`. If no :class:`~pyradise.fileio.extraction.ModalityExtractor` is
    provided an exception will be raised.

    The :class:`SubjectDicomCrawler` can be used to generate the modality configuration file skeleton for a
    specific subject. In this case set the ``generate_modality_config`` parameter to ``True`` and execute the
    crawling process. The generated modality configuration file skeleton will be saved in the subject directory.

    Important:
        This crawler exclusively support the DICOM file format and does not support any other file format.

    Args:
        path (str): The subject directory path to crawl.
        modality_extractor (Optional[ModalityExtractor]): The modality extractor (default: None).
        modality_config_file_name (str): The file name for the modality configuration file within the subject
         directory (default: modality_config.json).
        write_modality_config (bool): If True writes the modality configuration retrieved to the subject directory
         (default: False).
    """

    def __init__(
        self,
        path: str,
        modality_extractor: Optional[ModalityExtractor] = None,
        modality_config_file_name: str = "modality_config.json",
        write_modality_config: bool = False,
    ) -> None:
        super().__init__(path)
        self.modality_extractor: Optional[ModalityExtractor] = modality_extractor
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
                if file == "DICOMDIR":
                    continue

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

            # check if the image belongs to the DICOM RT SOP Classes
            dataset = load_dataset_tag(series_paths[0], (Tag(0x0008, 0x0016),))
            if "481" in str(dataset.get("SOPClassUID", "")):
                continue

            image_series_paths.append(tuple(series_paths))

        return tuple(image_series_paths)

    @staticmethod
    def _get_registration_files(paths: Tuple[str, ...]) -> Tuple[str, ...]:
        """Get all DICOM registration files in the subject directory.

        Args:
            paths (Tuple[str, ...]): The DICOM file paths to check if they specify a DICOM registration file.

        Returns:
            Tuple[str, ...]: The DICOM registration file paths.
        """
        valid_sop_class_uids = (
            "1.2.840.10008.5.1.4.1.1.66.1",  # Spatial Registration Storage
            "1.2.840.10008.5.1.4.1.1.66.3",
        )  # Deformable Spatial Registration Storage

        registration_files = []
        for path in paths:
            dataset = load_dataset_tag(path, (Tag(0x0008, 0x0016),))

            if dataset.get("SOPClassUID", None) in valid_sop_class_uids:
                registration_files.append(path)

        return tuple(registration_files)

    @staticmethod
    def _get_rtss_files(paths: Tuple[str, ...]) -> Tuple[str, ...]:
        """Get all DICOM RTSS files in the subject directory.

        Args:
            paths (Tuple[str, ...]): The DICOM file paths to check if they specify a DICOM RTSS file.

        Returns:
            Tuple[str, ...]: The DICOM RTSS file paths.
        """
        valid_sop_class_uid = "1.2.840.10008.5.1.4.1.1.481.3"  # RT Structure Set Storage

        rtss_files = []
        for path in paths:
            dataset = load_dataset_tag(path, (Tag(0x0008, 0x0016),))

            if dataset.get("SOPClassUID", None) == valid_sop_class_uid:
                rtss_files.append(path)

        return tuple(rtss_files)

    @staticmethod
    def _generate_image_infos(image_paths: Tuple[Tuple[str, ...], ...]) -> Tuple[DicomSeriesImageInfo]:
        """Generate the :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo` entries for the DICOM file paths
        specified.

        Args:
            image_paths (Tuple[Tuple[str, ...], ...]): The DICOM image file paths provided.

        Returns:
            Tuple[DicomSeriesImageInfo, ...]: The retrieved :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo`
            entries.
        """
        infos = []

        for paths in image_paths:
            image_info = DicomSeriesImageInfo(paths)
            infos.append(image_info)

        return tuple(infos)

    @staticmethod
    def _generate_registration_infos(
        registration_paths: Tuple[str, ...], image_infos: Tuple[DicomSeriesImageInfo, ...]
    ) -> Tuple[DicomSeriesRegistrationInfo]:
        """Generate the :class:`~pyradise.fileio.series_info.DicomSeriesRegistrationInfo` entries for the DICOM file
        paths specified.

        Args:
            registration_paths (Tuple[str, ...]): The DICOM registration file paths provided.
            image_infos (Tuple[DicomSeriesImageInfo, ...]): The available
             :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo` entries.

        Returns:
            Tuple[DicomSeriesRegistrationInfo, ...]: The retrieved
             :class:`~pyradise.fileio.series_info.DicomSeriesRegistrationInfo` entries.
        """
        infos = []

        for path in registration_paths:
            registration_info = DicomSeriesRegistrationInfo(path, image_infos, persistent_image_infos=False)
            infos.append(registration_info)

        return tuple(infos)

    @staticmethod
    def _generate_rtss_info(rtss_paths: Tuple[str, ...]) -> Tuple[DicomSeriesRTSSInfo]:
        """Generate the :class:`~pyradise.fileio.series_info.DicomSeriesRTStructureSetInfo` entries for the DICOM file
        paths specified.

        Args:
            rtss_paths (Tuple[str, ...]): The DICOM RTSS file paths.

        Returns:
            Tuple[DicomSeriesRTStructureSetInfo, ...]: AThe retrieved
             :class:`~pyradise.fileio.series_info.DicomSeriesRTStructureSetInfo` entries.
        """
        infos = []

        for path in rtss_paths:
            rtss_info = DicomSeriesRTSSInfo(path)
            infos.append(rtss_info)

        return tuple(infos)

    def _export_modality_config(self, infos: Tuple[DicomSeriesInfo, ...]) -> None:
        """Export the retrieved :class:`~pyradise.fileio.modality_config.ModalityConfiguration` to a file.

        Args:
            infos (Tuple[DicomSeriesInfo, ...]): The :class:`~pyradise.fileio.series_info.DicomSeriesInfo` entries
             containing the information to export.

        Returns:
            None
        """
        config = ModalityConfiguration.from_dicom_series_info(infos)
        config.to_file(os.path.join(self.path, self.config_file_name))

    def _apply_modality_config(self, infos: Tuple[DicomSeriesImageInfo, ...]) -> None:
        """Load the :class:`~pyradise.fileio.modality_config.ModalityConfiguration` from a file if available and apply
        it to the specified :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo` entries. If the
        :class:`~pyradise.fileio.modality_config.ModalityConfiguration` file is not available and a
        :class:`~pyradise.fileio.extraction.ModalityExtractor` is provided the extractor is used for modality
        determination.

        Args:
            infos (Tuple[DicomSeriesImageInfo, ...]): The available
             :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo` entries to which the
             loaded :class:`~pyradise.fileio.modality_config.ModalityConfiguration` can be applied.

        Returns:
            None
        """
        # try to apply the modality configuration if it exists
        modality_file_path = ""
        for root, _, files in os.walk(self.path):
            for file in files:
                if self.config_file_name in file:
                    modality_file_path = os.path.join(root, file)
                    break

        # apply the modality configuration if it exists
        if os.path.exists(modality_file_path):
            config = ModalityConfiguration.from_file(modality_file_path)
            config.add_modalities_to_info(infos)

            if config.has_default_modalities():
                warnings.warn("The modality configuration file contains at least one default modality.")

            if config.has_duplicate_modalities():
                raise ValueError(
                    "The modalities from the modality configuration file contain at least one duplicate "
                    "modality. This will cause ambiguity when loading the DICOM series."
                )

            if self.write_config:
                warnings.warn("The modality configuration file already exists and will not be overwritten.")

            return

        # if no modality configuration file exists, try to apply the default configuration
        else:
            if self.modality_extractor is not None:
                extraction_possible_for_all = True
                for info in infos:
                    modality = self.modality_extractor.extract(info.path[0])
                    if modality is not None:
                        info.set_modality(modality)
                    else:
                        info.set_modality(Modality.get_default())
                        extraction_possible_for_all = False

                config = ModalityConfiguration.from_dicom_series_info(infos)
                if self.write_config:
                    config.to_file(os.path.join(self.path, self.config_file_name))

                if config.has_duplicate_modalities():
                    raise ValueError(
                        "The extracted modalities contain at least one duplicate modality. "
                        "This will cause ambiguity when loading the DICOM series."
                    )

                if extraction_possible_for_all:
                    return
                else:
                    warnings.warn(
                        "Modality extraction failed for one DICOM series. The default modality will "
                        "be used for the series which failed during modality extraction."
                    )
                    return

            else:
                config = ModalityConfiguration.from_dicom_series_info(infos)

                if config.has_duplicate_modalities() and self.write_config is False:
                    raise ValueError(
                        "The extracted modalities contain at least one duplicate modality. "
                        "This will cause ambiguity when loading the DICOM series. Use either a modality "
                        "configuration file or a modality extractor to resolve this issue."
                    )

                if config.has_default_modalities() and self.write_config is False and len(config.configuration) > 1:
                    raise ValueError(
                        "The extracted modalities contain at least one default modality. "
                        "This will cause ambiguity when loading the DICOM series. Use either a modality "
                        "configuration file or a modality extractor to resolve this issue."
                    )

                if self.write_config:
                    config.to_file(os.path.join(self.path, self.config_file_name))
                    return

                if not config.has_duplicate_modalities():
                    return

                raise ValueError(
                    "The modality configuration file could not be found "
                    f"in the specified path ({self.path}) and there is no modality extractor provided!"
                )

    def execute(self) -> Tuple[DicomSeriesInfo, ...]:
        """Execute the crawling process to retrieve the :class:`~pyradise.fileio.series_info.DicomSeriesInfo` entries.

        Returns:
            Tuple[DicomSeriesInfo, ...]: The retrieved :class:`~pyradise.fileio.series_info.DicomSeriesInfo` entries.
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

        # apply the modality config and write it to disk if requested
        self._apply_modality_config(image_infos)

        return image_infos + registration_infos + rtss_infos


class DatasetDicomCrawler(Crawler):
    """A crawler for retrieving :class:`~pyradise.fileio.series_info.DicomSeriesInfo` entries from a dataset directory
    containing at least one subject directory with DICOM files (e.g. DICOM images, DICOM registrations, DICOM RTSS).
    Files of other formats then DICOM will be ignored and can not be crawled with this type of crawler.

    The :class:`DatasetDicomCrawler` is used for searching appropriate files within a specific dataset directory
    containing at least one subject folder with DICOM files. If there is just one subject in a single directory to be
    crawled we recommend using the :class:`SubjectDicomCrawler`. If you want to load a large dataset with many subjects,
    we recommend using the iterative crawling approach instead of crawling the data via :meth:`execute` to reduce
    memory consumption (see example below).

    The prioritized method to extract the :class:`~pyradise.data.modality.Modality` for the retrieved images is the
    usage of a modality configuration file. If no modality configuration file is available for a specific subject
    directory the :class:`DatasetDicomCrawler` will try to extract the :class:`~pyradise.data.modality.Modality` from
    the retrieved subject images using the :class:`~pyradise.fileio.extraction.ModalityExtractor`. If no
    :class:`~pyradise.fileio.extraction.ModalityExtractor` is provided an exception will be raised.

    The :class:`DatasetDicomCrawler` can be used to generate the modality configuration file skeletons for all
    subjects in the dataset directory. In this case set the ``generate_modality_config`` parameter to ``True`` and
    execute the crawling process. The generated modality configuration file skeletons will be saved in the appropriate
    subject directories.

    Important:
        This crawler exclusively support the DICOM file format and does not support any other file format.

    Example:

        Demonstration of the iterative and the non-iterative loading approach:

        >>> from pyradise.fileio import (DatasetDicomCrawler, SubjectLoader)
        >>>
        >>>
        >>> def main_iterative_crawling(dataset_path: str) -> None:
        >>>     # Create the crawler (using the modality configuration file)
        >>>     crawler = DatasetDicomCrawler(dataset_path)
        >>>
        >>>     # Use the crawler iteratively (more memory efficient)
        >>>     for series_info in crawler:
        >>>         subject = SubjectLoader().load(series_info)
        >>>         # Do something with the subject
        >>>         print(subject.get_name())
        >>>
        >>>
        >>> def main_crawling_using_execute_fn(dataset_path: str) -> None:
        >>>     # Create the crawler (using the modality configuration file)
        >>>     crawler = DatasetDicomCrawler(dataset_path)
        >>>
        >>>     # Use the crawler with the execute function
        >>>     # (all series info entries are loaded in one step)
        >>>     series_infos = crawler.execute()
        >>>
        >>>     # Iterate over the series infos
        >>>     for series_info in series_infos:
        >>>         subject = SubjectLoader().load(series_info)
        >>>         # Do something with the subject
        >>>         print(subject.get_name())

    Args:
        path (str): The dataset directory path to crawl.
        modality_extractor (Optional[ModalityExtractor]): The modality extractor (default: None)
        modality_config_file_name (str): The file name for the modality configuration file within the subject
         directory (default: modality_config.json).
        write_modality_config (bool): If True writes the modality configuration retrieved to the subject directory
         (default: False).
    """

    def __init__(
        self,
        path: str,
        modality_extractor: Optional[ModalityExtractor] = None,
        modality_config_file_name: str = "modality_config.json",
        write_modality_config: bool = False,
    ) -> None:
        super().__init__(path)
        self.modality_extractor: Optional[ModalityExtractor] = modality_extractor
        self.config_file_name = modality_config_file_name
        self.write_config = write_modality_config

        self.subject_dir_paths: Optional[str] = None

        self.current_idx = 0
        self.num_subjects = 0

    @staticmethod
    def _get_subject_dir_paths(path: str) -> Tuple[str, ...]:
        """Get the paths of the subject directories containing DICOM files.

        Args:
            path (str): The base directory path which contain the subject directories.

        Returns:
            Tuple[str, ...]: Paths to all subject directories containing DICOM files.
        """
        # Search for all dicom files and sort them according to their patient id
        subjects = {}
        patient_id_tag = Tag(0x0010, 0x0020)  # Patient ID

        for root, _, files in os.walk(path):
            for file in files:
                if file == "DICOMDIR":
                    continue

                file_path = os.path.join(root, file)

                # check if file is a dicom file
                if is_dicom_file(file_path):
                    # get the patient id
                    patient_id = str(load_dataset_tag(file_path, (patient_id_tag,)).get(patient_id_tag).value)

                    # collect the file paths per patient id
                    if patient_id not in subjects:
                        subjects[patient_id] = file_path
                    else:
                        common_path = os.path.commonpath([subjects.get(patient_id), file_path])
                        subjects[patient_id] = common_path

        return tuple(sorted(subjects.values()))

    def execute(self) -> Tuple[Tuple[DicomSeriesInfo, ...], ...]:
        """Execute the crawling process to retrieve the :class:`~pyradise.fileio.series_info.DicomSeriesInfo` entries.

        Returns:
            Tuple[Tuple[DicomSeriesInfo, ...], ...]: The retrieved :class:`~pyradise.fileio.series_info.DicomSeriesInfo`
             entries.
        """
        self.subject_dir_paths = self._get_subject_dir_paths(self.path)

        subject_infos = []
        for subject_dir_path in self.subject_dir_paths:
            subject_info = SubjectDicomCrawler(
                subject_dir_path, self.modality_extractor, self.config_file_name, self.write_config
            ).execute()

            subject_infos.append(subject_info) if subject_info else None

        return tuple(subject_infos)

    def __iter__(self) -> "DatasetDicomCrawler":
        self.subject_dir_paths = self._get_subject_dir_paths(self.path)
        self.num_subjects = len(self.subject_dir_paths)
        self.current_idx = 0
        return self

    def __next__(self) -> Tuple[DicomSeriesInfo, ...]:
        if self.current_idx < self.num_subjects:
            subject_info = SubjectDicomCrawler(
                self.subject_dir_paths[self.current_idx],
                self.modality_extractor,
                self.config_file_name,
                self.write_config,
            ).execute()
            self.current_idx += 1
            return subject_info

        raise StopIteration

    def __len__(self) -> int:
        return self.num_subjects
