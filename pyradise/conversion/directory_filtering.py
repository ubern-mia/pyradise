from abc import (
    ABC,
    abstractmethod)
from typing import (
    Any,
    Tuple,
    Optional)
import os
import mimetypes

import itk
from pydicom import Dataset
from pydicom.tag import Tag

from .utils import (
    check_is_dir_and_existing,
    load_dataset_tag)


class DirectoryFilter(ABC):
    """An abstract directory filter class."""

    def __init__(self,
                 path: str
                 ) -> None:
        """
        Args:
            path (str):  The directory path of the directory which will be filtered.
        """
        super().__init__()
        check_is_dir_and_existing(path)

        self.path = path

    @abstractmethod
    def filter(self) -> Any:
        """Filters the specified directory.

        Returns:
            Any: The filtered data.
        """
        raise NotImplementedError()


class DicomDirectoryFilter(DirectoryFilter, ABC):
    """An abstract directory filter class to retrieve valid DICOM files."""

    @staticmethod
    def get_dicom_file_paths(path: str) -> Tuple[str, ...]:
        """Gets the DICOM file paths from the specified directory and its possible subdirectories.

        Args:
            path (str): The directory path from which to get the DICOM file paths.

        Returns:
            Tuple[str, ...]: A tuple with the DICOM file paths.
        """
        check_is_dir_and_existing(path)

        files_paths = []
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                if mimetypes.guess_type(file_path)[0] == 'application/dicom':
                    files_paths.append(file_path)

        return tuple(files_paths)

    @staticmethod
    def is_valid_sop_class_uid(path: str,
                               valid_sop_class_uids: Tuple[str, ...],
                               dataset: Optional[Dataset] = None
                               ) -> bool:
        """Checks if a specified Dataset contains a SOPClassUID which is in a collection of SOPClassUIDs.

        Args:
            path (str): The path to the DICOM file.
            valid_sop_class_uids (Tuple[str, ...]: A tuple of valid SOPClassUIDs.
            dataset (Dataset): The Dataset if already loaded.

        Returns:
            bool: True if the SOPClassUID is valid, otherwise False.
        """
        if dataset is None:
            dataset = load_dataset_tag(path, (Tag(0x0008, 0x0016),))

        if dataset.SOPClassUID in valid_sop_class_uids:
            return True

        return False

    def get_image_series_extractor(self) -> itk.GDCMSeriesFileNames:
        """Gets the GDCMSeriesFileNames extractor which can sort and separate DICOM image file paths.

        Returns:
            itk.GDCMSeriesFileNames: The GDCMSeriesFileNames extractor.
        """
        series_info_extractor = itk.GDCMSeriesFileNames.New()
        series_info_extractor.SetRecursive(True)
        series_info_extractor.SetDirectory(self.path)
        series_info_extractor.Update()
        return series_info_extractor

    @abstractmethod
    def filter(self) -> Any:
        """Filters the specified directory for DICOM files.

        Returns:
            Any: The filtered data.
        """
        raise NotImplementedError()


class DicomCombinedDirectoryFilter(DicomDirectoryFilter):
    """A directory filter class to retrieve DICOM images, DICOM registrations, and DICOM RTSTRUCTS."""

    def filter(self) -> Tuple[Tuple[Tuple[str, ...], ...], Tuple[str, ...], Tuple[str, ...]]:
        """Filters the specified directory for DICOM image, DICOM registration, and DICOM RTSTRUCT files.

        Returns:
            Tuple[Tuple[Tuple[str, ...], ...], Tuple[str, ...], Tuple[str, ...]]: The file paths.
        """
        dir_dicom_file_paths = self.get_dicom_file_paths(self.path)

        # Get the image files
        image_series_info_extractor = self.get_image_series_extractor()
        image_series_paths = []

        for image_series_uid in image_series_info_extractor.GetSeriesUIDs():
            paths = image_series_info_extractor.GetFileNames(image_series_uid)
            normed_paths = [os.path.normpath(path) for path in paths]
            image_series_paths.append(tuple(normed_paths))

        # Separate the remaining SOPClasses
        flatten_image_series_paths = [path for paths in image_series_paths for path in paths]
        non_image_dicom_paths = set(dir_dicom_file_paths).difference(set(flatten_image_series_paths))

        registration_file_paths = []
        rtss_file_paths = []

        for dicom_file_path in non_image_dicom_paths:
            dataset = load_dataset_tag(dicom_file_path, (Tag(0x0008, 0x0016),))

            sop_class_uid = dataset.get('SOPClassUID', None)

            if sop_class_uid in ('1.2.840.10008.5.1.4.1.1.66.1',  # Spatial Registration Storage
                                 '1.2.840.10008.5.1.4.1.1.66.3'):  # Deformable Spatial Registration Storage
                registration_file_paths.append(dicom_file_path)

            elif sop_class_uid == '1.2.840.10008.5.1.4.1.1.481.3':  # RT Structure Set Storage
                rtss_file_paths.append(dicom_file_path)

        return tuple(image_series_paths), tuple(registration_file_paths), tuple(rtss_file_paths)


class DicomImageDirectoryFilter(DicomDirectoryFilter):
    """A directory filter class to retrieve valid DICOM image files."""

    def filter(self) -> Tuple[Tuple[str, ...], ...]:
        """Filters the specified directory for DICOM image files.

        Returns:
            Tuple[Tuple[str, ...], ...]: The DICOM image file paths per DICOM series.
        """
        series_file_paths = []

        valid_sop_class_uids = ('1.2.840.10008.5.1.4.1.1.1',  # Computed Radiography Image Storage
                                '1.2.840.10008.5.1.4.1.1.2',  # CT Image Storage
                                '1.2.840.10008.5.1.4.1.1.4',  # MR Image Storage
                                '1.2.840.10008.5.1.4.1.1.4.1',  # Enhanced MR Image Storage
                                '1.2.840.10008.5.1.4.1.1.481.1')  # RT Image Storage

        series_info_extractor = self.get_image_series_extractor()
        series_uids = series_info_extractor.GetSeriesUIDs()

        for serie_uid in series_uids:
            paths = series_info_extractor.GetFileNames(serie_uid)

            if not DicomDirectoryFilter.is_valid_sop_class_uid(paths[0], valid_sop_class_uids):
                continue

            series_file_paths.append(tuple(paths))

        return tuple(series_file_paths)


class DicomRegistrationDirectoryFilter(DicomDirectoryFilter):
    """A directory filter class to retrieve valid DICOM registration files."""

    def filter(self) -> Tuple[str, ...]:
        """Filters the specified directory for DICOM image files.

        Returns:
            Tuple[str, ...]: The DICOM registration file paths.
        """
        registration_paths = []

        valid_sop_class_uids = ('1.2.840.10008.5.1.4.1.1.66.1',  # Spatial Registration Storage
                                '1.2.840.10008.5.1.4.1.1.66.3')  # Deformable Spatial Registration Storage

        file_paths = self.get_dicom_file_paths(self.path)

        for path in file_paths:
            if DicomDirectoryFilter.is_valid_sop_class_uid(path, valid_sop_class_uids):
                registration_paths.append(path)

        return tuple(registration_paths)


class DicomRTStructureSetDirectoryFilter(DicomDirectoryFilter):
    """A directory filter class to retrieve valid DICOM RT Structure Set files."""

    def filter(self) -> Tuple[str, ...]:
        """Filters the specified directory for DICOM RT Structure Set files.

        Returns:
            Tuple[str, ...]: The DICOM RT Structure Set file paths.
        """
        rtss_paths = []

        valid_sop_class_uids = ('1.2.840.10008.5.1.4.1.1.481.3',)  # RT Structure Set Storage

        file_paths = self.get_dicom_file_paths(self.path)

        for path in file_paths:
            if DicomDirectoryFilter.is_valid_sop_class_uid(path, valid_sop_class_uids):
                rtss_paths.append(path)

        return tuple(rtss_paths)
