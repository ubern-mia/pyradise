from abc import (
    ABC,
    abstractmethod)
from typing import (
    Any,
    Optional,
    Dict)
from pathlib import Path

from pyradise.curation.data import (
    Modality,
    Organ,
    Rater)
from .definitions import (
    PATH,
    MODALITY,
    ORGAN,
    RATER,
    FILENAME)


# pylint: disable=too-many-branches, too-many-return-statements


class FileFilter(ABC):
    """Abstract base class for a file filter."""

    @abstractmethod
    def filter(self,
               path: str,
               *args,
               **kwargs
               ) -> Any:
        """Performs the filtering.

        Args:
            path (str): The file path to filter.
            *args: n/a
            **kwargs: n/a

        Returns:
            Any: The filtered data.
        """
        raise NotImplementedError()


class AnyFileFilter(FileFilter):
    """A filter class returning all files in a directory."""

    def filter(self,
               path: str,
               *args,
               **kwargs) -> Dict[str, str]:
        """Filters the specified path by returning all the information.

        Args:
            path (str): The file path to filter.
            *args: n/a
            **kwargs: n/a

        Returns:
            Dict[str, str]: The data extracted from the path.
        """
        internal_path = Path(path)

        return {PATH: str(internal_path),
                FILENAME: str(internal_path.name)}


class ImageTransformFileFilter(FileFilter):
    """A filter class returning all transform files for images in a directory."""

    def filter(self,
               path: str,
               *args,
               **kwargs) -> Optional[Dict[str, str]]:
        """Filters the specified path by returning all the transform files for images.

        Args:
            path (str): The file path to filter.
            *args: n/a
            **kwargs: n/a

        Returns:
            Dict[str, str]: The data extracted from the path.
        """
        internal_path = Path(path)
        file_name = internal_path.name

        if 'tfm_' not in file_name:
            return None

        if '_T1c_' in file_name:
            return {PATH: str(internal_path),
                    FILENAME: str(file_name),
                    MODALITY: Modality.T1c}

        if '_T1w_' in file_name:
            return {PATH: str(internal_path),
                    FILENAME: str(file_name),
                    MODALITY: Modality.T1w}

        if '_T2w_' in file_name:
            return {PATH: str(internal_path),
                    FILENAME: str(file_name),
                    MODALITY: Modality.T2w}

        if '_FLAIR_' in file_name:
            return {PATH: str(internal_path),
                    FILENAME: str(file_name),
                    MODALITY: Modality.FLAIR}

        if '_CT_' in file_name:
            return {PATH: str(internal_path),
                    FILENAME: str(file_name),
                    MODALITY: Modality.CT}

        return None


class ImagingDirectoryFilter(FileFilter):
    """A filter class for image paths."""

    @staticmethod
    def filter(path: str,
               *args,
               **kwargs
               ) -> Optional[Dict[str, Any]]:
        """Filters the specified image path.

        Args:
            path (str): The file path to filter.
            *args: n/a
            **kwargs: n/a

        Returns:
            Optional[Dict[str, Any]]: The data extracted from the path.
        """
        internal_path = Path(path)
        file_name = internal_path.name

        if 'T1c.nii.gz' in file_name:
            return {MODALITY: Modality.T1c,
                    PATH: str(internal_path)}
        if 'T1w.nii.gz' in file_name:
            return {MODALITY: Modality.T1w,
                    PATH: str(internal_path)}
        if 'T2w.nii.gz' in file_name:
            return {MODALITY: Modality.T2w,
                    PATH: str(internal_path)}
        if 'FLAIR.nii.gz' in file_name:
            return {MODALITY: Modality.FLAIR,
                    PATH: str(internal_path)}
        if 'CT.nii.gz' in file_name:
            return {MODALITY: Modality.CT,
                    PATH: str(internal_path)}

        return None


class SegmentationDirectoryFilter(FileFilter):
    """A filter class for segmentation file paths."""

    @staticmethod
    def get_rater_from_file_name(file_name: str) -> Rater:
        """Gets the rater from a file name.

        Args:
            file_name (str): The file name from which the extraction should happen.

        Returns:
            Rater: The retrieved rater.
        """
        return Rater(file_name.split('_')[4])

    @abstractmethod
    def filter(self,
               path: str,
               *args,
               **kwargs
               ) -> Any:
        """Filters the specified segmentation image path.

        Args:
            path (str): The file path to filter.
            *args: n/a
            **kwargs: n/a

        Returns:
            Optional[Dict[str, Any]]: The data extracted from the path.
        """
        raise NotImplementedError()


class OARDirectoryFilter(ImagingDirectoryFilter, SegmentationDirectoryFilter):
    """A filter class for OAR file paths"""

    def filter(self,
               path: str,
               *args,
               **kwargs
               ) -> Optional[Dict[str, Any]]:
        """Filters the specified image path.

        Args:
            path (str): The file path to filter.
            *args: n/a
            **kwargs: n/a

        Returns:
            Optional[Dict[str, Any]]: The data extracted from the path.
        """
        info = ImagingDirectoryFilter.filter(path)

        if info is not None:
            return info

        internal_path = Path(path)
        file_name = internal_path.name

        if 'Brainstem.nii.gz' in file_name:
            return {ORGAN: Organ('Brainstem', 1),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Cochlea_L.nii.gz' in file_name:
            return {ORGAN: Organ('Cochlea_L', 2),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Cochlea_R.nii.gz' in file_name:
            return {ORGAN: Organ('Cochlea_R', 3),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Eye_L.nii.gz' in file_name:
            return {ORGAN: Organ('Eye_L', 4),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Eye_R.nii.gz' in file_name:
            return {ORGAN: Organ('Eye_R', 5),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Hippocampus_L.nii.gz' in file_name:
            return {ORGAN: Organ('Hippocampus_L', 6),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Hippocampus_R.nii.gz' in file_name:
            return {ORGAN: Organ('Hippocampus_R', 7),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Lacrimal_L.nii.gz' in file_name:
            return {ORGAN: Organ('Lacrimal_L', 8),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Lacrimal_R.nii.gz' in file_name:
            return {ORGAN: Organ('Lacrimal_R', 9),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Lens_L.nii.gz' in file_name:
            return {ORGAN: Organ('Lens_L', 10),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Lens_R.nii.gz' in file_name:
            return {ORGAN: Organ('Lens_R', 11),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'OpticChiasm.nii.gz' in file_name:
            return {ORGAN: Organ('OpticChiasm', 12),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'OpticNerve_L.nii.gz' in file_name:
            return {ORGAN: Organ('OpticNerve_L', 13),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'OpticNerve_R.nii.gz' in file_name:
            return {ORGAN: Organ('OpticNerve_R', 14),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Pituitary.nii.gz' in file_name:
            return {ORGAN: Organ('Pituitary', 15),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Retina_L.nii.gz' in file_name:
            return {ORGAN: Organ('Retina_L', 16),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Retina_R.nii.gz' in file_name:
            return {ORGAN: Organ('Retina_R', 17),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}

        return None


class CavityDirectoryFilter(ImagingDirectoryFilter, SegmentationDirectoryFilter):
    """A filter class for Cavity image paths."""

    def filter(self,
               path: str,
               *args,
               **kwargs
               ) -> Optional[Dict[str, Any]]:
        """Filters the specified image path.

        Args:
            path (str): The file path to filter.
            *args: n/a
            **kwargs: n/a

        Returns:
            Optional[Dict[str, Any]]: The data extracted from the path.
        """
        info = ImagingDirectoryFilter.filter(path)

        if info is not None:
            return info

        internal_path = Path(path)
        file_name = internal_path.name

        if 'Resection_Cavity.nii.gz' in file_name:
            return {ORGAN: Organ('Resection_Cavity', 1),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Enhancing_Tumor.nii.gz' in file_name:
            return {ORGAN: Organ('Enhancing_Tumor', 2),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Edema.nii.gz' in file_name:
            return {ORGAN: Organ('Edema', 3),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Air.nii.gz' in file_name:
            return {ORGAN: Organ('Air', 4),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Blood.nii.gz' in file_name:
            return {ORGAN: Organ('Blood', 5),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'GTVp.nii.gz' in file_name:
            return {ORGAN: Organ('GTVp', 6),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}
        if 'Tumor.nii.gz' in file_name:
            return {ORGAN: Organ('Tumor', 7),
                    RATER: self.get_rater_from_file_name(file_name),
                    PATH: str(internal_path)}

        return None
