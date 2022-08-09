import os
from typing import (
    Tuple,
    Union,
    Dict,
    Type,
    Any)

import numpy as np
import SimpleITK as sitk
from pymia.data.conversion import ImageProperties
from pymia.data.subjectfile import SubjectFile
from pymia.data.creation import (
    Load,
    Traverser,
    get_default_callbacks,
    Hdf5Writer)
from pymia.data.transformation import (
    Transform,
    IntensityNormalization)

from pyradise.curation.data import (
    Modality,
    OrganRaterCombination)


FileKeyType = Union[Modality, OrganRaterCombination]

DEFAULT_IMAGE_FILE_KEYS = (Modality.T1c, Modality.T1w, Modality.T2w, Modality.FLAIR)
DEFAULT_LABEL_FILE_KEYS = (OrganRaterCombination('all', 'Combination'),)
DEFAULT_FILE_KEYS = DEFAULT_IMAGE_FILE_KEYS + DEFAULT_LABEL_FILE_KEYS


class SubjectFileLoader(Load):
    """A class for loading subject files."""

    def __call__(self,
                 file_name: str,
                 id_: str,
                 category: str,
                 subject_id: str
                 ) -> Tuple[np.ndarray, Union[ImageProperties, None]]:
        """Loads the appropriate data.

        Args:
            file_name (str): The file path to the file.
            id_ (str): The subject name (Maybe wrong!).
            category (str): The category.
            subject_id (str): The subject name.

        Returns:
            Tuple[np.ndarray,  Union[ImageProperties, None]]: The image array and properties if available.
        """
        if category == 'images':
            image = sitk.ReadImage(file_name, sitk.sitkFloat32)

        elif category == 'labels':
            image = sitk.ReadImage(file_name, sitk.sitkUInt8)

        else:
            raise FileNotFoundError(f'The file {file_name} can not be found!')

        properties = ImageProperties(image)
        return sitk.GetArrayFromImage(image), properties


class FilePathGenerator:
    """A class for generating file paths."""

    def __init__(self,
                 label_keys: Tuple[OrganRaterCombination, ...],
                 image_keys: Tuple[Modality, ...] = (Modality.T1c, Modality.T1w, Modality.T2w, Modality.FLAIR),
                 ) -> None:
        """Constructs a file path generator.

        Args:
            label_keys (Tuple[OrganRaterCombination, ...]): The keys of the labels.
            image_keys (Tuple[Modality, ...]): The images keys (default=(Modality.T1c, Modality.T1w, Modality.T2w,
             Modality.FLAIR)).
        """
        super().__init__()
        self.image_keys = image_keys
        self.label_keys = label_keys
        self.file_keys_ = self.image_keys + self.label_keys

    @property
    def file_keys(self) -> Tuple[FileKeyType, ...]:
        """Get all the file keys.

        Returns:
            Tuple[FileKeyType, ...]: The combined file keys.
        """
        return self.file_keys_

    def get_full_path(self,
                      subject: str,
                      subject_directory_path: str,
                      file_key: FileKeyType
                      ) -> str:
        """Get the full file path.

        Args:
            subject (str): The subject name.
            subject_directory_path (str): The subject directory path.
            file_key (FileKeyType): The file key.

        Returns:
            str: The full file path.
        """
        if file_key in self.image_keys:
            file_name = f'img_{subject}_{file_key.name}.nii.gz'

        else:
            file_name = f'seg_{subject}_{file_key.name}.nii.gz'

        return os.path.join(subject_directory_path, file_name)


class FileSystemCrawler:
    """A class for crawling the file system for data."""

    def __init__(self,
                 dataset_base_dir: str,
                 file_path_generator: FilePathGenerator,
                 valid_file_extensions: Tuple[str, ...] = ('.nii.gz',)
                 ) -> None:
        """Constructs a file system crawler.

        Args:
            dataset_base_dir (str): The path to the base directory of the dataset.
            file_path_generator (FilePathGenerator): The file path generator to generate the full paths.
            valid_file_extensions (Tuple[str, ...]): All valid file extensions (default=('.nii.gz',)).
        """
        super().__init__()

        if not os.path.exists(dataset_base_dir):
            raise FileNotFoundError(f'The directory path {dataset_base_dir} is invalid!')

        if not os.path.isdir(dataset_base_dir):
            raise NotADirectoryError(f'The path {dataset_base_dir} is not a directory!')

        self.dataset_base_dir = dataset_base_dir

        self.file_path_generator = file_path_generator
        self.valid_file_extensions = valid_file_extensions

    def _is_valid_file(self,
                       file_name: str
                       ) -> bool:
        extension_criterion = any(extension in file_name for extension in self.valid_file_extensions)

        file_key_criterion = any(file_key.name in file_name for file_key in self.file_path_generator.file_keys)

        return all((extension_criterion, file_key_criterion))

    def _get_subject_directories(self) -> Dict[str, str]:
        candidates = [entry for entry in os.scandir(self.dataset_base_dir) if entry.is_dir()]

        subject_directories = {}

        for candidate in candidates:
            files = [entry for entry in os.scandir(candidate.path) if self._is_valid_file(entry.name)]

            if len(files) == len(self.file_path_generator.file_keys):
                subject_directories.update({candidate.name: candidate.path})

        if not subject_directories:
            raise ValueError('No valid subject could be found! Please check your file names.')

        return subject_directories

    def crawl(self) -> Tuple[Dict[str, str], ...]:
        """Crawl the information about the dataset.

        Returns:
            Tuple[Dict[str, str], ...]: The crawled information about the dataset.
        """
        subject_infos = self._get_subject_directories()

        subject_names = list(subject_infos.keys())
        subject_names.sort()

        dataset_data = []

        for subject_name in subject_names:
            subject_path = subject_infos.get(subject_name)

            subject_data = {'subject_name': subject_name, 'subject_path': subject_path}

            for file_key in self.file_path_generator.file_keys:
                file_path = self.file_path_generator.get_full_path(subject_name, subject_path, file_key)

                if not os.path.exists(file_path):
                    raise ValueError(f'The file path {file_path} is invalid!')

                subject_data[file_key] = file_path

            dataset_data.append(subject_data)

        return tuple(dataset_data)


class SimpleSubjectFile(SubjectFile):
    """A class representing a subject with a flexible interface."""

    def __init__(self,
                 subject: str,
                 files: dict,
                 label_identifiers: Dict[str, Any],
                 image_identifiers: Dict[str, Any]
                 ) -> None:
        """Constructs a new subject.

        Args:
            subject (str): The subject name.
            files (dict): A dictionary containing the file paths.
            label_identifiers (Dict[str, Any]): The identifiers for the labels.
            image_identifiers (Dict[str, Any]): The identifiers for the images.
        """
        if not label_identifiers:
            raise ValueError('No image label identifiers provided!')

        if not image_identifiers:
            raise ValueError('No image identifiers provided!')

        label_files = {key: files.get(value) for key, value in label_identifiers.items()}
        image_files = {key: files.get(value) for key, value in image_identifiers.items()}

        super().__init__(subject=subject,
                         images=image_files,
                         labels=label_files)
        self.subject_path = files.get('subject_path', '')


class FileSystemDatasetCreator:
    """A class representing a dataset creator for HDF5 files which contains data from the file system."""

    # pylint: disable=too-many-arguments
    def __init__(self,
                 dataset_directory_path: str,
                 output_file_path: str,
                 file_path_generator: FilePathGenerator,
                 label_identifiers: Dict[str, Any],
                 image_identifiers: Dict[str, Any],
                 crawler_type: Type[FileSystemCrawler] = FileSystemCrawler,
                 subject_type: Type[SimpleSubjectFile] = SimpleSubjectFile,
                 valid_file_extensions: Tuple[str, ...] = ('.nii.gz',),
                 transform: Transform = IntensityNormalization()
                 ) -> None:
        """Constructs the file system dataset creator.

        Args:
            dataset_directory_path (str): The work path to the dataset base.
            output_file_path (str): The path to the output database file.
            file_path_generator (FilePathGenerator): The file path generator.
            label_identifiers (Dict[str, Any]): The label identifier and its label.
            image_identifiers (Dict[str, Any]): The image identifiers and its modalities.
            crawler_type (Type[FileSystemCrawler]): The crawler type (default=FileSystemCrawler).
            subject_type (Type[SimpleSubjectFile]): The subject type (default=SimpleSubjectFile).
            valid_file_extensions (Tuple[str, ...]: All valid file extensions (default=('.nii.gz',)).
            transform (Transform): The transform to apply before constructing the dataset.

        Examples:
            Usage example:
            >>> input_path = 'C:/YOUR/PATH/TO/THE/DATASET/BASE/DIRECTORY'
            >>> output_path = 'C:/YOUR/PATH/TO/THE/DATABASE_FILE.h5'
            >>>
            >>> lbl_identifiers = {'LB': OrganRaterCombination('GTVm', 'Combination')}
            >>>
            >>> generator = FilePathGenerator(tuple(label_identifiers.values()))
            >>> creator = FileSystemDatasetCreator(input_path, output_path, generator,
            >>>                                    label_identifiers=lbl_identifiers)
            >>> creator.create()
        """
        super().__init__()

        self.dataset_directory_path = dataset_directory_path
        self.output_file_path = output_file_path

        self.crawler = crawler_type(dataset_directory_path, file_path_generator, valid_file_extensions)

        self.subject_type = subject_type
        self.label_identifiers = label_identifiers
        self.image_identifiers = image_identifiers
        self.transform = transform

    def create(self) -> None:
        """Creates a HDF5 dataset.

        Returns:
            None
        """
        # pylint: disable=abstract-class-instantiated

        crawled_data = self.crawler.crawl()

        subjects = []

        for subject_data in crawled_data:
            subject = self.subject_type(subject_data.get('subject_name'), subject_data, self.label_identifiers,
                                        self.image_identifiers)
            subjects.append(subject)

        with Hdf5Writer(self.output_file_path) as writer:
            callbacks = get_default_callbacks(writer)
            traverser = Traverser()
            traverser.traverse(subjects, SubjectFileLoader(), callbacks, self.transform)


# usage example
# if __name__ == '__main__':
#
#     input_path_ = 'D:/DataBackupsConversion/20220207_ISAS_METPO_curation_MNI_152'
#     output_path_ = 'D:/test.h5'
#
#     label_identifiers_ = {'LB': OrganRaterCombination('GTVm', 'Combination')}
#     image_identifiers_ = {'CT': Modality.CT}
#
#     file_path_generator_ = FilePathGenerator(label_keys=tuple(label_identifiers_.values()),
#                                              image_keys=tuple(image_identifiers_.values()))
#     creator_ = FileSystemDatasetCreator(input_path_, output_path_, file_path_generator_,
#                                         label_identifiers_, image_identifiers_)
#     creator_.create()
