from typing import (
    Union,
    Optional,
    Tuple,
    TypeVar)
import os
from enum import Enum
from re import sub
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
from io import BytesIO

import SimpleITK as sitk
import itk
from pydicom import Dataset

from pyradise.curation.data import (
    Subject,
    IntensityImage,
    SegmentationImage)

DicomSeriesInfo = TypeVar('DicomSeriesInfo')

ILLEGAL_FOLDER_CHARS = "[<>:/\\|?*\"]|[\0-\31]"


def remove_illegal_folder_characters(name: str) -> str:
    """Removes illegal characters from a folder name.

    Args:
        name (str): The folder name with potential illegal characters.

    Returns:
        str: The folder name without illegal characters.
    """
    return sub(ILLEGAL_FOLDER_CHARS, "", name)


class ImageFileFormat(Enum):
    """An enum class representing all image format types."""
    NIFTI = '.nii'
    NIFTI_GZ = '.nii.gz'
    NRRD = '.nrrd'


class SubjectWriter:
    """A class for writing a subject to a directory."""

    def __init__(self,
                 path: str,
                 image_file_format: ImageFileFormat = ImageFileFormat.NIFTI_GZ,
                 allow_override: bool = False
                 ) -> None:
        """Constructs a subject writer.

        Args:
            path (str): The path to the subject directory.
            image_file_format (ImageFileFormat): The output file format (default=ImageFileFormat.NIFTI_GZ).
            allow_override (bool): If true the writer can overwrite existing files (default=False).
        """
        super().__init__()
        self.path = path
        self.image_file_format = image_file_format
        self.allow_override = allow_override

    @staticmethod
    def _generate_base_file_name(subject: Subject,
                                 image: Union[IntensityImage, SegmentationImage]
                                 ) -> str:
        if isinstance(image, IntensityImage):
            return f'{subject.name}_{image.get_modality(as_str=True)}'

        if isinstance(image, SegmentationImage):
            organ = image.get_organ()
            if image.get_rater() is None:
                rater_name = 'unknown'

            else:
                rater_name = image.get_rater().name

            return f'{subject.name}_{rater_name}_{organ.name}'

        raise TypeError(f'The image type ({type(image)}) is not supported!')

    def generate_image_file_name(self,
                                 subject: Subject,
                                 image: Union[IntensityImage, SegmentationImage],
                                 with_extension: bool = False
                                 ) -> str:
        """Generate an image file name.

        Args:
            subject (Subject): The subject of the image.
            image (Union[IntensityImage, SegmentationImage]): The image for which the file name should be generated.
            with_extension (bool): If True adds the file extension to the file name otherwise not.

        Returns:
            str: The file name of the image file.
        """
        if isinstance(image, IntensityImage):
            file_name = 'img_' + self._generate_base_file_name(subject, image)

        else:
            file_name = 'seg_' + self._generate_base_file_name(subject, image)

        if with_extension:
            return file_name + self.image_file_format.value

        return file_name

    def generate_transform_file_name(self,
                                     subject: Subject,
                                     image: Union[IntensityImage, SegmentationImage],
                                     index: Union[int, str],
                                     extension: Optional[str] = '.tfm'
                                     ) -> str:
        """Generate a transformation file name.

        Args:
            subject (Subject): The subject where the transformation belongs to.
            image (Union[IntensityImage, SegmentationImage]): The image to which the transformation belongs to.
            index (Union[int, str]): The index of the transformation.
            extension (str): The file extension for the transformation file.

        Returns:
            str: The file name of the transformation file.
        """
        if extension is not None:
            return f'tfm_{self._generate_base_file_name(subject, image)}_{str(index)}{extension}'

        return f'tfm_{self._generate_base_file_name(subject, image)}_{str(index)}'

    def _check_file_path(self, path: str) -> None:
        if os.path.exists(path):
            if self.allow_override:
                os.remove(path)
            else:
                raise FileExistsError(f'The file with path {path} is already existing and '
                                      f'allow_override is set to false!')

    def write(self,
              subject: Subject,
              write_transforms: bool = True
              ) -> None:
        """Writes a subject to the specified directory.

        Args:
            subject (Subject): The subject which will be written.
            write_transforms (bool): If True writes also the transformation files for each image and segmentation.

        Returns:
            None
        """
        images = []
        images.extend(subject.intensity_images)
        images.extend(subject.segmentation_images)

        for image in images:
            image_file_name = self.generate_image_file_name(subject, image)
            image_file_path = os.path.join(self.path, image_file_name + self.image_file_format.value)

            self._check_file_path(image_file_path)

            itk.imwrite(image.get_image(), image_file_path)

            if write_transforms:
                for i, transform_info in enumerate(image.get_transform_tape().get_recorded_elements()):
                    transform_file_name = self.generate_transform_file_name(subject, image, i)
                    transform_file_path = os.path.join(self.path, transform_file_name)

                    self._check_file_path(transform_file_path)

                    sitk.WriteTransform(transform_info.get_transform(False), transform_file_path)


class DicomSubjectWriter:
    """A class for writing DICOM data to a folder or a zip file."""

    def __init__(self,
                 output_path: str,
                 folder_name: str,
                 as_zip: bool
                 ) -> None:
        """Constructs the DICOM writer.

        Args:
            output_path (str): The output path.
            folder_name (str): The name of the output folder.
            as_zip (bool): Indicates if the output should be a zip file or a normal folder
        """
        super().__init__()

        if not os.path.exists(output_path):
            raise Exception(f'The output path {output_path} is invalid!')

        if not os.path.isdir(output_path):
            raise NotADirectoryError(f'The output path {output_path} is not a directory!')

        self.output_path = output_path

        self.folder_name = remove_illegal_folder_characters(folder_name)
        self.as_zip = as_zip

    def _write_to_folder(self,
                         series_infos: Tuple[DicomSeriesInfo],
                         datasets: Tuple[Tuple[str, Dataset], ...]
                         ) -> None:
        # prepare the output directory
        output_dir_path = os.path.join(self.output_path, self.folder_name)
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)

        # prepare for copying the data
        source_file_paths = []
        target_file_paths = []

        for series_info in series_infos:
            paths = series_info.get_path()
            source_file_paths.extend(paths)

            for path in paths:
                target_path = os.path.join(output_dir_path, Path(path).name)
                target_file_paths.append(target_path)

        # copy the files
        for source_path, target_path in zip(source_file_paths, target_file_paths):
            copy2(source_path, target_path)

        # write the datasets
        if datasets:
            for name, dataset in datasets:
                if not name.endswith('.dcm'):
                    name += '.dcm'

                output_path = os.path.join(output_dir_path, name)
                dataset.save_as(output_path)

    def _write_to_zip(self,
                      series_infos: Tuple[DicomSeriesInfo],
                      datasets: Tuple[Tuple[str, Dataset], ...]
                      ) -> None:
        if self.folder_name.endswith('.zip'):
            folder_name = self.folder_name
        else:
            folder_name = self.folder_name + '.zip'

        output_path = os.path.join(self.output_path, folder_name)

        if os.path.exists(output_path):
            raise Exception(f'The output file {output_path} is already existing!')

        with ZipFile(output_path, 'w') as file:

            # write / copy the series infos
            for series_info in series_infos:
                source_paths = series_info.get_path()

                for path in source_paths:
                    file.write(path, os.path.basename(path))

            if datasets:
                for name, dataset in datasets:
                    if name.endswith('.dcm'):
                        file_name = name
                    else:
                        file_name = name + '.dcm'

                    out = BytesIO()
                    dataset.save_as(out)

                    file.writestr(file_name, out.getvalue())

    def write(self,
              series_infos: Tuple[DicomSeriesInfo, ...],
              datasets: Tuple[Tuple[str, Dataset], ...]
              ) -> None:
        """Writes the data to a folder or a zip file.

        Args:
            series_infos (Tuple[DicomSeriesInfo]): The series infos containing the path for DICOM files to copy.
            datasets (Tuple[Tuple[str, Dataset], ...]): The additional datasets which should be stored.

        Returns:
            None
        """
        if self.as_zip:
            self._write_to_zip(series_infos, datasets)
        else:
            self._write_to_folder(series_infos, datasets)


class DirectorySubjectWriter:
    """A writer class for copying a work folder and addition additional datasets."""

    def __init__(self,
                 input_dicom_dir_path: str,
                 output_path: str,
                 output_name: Optional[str] = None,
                 as_zip: bool = False
                 ) -> None:
        """Constructs a writer class.

        Args:
            input_dicom_dir_path (str): The path to the DICOM directory.
            output_path (str): The path to the output base directory.
            output_name (Optional[str]): The output name of the folder or the zip file (default=None).
            as_zip (bool): Indicates if the output should be a zip file or a folder (default=False).
        """
        super().__init__()

        if not os.path.exists(input_dicom_dir_path):
            raise Exception(f'The DICOM work directory path {input_dicom_dir_path} is invalid!')

        if not os.path.isdir(input_dicom_dir_path):
            raise NotADirectoryError(f'The DICOM work directory path {input_dicom_dir_path} is not a directory!')

        self.input_dicom_dir_path = input_dicom_dir_path

        if not os.path.exists(output_path):
            raise Exception(f'The output path {output_path} is invalid!')

        if not os.path.isdir(output_path):
            raise NotADirectoryError(f'The output path {output_path} is not a directory!')

        self.output_path = output_path

        if isinstance(output_name, str):
            self.output_name = remove_illegal_folder_characters(output_name)
        else:
            self.output_name = output_name

        self.as_zip = as_zip

    def _write_to_zip(self,
                      datasets: Tuple[Tuple[str, Dataset], ...]
                      ) -> None:
        if self.output_name is None:
            raise ValueError('For zipping an output name must be provided!')

        if self.output_name.endswith('.zip'):
            output_name = self.output_name
        else:
            output_name = self.output_name + '.zip'

        output_path = os.path.join(self.output_path, output_name)

        if os.path.exists(output_path):
            raise Exception(f'The output file {output_path} is already existing!')

        with ZipFile(output_path, 'w') as zip_file:

            # write / copy the files and folders
            for root, _, files in os.walk(self.input_dicom_dir_path, topdown=True):
                for file in files:
                    zip_file.write(os.path.join(root, file),
                                   os.path.join(root.replace(self.input_dicom_dir_path, ""), file))

            if datasets:
                for name, dataset in datasets:
                    if name.endswith('.dcm'):
                        file_name = name
                    else:
                        file_name = name + '.dcm'

                    out = BytesIO()
                    dataset.save_as(out)

                    zip_file.writestr(file_name, out.getvalue())

    def _write_to_folder(self,
                         datasets: Tuple[Tuple[str, Dataset], ...]
                         ) -> None:
        # prepare the output directory
        if self.output_name:
            output_dir_path = os.path.join(self.output_path, self.output_name)
        else:
            output_dir_path = self.output_path

        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)

        # copy the files
        for root, dirs, files in os.walk(self.input_dicom_dir_path, topdown=True):
            for dir_ in dirs:
                os.mkdir(os.path.join(output_dir_path, dir_))
            for file in files:
                sub_dir_path = root.replace(self.input_dicom_dir_path, "")[1:]
                target_path = os.path.join(output_dir_path, sub_dir_path, file)
                copy2(os.path.join(root, file), target_path)

        # write the datasets
        if datasets:
            for name, dataset in datasets:
                if not name.endswith('.dcm'):
                    name += '.dcm'

                output_path = os.path.join(output_dir_path, name)
                dataset.save_as(output_path)

    def write(self,
              datasets: Tuple[Tuple[str, Dataset], ...],
              ) -> None:
        """Write the data.

        Args:
            datasets (Tuple[Tuple[str, Dataset], ...]): Datasets to be written to the output directory or zip file.

        Returns:
            None
        """
        if self.as_zip:
            self._write_to_zip(datasets)
        else:
            self._write_to_folder(datasets)
