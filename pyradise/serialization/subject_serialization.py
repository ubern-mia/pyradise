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

from pyradise.data import (
    Subject,
    IntensityImage,
    SegmentationImage)

__all__ = ['SubjectWriter', 'DirectorySubjectWriter', 'DicomSeriesSubjectWriter', 'ImageFileFormat']


DicomSeriesInfo = TypeVar('DicomSeriesInfo')


def remove_illegal_folder_characters(name: str) -> str:
    """Removes illegal characters from a folder name.

    Args:
        name (str): The folder name with potential illegal characters.

    Returns:
        str: The folder name without illegal characters.
    """
    illegal_characters = r"""[<>:/\\|?*\']|[\0-\31]"""
    return sub(illegal_characters, "", name)


class ImageFileFormat(Enum):
    """An enumeration for image file formats."""

    NIFTI = '.nii'
    """Image format NIFTI / extension .nii"""

    NIFTI_GZ = '.nii.gz'
    """Image format NIFTI GZ / extension .nii.gz"""

    NRRD = '.nrrd'
    """Image format NRRD / extension .nrrd"""


class SubjectWriter:
    """A class for writing the content of a subject to a directory.

    Args:
        image_file_format (ImageFileFormat): The output file format (default: ImageFileFormat.NIFTI_GZ).
        allow_override (bool): If True the writer can overwrite existing files, otherwise not (default: False).
    """

    def __init__(self,
                 image_file_format: ImageFileFormat = ImageFileFormat.NIFTI_GZ,
                 allow_override: bool = False
                 ) -> None:
        super().__init__()

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

    def _generate_image_file_name(self,
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

    def _generate_transform_file_name(self,
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
              path: str,
              subject: Subject,
              write_transforms: bool = True
              ) -> None:
        """Write a subject to the specified directory.

        Args:
            path (str): The path to the subject directory.
            subject (Subject): The subject which will be written.
            write_transforms (bool): If True writes the transformation files for each image and segmentation,
             otherwise not.

        Returns:
            None
        """
        images = []
        images.extend(subject.intensity_images)
        images.extend(subject.segmentation_images)

        for image in images:
            image_file_name = self._generate_image_file_name(subject, image)
            image_file_path = os.path.join(path, image_file_name + self.image_file_format.value)

            self._check_file_path(image_file_path)

            itk.imwrite(image.get_image(), image_file_path)

            if write_transforms:
                for i, transform_info in enumerate(image.get_transform_tape().get_recorded_elements()):
                    transform_file_name = self._generate_transform_file_name(subject, image, i)
                    transform_file_path = os.path.join(path, transform_file_name)

                    self._check_file_path(transform_file_path)

                    sitk.WriteTransform(transform_info.get_transform(False), transform_file_path)


class DicomSeriesSubjectWriter:
    """A combined writer which copies DICOM files specified by a tuple of :class:`DicomSeriesInfo` from a directory
    to an output directory while adding the specified datasets as files. Furthermore, the writer is feasible to save
    the data as a zip instead of a folder.

    In contrast to the :class:`DirectorySubjectWriter` the :class:`DicomSeriesSubjectWriter` provides a different
    interface which takes a tuple of :class:`DicomSeriesInfo` instead of a directory path for copying existing data.

    Args:
        as_zip (bool): Indicates if the output should be a zip file or a normal folder (default: False).
    """

    def __init__(self, as_zip: bool = False) -> None:
        super().__init__()

        self.as_zip = as_zip

    @staticmethod
    def _write_to_folder(series_infos: Tuple[DicomSeriesInfo],
                         datasets: Tuple[Tuple[str, Dataset], ...],
                         output_path: str,
                         folder_name: Optional[str],
                         ) -> None:
        # prepare the output directory
        if folder_name:
            output_dir_path = os.path.join(output_path, folder_name)
        else:
            output_dir_path = output_path

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

                dataset.save_as(os.path.join(output_dir_path, name))

    @staticmethod
    def _write_to_zip(series_infos: Tuple[DicomSeriesInfo],
                      datasets: Tuple[Tuple[str, Dataset], ...],
                      output_path: str,
                      folder_name: Optional[str],
                      ) -> None:
        if not folder_name:
            raise ValueError('For zipping an folder name must be provided!')

        if not folder_name.endswith('.zip'):
            folder_name += '.zip'

        output_path = os.path.join(output_path, folder_name)

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
              datasets: Tuple[Tuple[str, Dataset], ...],
              output_path: str,
              folder_name: Optional[str],
              series_infos: Optional[Tuple[DicomSeriesInfo, ...]] = None
              ) -> None:
        """Write the data to a folder or a zip file.

        Notes:
            If the ``series_infos`` is not provided, no data will be copied into the output.

        Args:
            datasets (Tuple[Tuple[str, Dataset], ...]): The additional datasets which should be stored.
            output_path (str): The output path.
            folder_name (str): The name of the output folder or the zip file.
            series_infos (Optional[Tuple[DicomSeriesInfo]]): The series infos containing the path for DICOM files to
             copy (default: None).

        Returns:
            None
        """
        if not os.path.exists(output_path):
            raise Exception(f'The output path {output_path} is invalid!')

        if not os.path.isdir(output_path):
            raise NotADirectoryError(f'The output path {output_path} is not a directory!')

        folder_name = remove_illegal_folder_characters(folder_name)

        if not series_infos:
            series_infos = []

        if self.as_zip:
            self._write_to_zip(series_infos, datasets, output_path, folder_name)
        else:
            self._write_to_folder(series_infos, datasets, output_path, folder_name)


class DirectorySubjectWriter:
    """A combined writer which copies data from a directory to another directory and adds additional datasets.
    Furthermore, the writer can zip all data at the output directory.

    In contrast to the :class:`DicomSeriesSubjectWriter` the :class:`DirectorySubjectWriter` provides a different
    interface which takes a directory path instead of a tuple of :class:`DicomSeriesInfo` for copying existing data.

    Notes:
        This writer class is implemented with a focus on deployable segmentation pipelines which need to enrich the
        input data with for example a DICOM RT-STRUCT. For this scenario we assume that we have an input directory,
        a work directory, and an output directory. The pipeline may sort or unzip the input directory data to the work
        directory from where it does the ingestion. After the processing the input data from the work directory and
        the newly generated DICOM RT need to be stored in the output directory. This writer class provides the
        functionality to copy the data from the work directory to the output directory while also writing the
        additional DICOM RT dataset to a file. If ``as_zip`` is set to True the output will be zipped.

    Args:
        as_zip (bool): Indicates if the output should be a zip file or a folder (default: False).
    """

    def __init__(self, as_zip: bool = False) -> None:
        super().__init__()

        self.as_zip = as_zip

    @staticmethod
    def _write_to_zip(datasets: Tuple[Tuple[str, Dataset], ...],
                      copy_dir_path: Optional[str],
                      output_path: str,
                      folder_name: Optional[str],
                      ) -> None:
        if folder_name is None:
            raise ValueError('For zipping an output name must be provided!')

        if not folder_name.endswith('.zip'):
            folder_name += '.zip'

        output_path = os.path.join(output_path, folder_name)

        if os.path.exists(output_path):
            raise Exception(f'The output file {output_path} is already existing!')

        with ZipFile(output_path, 'w') as zip_file:

            # write / copy the files and folders
            if copy_dir_path:
                for root, _, files in os.walk(copy_dir_path, topdown=True):
                    for file in files:
                        zip_file.write(os.path.join(root, file),
                                       os.path.join(root.replace(copy_dir_path, ""), file))

            if datasets:
                for name, dataset in datasets:
                    if name.endswith('.dcm'):
                        file_name = name
                    else:
                        file_name = name + '.dcm'

                    out = BytesIO()
                    dataset.save_as(out)

                    zip_file.writestr(file_name, out.getvalue())

    @staticmethod
    def _write_to_folder(datasets: Tuple[Tuple[str, Dataset], ...],
                         copy_dir_path: Optional[str],
                         output_path: str,
                         folder_name: Optional[str],
                         ) -> None:
        # prepare the output directory
        if folder_name:
            output_dir_path = os.path.join(output_path, folder_name)
        else:
            output_dir_path = output_path

        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)

        # copy the files
        if copy_dir_path:
            for root, dirs, files in os.walk(copy_dir_path, topdown=True):
                for dir_ in dirs:
                    os.mkdir(os.path.join(output_dir_path, dir_))
                for file in files:
                    sub_dir_path = root.replace(copy_dir_path, "")[1:]
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
              output_path: str,
              folder_name: Optional[str] = None,
              copy_dir_path: Optional[str] = None
              ) -> None:
        """Write the data.

        Notes:
            If the ``copy_dir_path`` is not provided, no data will be copied into the output.

        Args:
            datasets (Tuple[Tuple[str, Dataset], ...]): Datasets to be written to the output directory or zip file.
            output_path (str): The path to the output base directory.
            folder_name (Optional[str]): The name of the folder or the zip file (default: None).
            copy_dir_path (str): The path to the directory from which all data should be copied.

        Returns:
            None
        """
        if copy_dir_path is not None:
            if not os.path.exists(copy_dir_path):
                raise Exception(f'The DICOM work directory path {copy_dir_path} is invalid!')

            if not os.path.isdir(copy_dir_path):
                raise NotADirectoryError(f'The DICOM work directory path {copy_dir_path} is not a directory!')

        if not os.path.exists(output_path):
            raise Exception(f'The output path {output_path} is invalid!')

        if not os.path.isdir(output_path):
            raise NotADirectoryError(f'The output path {output_path} is not a directory!')

        if isinstance(folder_name, str):
            folder_name = remove_illegal_folder_characters(folder_name)

        if self.as_zip:
            self._write_to_zip(datasets, copy_dir_path, output_path, folder_name)
        else:
            self._write_to_folder(datasets, copy_dir_path, output_path, folder_name)
