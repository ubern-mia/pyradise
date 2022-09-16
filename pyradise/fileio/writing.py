from typing import (
    Union,
    Optional,
    Tuple,
    Callable)
import os
from enum import Enum
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
from pyradise.utils import remove_illegal_folder_chars
from .series_info import DicomSeriesInfo

__all__ = ['SubjectWriter', 'DirectorySubjectWriter', 'DicomSeriesSubjectWriter', 'ImageFileFormat',
           'default_intensity_file_name_fn', 'default_segmentation_file_name_fn']


def default_intensity_file_name_fn(subject: Subject,
                                   image: IntensityImage
                                   ) -> str:
    """The default intensity file name generation function.

    Notes:
        The file name must not contain the file extension because this is provided by the writer.

    Args:
        subject (Subject): The subject.
        image (IntensityImage): The intensity image.

    Returns:
        str: The file name.
    """
    return f'img_{subject.name}_{image.get_modality(as_str=True)}'


def default_segmentation_file_name_fn(subject: Subject,
                                      image: SegmentationImage
                                      ) -> str:
    """The default segmentation file name generation function.

    Notes:
        The file name must not contain the file extension because this is provided by the writer.

    Args:
        subject (Subject): The subject.
        image (SegmentationImage): The segmentation image.

    Returns:
        str: The file name.
    """
    rater_name = image.get_rater().name if image.get_rater() else 'NA'
    return f'seg_{subject.name}_{rater_name}_{image.get_organ(as_str=True)}'


class ImageFileFormat(Enum):
    """An enumeration of possible output image file formats.

    Notes:
        The current implementation supports only the following formats:
        - NIFTI (.nii, .nii.gz)
        - NRRD (.nrrd)

        More image file formats will be added in the future.
    """

    NIFTI = '.nii'
    """Image format NIFTI / extension .nii"""

    NIFTI_GZ = '.nii.gz'
    """Image format NIFTI GZ / extension .nii.gz"""

    NRRD = '.nrrd'
    """Image format NRRD / extension .nrrd"""


class SubjectWriter:
    """A class for writing the content of a subject to a directory.

    Notes:
        The current implementation of this class support the addition of separate functions for generating
        the file names. Please be aware that certain patterns in the naming can cause problems when loading the data
        again (e.g. multiple word organ names separated by underline while separating the rater also by underline).

        Currently, the serialization of :class:`IntensityImage`, :class:`SegmentationImage` and transformations
        is supported.

    Args:
        file_format (ImageFileFormat): The output file format (default: ImageFileFormat.NIFTI_GZ).
        intensity_file_name_fn (Callable[[Subject, IntensityImage], str]): The function for generating the file names
         of the intensity images.
        segmentation_file_name_fn (Callable[[Subject, SegmentationImage], str]): The function for generating the file
         names of the segmentation images.
        allow_override (bool): If True the writer can overwrite existing files, otherwise not (default: False).
    """

    def __init__(self,
                 file_format: ImageFileFormat = ImageFileFormat.NIFTI_GZ,
                 intensity_file_name_fn: Callable[[Subject, IntensityImage], str] = default_intensity_file_name_fn,
                 segmentation_file_name_fn: Callable[[Subject, SegmentationImage], str] =
                 default_segmentation_file_name_fn,
                 allow_override: bool = False
                 ) -> None:
        super().__init__()

        self.image_file_format = file_format
        self.intensity_file_name_fn = intensity_file_name_fn
        self.segmentation_file_name_fn = segmentation_file_name_fn
        self.allow_override = allow_override

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

        Raises:
            ValueError: If the image is not an :class:`IntensityImage` or :class:`SegmentationImage`.

        Returns:
            str: The file name of the image file.
        """
        if isinstance(image, IntensityImage):
            file_name = self.intensity_file_name_fn(subject, image)

        elif isinstance(image, SegmentationImage):
            file_name = self.segmentation_file_name_fn(subject, image)

        else:
            raise ValueError(f'Unsupported data type {type(image)} received for serialization.')

        if with_extension:
            return file_name + self.image_file_format.value

        return file_name

    def _generate_transform_file_name(self,
                                      subject: Subject,
                                      image: Union[IntensityImage, SegmentationImage],
                                      index: Union[int, str],
                                      extension: str = '.tfm'
                                      ) -> str:
        """Generate a transformation file name.

        Args:
            subject (Subject): The subject where the transformation belongs to.
            image (Union[IntensityImage, SegmentationImage]): The image to which the transformation belongs to.
            index (Union[int, str]): The index of the transformation.
            extension (str): The file extension for the transformation file (default: '.tfm').

        Returns:
            str: The file name of the transformation file.
        """
        if isinstance(image, IntensityImage):
            file_name = 'tfm_' + self.intensity_file_name_fn(subject, image) + f'_{str(index)}{extension}'
        elif isinstance(image, SegmentationImage):
            file_name = 'tfm_' + self.segmentation_file_name_fn(subject, image) + f'_{str(index)}{extension}'
        else:
            raise ValueError(f'Unsupported data type {type(image)} received for serialization.')

        return file_name

    def _check_file_path(self, path: str) -> None:
        """Check if the file path is valid.

        Args:
            path (str): The file path to check.

        Returns:
            None
        """
        if os.path.exists(path):
            if self.allow_override:
                os.remove(path)
            else:
                raise FileExistsError(f'The file with path {path} is already existing and '
                                      'allow_override is set to false!')

    def write(self,
              path: str,
              subject: Subject,
              write_transforms: bool = True
              ) -> None:
        """Write a :class:`Subject` instance to the specified directory.

        Args:
            path (str): The path to the subject directory.
            subject (Subject): The :class:`Subject` which will be written.
            write_transforms (bool): If True writes the transformation files for each :class:`IntensityImage` and
             :class:`SegmentationImage`, otherwise not (default: True).

        Returns:
            None
        """
        if not os.path.exists(path):
            raise NotADirectoryError(f'The directory {path} does not exist!')

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

    def write_to_subject_folder(self,
                                base_dir_path: str,
                                subject: Subject,
                                write_transforms: bool = True
                                ) -> None:
        """Write a :class:`Subject` instance to a separate subject directory within the specified base directory.

        Notes:
            The subject directory will be generated within the base directory path and will be named after the subject.
            This is function is just a small wrapper around the write function and reduces the amount of code needed to
            write subjects to separate directories.

        Args:
            base_dir_path (str): The path to the base directory where the subject directory will be placed.
            subject (Subject): The :class:`Subject` which will be written.
            write_transforms (bool): If True writes the transformation files for each :class:`IntensityImage` and
             :class:`SegmentationImage`, otherwise not (default: True).

        Returns:
            None
        """
        subject_path = os.path.normpath(os.path.join(base_dir_path, subject.name))
        if not os.path.exists(subject_path):
            os.mkdir(subject_path)
        else:
            raise FileExistsError(f'The subject directory {subject_path} is already existing!')

        self.write(subject_path, subject, write_transforms)


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
        """Write the provided datasets to the specified folder while copying also the files associated with the
        ``series_infos`` to the ``output_path``.

        Args:
            series_infos (Tuple[DicomSeriesInfo]): The DICOM series infos which will be copied.
            datasets (Tuple[Tuple[str, Dataset], ...]): The datasets which will be written to the folder.
            output_path (str): The path to the output folder.
            folder_name (Optional[str]): The name of the folder where the data will be written to. If None no new
             folder will be generated and the data will be directly be written into the specified ``output_path``.

        Returns:
            None
        """
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
                      folder_name: str,
                      ) -> None:
        """Write the provided datasets to the specified folder as a zip file while also copying the files associated
        with the ``series_infos`` into the zip file.

        Args:
            series_infos (Tuple[DicomSeriesInfo]): The DICOM series infos which will be copied into the zip file.
            datasets (Tuple[Tuple[str, Dataset], ...]): The datasets which will be written to the zip file.
            output_path (str): The path to the output folder.
            folder_name (str): The name of the zip file where the data will be written to.

        Returns:
            None
        """
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
            If the ``series_infos`` is not provided, no data will be copied into the output. However, the datasets will
            be written to the output.

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
            raise Exception(f'The output path {output_path} is already existing!')

        if not os.path.isdir(output_path):
            raise NotADirectoryError(f'The output path {output_path} is not a directory!')

        folder_name = remove_illegal_folder_chars(folder_name)

        if not series_infos:
            series_infos = []

        if self.as_zip:
            if not folder_name:
                raise ValueError('For zipping an folder name must be provided!')
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
        directory from where it does the ingestion. After the process the input data from the work directory and
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
                      folder_name: str,
                      ) -> None:
        """Write the provided datasets to the specified folder as a zip file while also copying the files located in
        the ``copy_dir_path`` into the zip file.

        Args:
            copy_dir_path (Optional[str]): The path to the directory which will be copied to the output.
            datasets (Tuple[Tuple[str, Dataset], ...]): The datasets which will be written to the zip file.
            output_path (str): The path to the output folder.
            folder_name (str): The name of the zip file where the data will be written to.

        Returns:
            None
        """
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
        """Write the provided datasets to the specified folder while copying also the files associated with the
        ``copy_dir_path`` to the ``output_path``.

        Args:
            datasets (Tuple[Tuple[str, Dataset], ...]): The datasets which will be written to the folder.
            copy_dir_path (Optional[str]): The path to the directory which will be copied to the output.
            output_path (str): The output path.
            folder_name (Optional[str]): The name of the folder where the data will be written to. If None no new
             folder will be generated and the data will be directly be written into the specified ``output_path``.

        Returns:
            None
        """
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
        """Write the data to a folder or a zip file.

        Notes:
            If the ``copy_dir_path`` is not provided, no data will be copied into the output. However, the datasets will
            be written to the output.

        Args:
            datasets (Tuple[Tuple[str, Dataset], ...]): Datasets to be written to the output directory or zip file.
            output_path (str): The path to the output base directory.
            folder_name (Optional[str]): The name of the folder or the zip file (default: None).
            copy_dir_path (str): The path to the directory from which all data should be copied (default: None).

        Returns:
            None
        """
        if copy_dir_path is not None:
            if not os.path.exists(copy_dir_path):
                raise Exception(f'The copy directory path {copy_dir_path} is invalid!')

            if not os.path.isdir(copy_dir_path):
                raise NotADirectoryError(f'The copy directory path {copy_dir_path} is not a directory!')

        if not os.path.exists(output_path):
            raise Exception(f'The output path {output_path} is already existing!')

        if not os.path.isdir(output_path):
            raise NotADirectoryError(f'The output path {output_path} is not a directory!')

        if isinstance(folder_name, str):
            folder_name = remove_illegal_folder_chars(folder_name)

        if self.as_zip:
            if not folder_name:
                raise ValueError('For zipping an folder name must be provided!')
            self._write_to_zip(datasets, copy_dir_path, output_path, folder_name)
        else:
            self._write_to_folder(datasets, copy_dir_path, output_path, folder_name)