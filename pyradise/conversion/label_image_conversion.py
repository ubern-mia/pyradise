from typing import (
    Tuple,
    Optional,
    Union)
import os
from copy import deepcopy

import numpy as np
import SimpleITK as sitk
from pydicom import Dataset

from .base_conversion import Converter
from .crawling import DicomSubjectDirectoryCrawler
from .series_information import DicomSeriesImageInfo
from .dicom_conversion import ImageToRTSSConverter


class SimpleITKLabelsToDicomConverter(Converter):
    """A class for converting SimpleITK label images into a DICOM RT Structure Set."""

    def __init__(self,
                 label_images: Union[sitk.Image, Tuple[sitk.Image, ...]],
                 image_series_info: DicomSeriesImageInfo,
                 label_names: Union[str, Tuple[str, ...]],
                 label_colors: Optional[Union[Tuple[int, int, int], Tuple[Tuple[int, int, int], ...]]] = None
                 ) -> None:
        """Constructs the SimpleITK to DICOM RT Structure Set converter.

        Args:
            label_images (Union[sitk.Image, Tuple[sitk.Image, ...]]): The label image to convert.
            image_series_info (DicomSeriesImageInfo): The DICOM image series used for the conversion.
            label_names (Union[str, Tuple[str, ...]]): The label names which will be assigned to the structures.
            label_colors (Optional[Union[Tuple[int, int, int], Tuple[Tuple[int, int, int], ...]]]): The colors to
             the structures.
        """
        super().__init__()

        if isinstance(label_images, sitk.Image):
            self.label_images = (label_images,)  # type: Tuple[sitk.Image, ...]
        else:
            self.label_images = label_images  # type: Tuple[sitk.Image, ...]

        self.image_series_info = image_series_info

        if isinstance(label_names, str):
            self.label_names = (label_names,)
        else:
            self.label_names = label_names

        assert len(self.label_names) == len(self.label_images), 'The number of images must be equal ' \
                                                                'to the number of label names!'
        if label_colors is None:
            self.label_colors = None
        elif isinstance(label_colors[0], int):
            self.label_colors = (label_colors,)
        else:
            self.label_colors = label_colors

        if self.label_colors is not None:
            assert len(self.label_colors) == len(self.label_images), ' The number of label colors must be equal to ' \
                                                                     'the number of images!'

        # check if the images are binary
        for image in self.label_images:
            image_np = sitk.GetArrayFromImage(image)

            assert len(np.unique(image_np)) == 2, 'All provided images need to be binary images but at least one ' \
                                                  'image is a non-binary image!'

    @staticmethod
    def _get_image_series_info_from_path(path: str,
                                         series_instance_uid: Optional[str]
                                         ) -> DicomSeriesImageInfo:
        series_infos = DicomSubjectDirectoryCrawler(path).execute()

        series_infos = [series_info for series_info in series_infos if isinstance(series_info, DicomSeriesImageInfo)]

        if len(series_infos) > 1 and series_instance_uid is None:
            print('Available DICOM SeriesInstanceUIDs:')
            for i, series_info in enumerate(series_infos):
                print(f'DICOM Series {i}:\t{series_info.series_instance_uid}')

            raise Exception('There are multiple DICOM series in the selected directory but no SeriesInstanceUID is '
                            'provided for selecting the right series!')

        if len(series_infos) > 1 and series_instance_uid is not None:
            selected_series_infos = []

            for series_info in series_infos:
                if series_info.series_instance_uid == series_instance_uid:
                    selected_series_infos.append(series_info)

            if not selected_series_infos:
                raise Exception(f'There is no DICOM series in the path {path} with a '
                                f'SeriesInstanceUID {series_instance_uid}!')
            if len(selected_series_infos) > 1:
                raise Exception(f'There are multiple DICOM series with SeriesInstanceUID {series_instance_uid}!')

            return selected_series_infos[0]

        return series_infos[0]

    @staticmethod
    def _load_and_split_label_image(path: str) -> Tuple[sitk.Image]:
        image = sitk.ReadImage(path, sitk.sitkUInt8)  # type: sitk.Image

        image_np = sitk.GetArrayFromImage(image)

        image_labels = list(np.unique(image_np))
        if 0 in image_labels:
            image_labels.pop(image_labels.index(0))

        images = []
        if len(image_labels) > 1:
            for image_label in image_labels:
                image_np_ = deepcopy(image_np)

                image_np_[image_np_ != image_label] = 0
                image_np_[image_np_ == image_label] = 1

                image_ = sitk.GetImageFromArray(image_np_)  # type: sitk.Image
                image_.CopyInformation(image)

                images.append(image_)

        else:
            images.append(image)

        return tuple(images)

    @classmethod
    def from_path(cls,
                  image_file_path: str,
                  dicom_dir_path: str,
                  label_names: Optional[Tuple[str, ...]],
                  label_colors: Optional[Union[Tuple[int, int, int], Tuple[Tuple[int, int, int]]]],
                  series_instance_uid: Optional[str] = None
                  ) -> "SimpleITKLabelsToDicomConverter":
        """Interface for generating DICOM RT-STRUCT directly from a file.

        Args:
            image_file_path (str): The path to the label file.
            dicom_dir_path (str): The path to the directory holding the DICOM image series.
            label_names (Optional[Tuple[str, ...]]): The label names of the structures.
            label_colors (Optional[Union[Tuple[int, int, int], Tuple[Tuple[int, int, int]]]]): The label colors used
             for the structure coloring.
            series_instance_uid (Optional[str]): The SeriesInstanceUID for the image series to use.

        Returns:
            SimpleITKLabelsToDicomConverter: An instance of the converter class.
        """

        if not os.path.exists(image_file_path):
            raise FileNotFoundError(f'The image file ({image_file_path}) is not existing!')

        if not os.path.exists(dicom_dir_path):
            raise Exception(f'The DICOM directory path ({dicom_dir_path}) is not existing!')

        if not os.path.isdir(dicom_dir_path):
            raise Exception(f'The DICOM directory path ({dicom_dir_path}) is not a directory path!')

        # load and check the image series infos
        series_info = SimpleITKLabelsToDicomConverter._get_image_series_info_from_path(dicom_dir_path,
                                                                                       series_instance_uid)

        # load and split the image if necessary
        images = SimpleITKLabelsToDicomConverter._load_and_split_label_image(image_file_path)

        # check or construct the label names
        if label_names is None:
            label_names = [f'Structure_{i}' for i in range(len(images))]

        else:
            assert len(label_names) == len(images), f'The number of label names ({len(label_names)}) must ' \
                                                    f'be equal to the number of binary images ({len(images)})!'

        # check or construct the label colors
        if label_colors is not None:
            assert len(label_colors) == len(images), f'The number of label colors ({len(label_colors)}) must ' \
                                                     f'be equal to the number of binary images ({len(images)})!'

        # construct the class
        converter = cls(tuple(images), series_info, tuple(label_names), label_colors)

        return converter

    def convert(self) -> Dataset:
        """Convert the provided image into a DICOM RT-STRUCT.

        Returns:
            Dataset: The Dataset of the DICOM RT Structures Set.
        """

        image_to_rtss_converter = ImageToRTSSConverter(self.label_images, self.image_series_info.get_path(),
                                                       self.label_names, self.label_colors)
        rtss = image_to_rtss_converter.convert()

        return rtss
