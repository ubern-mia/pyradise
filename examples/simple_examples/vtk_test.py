from typing import Union, Tuple, Dict, Optional, List
from dataclasses import dataclass
from enum import IntEnum
from datetime import datetime
import warnings

import SimpleITK as sitk
import numpy as np
from scipy.interpolate import (
    splprep,
    splev)
import cv2 as cv


import vtkmodules

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import (
    VTK_VERSION_NUMBER,
    vtkLookupTable,
    vtkVersion
)

from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonColor import vtkNamedColors

from vtkmodules.vtkCommonDataModel import (vtkPolyData, vtkPlane)

from vtkmodules.vtkFiltersCore import (
    vtkFlyingEdges3D,
    vtkPolyDataNormals,
    vtkStripper,
    vtkCutter,
    vtkWindowedSincPolyDataFilter,
    vtkAppendPolyData,
    vtkCleanPolyData
)
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkImagingCore import vtkImageThreshold
from vtkmodules.vtkImagingGeneral import vtkImageGaussianSmooth
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from vtkmodules.vtkIOGeometry import (
    vtkSTLReader,
    vtkSTLWriter
)

from pydicom import dcmread, Dataset, FileDataset
from pydicom.uid import (
    generate_uid,
    ImplicitVRLittleEndian,
    PYDICOM_IMPLEMENTATION_UID)
from pydicom.dataset import FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.tag import Tag

from pyradise.utils import load_dataset, load_datasets, get_spacing_between_slices, get_slice_direction, convert_to_itk_image
from pyradise.fileio import DicomSeriesImageInfo


ROI_GENERATION_ALGORITHMS = ['AUTOMATIC', 'SEMIAUTOMATIC', 'MANUAL']

COLOR_PALETTE = [[255, 0, 255],
                 [0, 235, 235],
                 [255, 255, 0],
                 [255, 0, 0],
                 [0, 132, 255],
                 [0, 240, 0],
                 [255, 175, 0],
                 [0, 208, 255],
                 [180, 255, 105],
                 [255, 20, 147],
                 [160, 32, 240],
                 [0, 255, 127],
                 [255, 114, 0],
                 [64, 224, 208],
                 [0, 178, 47],
                 [220, 20, 60],
                 [238, 130, 238],
                 [218, 165, 32],
                 [255, 140, 190],
                 [0, 0, 255],
                 [255, 225, 0]]

@dataclass
class ROIData:
    """Data class to collect ROI data.

    Args:
        mask (np.ndarray): The segmentation mask.
        color (Union[str, List[int]]): The color of the ROI.
        number (int): The ROINumber.
        frame_of_reference (int): The FrameOfReferenceUID.
        description (str): The description of the ROI.
        use_pin_hole (bool): If the pinhole algorithm should be used (default: False).
        approximate_contours (bool): If True the contours will be approximated, otherwise not (default: True).
        roi_generation_algorithm (Union[str, int]): The ROI generation algorithm selected (currently no function,
         default: 0).
    """
    mask: np.ndarray
    color: Union[str, List[int]]
    number: int
    name: str
    frame_of_reference_uid: int
    description: str = ''
    use_pin_hole: bool = False
    approximate_contours: bool = True
    roi_generation_algorithm: Union[str, int] = 0

    def __post_init__(self):
        self._validate_color()
        self._add_default_values()
        self._validate_roi_generation_algorithm()

    def _add_default_values(self):
        if self.color is None:
            self.color = COLOR_PALETTE[(self.number - 1) % len(COLOR_PALETTE)]

        if self.name is None:
            self.name = f'Structure_{self.number}'

    def _validate_color(self):
        if self.color is None:
            return

        if isinstance(self.color, list):
            if len(self.color) != 3:
                raise ValueError(f'{self.color} is an invalid color for an ROI')
            for color in self.color:
                assert 0 <= color <= 255, ValueError(f'{self.color} is an invalid color for an ROI')

        else:
            self.color: str = str(self.color)
            self.color = self.color.strip('#')

            if len(self.color) == 3:
                self.color = ''.join([x * 2 for x in self.color])

            if not len(self.color) == 6:
                raise ValueError(f'{self.color} is an invalid color for an ROI')

            try:
                self.color = [int(self.color[i:i + 2], 16) for i in (0, 2, 4)]

            except Exception(f'{self.color} is an invalid color for an ROI') as error:
                raise error

    def _validate_roi_generation_algorithm(self):

        if isinstance(self.roi_generation_algorithm, int):
            # for ints we use the predefined values in ROI_GENERATION_ALGORITHMS
            if self.roi_generation_algorithm > 2 or self.roi_generation_algorithm < 0:
                raise ValueError('roi_generation_algorithm must be either an int (0=\'AUTOMATIC\', '
                                 '1=\'SEMIAUTOMATIC\', 2=\'MANUAL\') or a str (not recommended).')

            self.roi_generation_algorithm = ROI_GENERATION_ALGORITHMS[self.roi_generation_algorithm]

        elif isinstance(self.roi_generation_algorithm, str):
            # users can pick a str if they want to use a value other than the three default values
            if self.roi_generation_algorithm not in ROI_GENERATION_ALGORITHMS:
                print('Got self.roi_generation_algorithm {}. Some viewers might complain about this option. '
                      'Better options might be 0=\'AUTOMATIC\', 1=\'SEMIAUTOMATIC\', or 2=\'MANUAL\'.'
                      .format(self.roi_generation_algorithm))

        else:
            raise TypeError('Expected int (0=\'AUTOMATIC\', 1=\'SEMIAUTOMATIC\', 2=\'MANUAL\') '
                            'or a str (not recommended) for self.roi_generation_algorithm. Got {}.'
                            .format(type(self.roi_generation_algorithm)))


class Hierarchy(IntEnum):
    """ Utility enum class for OpenCV to detect what the positions in the hierarchy array mean.
    """
    NEXT_NODE = 0
    PREVIOUS_NODE = 1
    FIRST_CHILD = 2
    PARENT_NODE = 3


class SegmentToRTSSConverter:
    """A low-level :class:`Converter` class for converting one or multiple
    :class:`~pyradise.data.image.SegmentationImage` instances to a DICOM-RTSS :class:`~pydicom.dataset.Dataset`.
    In contrast to the :class:`SubjectToRTSSConverter` class, this class generates the DICOM-RTSS from a sequence of
    binary :class:`SimpleITK.Image` instances and the appropriate DICOM image :class:`~pydicom.dataset.Dataset`
    instances instead of the :class:`~pyradise.fileio.series_info.DicomSeriesInfo` entries.

    Warning:
        The provided ``label_images`` must be binary, otherwise the conversion will fail.

    Notes:
        Typically, this class is not used directly by the used but via the :class:`SubjectToRTSSConverter` which
        processes :class:`~pyradise.fileio.series_info.DicomSeriesInfo` entries and thus provides a more suitable
        interface.

    Args:
        label_images (Union[Tuple[str, ...], Tuple[sitk.Image, ...]]): The path to the images or a sequence of
         :class:`SimpleITK.Image` instances.
        ref_image_datasets (Union[Tuple[str, ...], Tuple[Dataset, ...]]): The referenced DICOM image
         :class:`~pydicom.dataset.Dataset` instances.
        roi_names (Union[Tuple[str, ...], Dict[int, str], None]): The label names which will be assigned to the ROIs.
        colors (Optional[Tuple[Tuple[int, int, int], ...]]): The colors which will be assigned to the ROIs.
    """

    def __init__(self,
                 label_images: Union[Tuple[str, ...], Tuple[sitk.Image, ...]],
                 ref_image_datasets: Union[Tuple[str, ...], Tuple[Dataset, ...]],
                 roi_names: Union[Tuple[str, ...], Dict[int, str], None],
                 colors: Optional[Tuple[Tuple[int, int, int], ...]],
                 ) -> None:
        # pylint: disable=consider-using-generator
        super().__init__()

        if isinstance(label_images[0], str):
            self.label_images = tuple([sitk.ReadImage(path, sitk.sitkUInt8) for path in label_images])
        else:
            self.label_images = label_images

        if isinstance(ref_image_datasets[0], str):
            self.image_datasets: Tuple[Dataset, ...] = load_datasets(ref_image_datasets)
        else:
            self.image_datasets: Tuple[Dataset, ...] = ref_image_datasets

        if isinstance(roi_names, dict):
            sorted_keys = sorted(roi_names.keys())
            self.roi_names = tuple([str(roi_names.get(key)) for key in sorted_keys])
        elif not roi_names:
            self.roi_names = tuple([f'Structure_{i}' for i in range(len(self.label_images))])
        else:
            self.roi_names = roi_names

        if not colors:
            self.colors = COLOR_PALETTE
        else:
            self.colors = colors

        assert len(self.roi_names) >= len(self.label_images), 'The number of ROI names must be equal or larger ' \
                                                              'than the number of label images!'
        assert len(self.colors) >= len(self.label_images), 'The number of colors must be equal or larger ' \
                                                           'than the number of label images!'

        self._validate_label_images()

    def _validate_label_images(self) -> None:
        """Validate the label images.

        Raises:
            ValueError: If the label images are not binary or the pixel type is not an integer.
        """
        # check if the label images are binary and that they have an integer pixel type
        for image in self.label_images:
            if np.unique(sitk.GetArrayFromImage(image)).size > 2:
                raise ValueError('The label images must be binary!')

            if 'float' in image.GetPixelIDTypeAsString():
                raise ValueError('The label images must have an integer pixel type!')

    @staticmethod
    def _generate_basic_rtss(ref_image_datasets: Tuple[Dataset, ...],
                             file_name: str = 'rt_struct'
                             ) -> FileDataset:
        """Generate the basic RTSS skeleton.

        Args:
            ref_image_datasets (Tuple[Dataset, ...]): The referenced image datasets.
            file_name (str): The file name.

        Returns:
            FileDataset: The basic RT Structure Set.
        """
        # create the file meta for the dataset
        file_meta = FileMetaDataset()
        file_meta.FileMetaInformationGroupLength = 202
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

        # create the dataset
        rtss = FileDataset(file_name, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # add the basic information
        now = datetime.now()
        rtss.SpecificCharacterSet = 'ISO_IR 100'
        rtss.InstanceCreationDate = now.strftime('%Y%m%d')
        rtss.InstanceCreationTime = now.strftime('%H%M%S.%f')
        rtss.StructureSetLabel = 'Autogenerated'
        rtss.StructureSetDate = now.strftime('%Y%m%d')
        rtss.StructureSetTime = now.strftime('%H%M%S.%f')
        rtss.Modality = 'RTSTRUCT'
        rtss.Manufacturer = 'University of Bern, Switzerland'
        rtss.ManufacturerModelName = 'PyRaDiSe'
        rtss.InstitutionName = 'University of Bern, Switzerland'
        rtss.OperatorsName = 'PyRaDiSe Converter Algorithm'
        rtss.ApprovalStatus = 'UNAPPROVED'

        rtss.is_little_endian = True
        rtss.is_implicit_VR = True

        # set values already defined in the file meta
        rtss.SOPClassUID = rtss.file_meta.MediaStorageSOPClassUID
        rtss.SOPInstanceUID = rtss.file_meta.MediaStorageSOPInstanceUID

        # add study and series information
        reference_dataset = ref_image_datasets[0]
        rtss.StudyDate = reference_dataset.StudyDate
        rtss.SeriesDate = getattr(reference_dataset, 'SeriesDate', '')
        rtss.StudyTime = reference_dataset.StudyTime
        rtss.SeriesTime = getattr(reference_dataset, 'SeriesTime', '')
        rtss.StudyDescription = getattr(reference_dataset, 'StudyDescription', '')
        rtss.SeriesDescription = getattr(reference_dataset, 'SeriesDescription', '')
        rtss.StudyInstanceUID = reference_dataset.StudyInstanceUID
        rtss.SeriesInstanceUID = generate_uid()
        rtss.StudyID = reference_dataset.StudyID
        rtss.SeriesNumber = "99"
        rtss.ReferringPhysicianName = "UNKNOWN"
        rtss.AccessionNumber = "0"

        # add the patient information
        rtss.PatientName = getattr(reference_dataset, 'PatientName', '')
        rtss.PatientID = getattr(reference_dataset, 'PatientID', '')
        rtss.PatientBirthDate = getattr(reference_dataset, 'PatientBirthDate', '')
        rtss.PatientSex = getattr(reference_dataset, 'PatientSex', '')
        rtss.PatientAge = getattr(reference_dataset, 'PatientAge', '')
        rtss.PatientSize = getattr(reference_dataset, 'PatientSize', '')
        rtss.PatientWeight = getattr(reference_dataset, 'PatientWeight', '')

        # ImageToRTSSConverter._add_referenced_frame_of_ref_sequence(rtss, reference_image_datasets)

        # construct the ContourImageSequence
        contour_image_sequence = Sequence()
        for image_dataset in ref_image_datasets:
            contour_image_entry = Dataset()
            contour_image_entry.ReferencedSOPClassUID = image_dataset.file_meta.MediaStorageSOPClassUID
            contour_image_entry.ReferencedSOPInstanceUID = image_dataset.file_meta.MediaStorageSOPInstanceUID
            contour_image_sequence.append(contour_image_entry)

        # construct the RTReferencedSeriesSequence
        rt_referenced_series_sequence = Sequence()
        rt_referenced_series_entry = Dataset()
        rt_referenced_series_entry.SeriesInstanceUID = reference_dataset.SeriesInstanceUID
        rt_referenced_series_entry.ContourImageSequence = contour_image_sequence
        rt_referenced_series_sequence.append(rt_referenced_series_entry)

        # construct the RTReferencedStudySequence
        rt_referenced_study_sequence = Sequence()
        rt_referenced_study_entry = Dataset()
        rt_referenced_study_entry.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.1'  # RT Structure Set Storage
        rt_referenced_study_entry.ReferencedSOPInstanceUID = reference_dataset.StudyInstanceUID
        rt_referenced_study_entry.RTReferencedSeriesSequence = rt_referenced_series_sequence
        rt_referenced_study_sequence.append(rt_referenced_study_entry)

        # construct the ReferencedFrameOfReferenceSequence
        rtss.ReferencedFrameOfReferenceSequence = Sequence()
        referenced_frame_of_ref_entry = Dataset()
        referenced_frame_of_ref_entry.FrameOfReferenceUID = reference_dataset.FrameOfReferenceUID
        referenced_frame_of_ref_entry.RTReferencedStudySequence = rt_referenced_study_sequence
        rtss.ReferencedFrameOfReferenceSequence.append(referenced_frame_of_ref_entry)

        # construct the ROIContourSequence, StructureSetROISequence and RTROIObservationsSequence
        rtss.ROIContourSequence = Sequence()
        rtss.StructureSetROISequence = Sequence()
        rtss.RTROIObservationsSequence = Sequence()

        return rtss

    @staticmethod
    def _create_roi_contour(roi_data: ROIData,
                            image_datasets: Tuple[Dataset, ...],
                            rtss: Dataset
                            ) -> None:
        """Create a ROIContourSequence entry for the given ROI data.

        Args:
            roi_data (ROIData): The ROI data to be used for creating the ROIContourSequence entry.
            image_datasets (Tuple[Dataset, ...]): The referenced image datasets.
            rtss (Dataset): The RTSS dataset.

        Returns:
            None
        """
        roi_contour = Dataset()
        roi_contour.ROIDisplayColor = roi_data.color
        roi_contour.ContourSequence = SegmentToRTSSConverter._create_contour_sequence(roi_data, image_datasets)
        roi_contour.ReferencedROINumber = str(roi_data.number)

        # add the ROIContourSequence entry to the RTSS
        rtss.ROIContourSequence.append(roi_contour)

    @staticmethod
    def _create_contour_sequence(roi_data: ROIData,
                                 image_datasets: Tuple[Dataset, ...]
                                 ) -> Sequence:
        """Create a ContourSequence for the given ROI data by iterating through each slice of the mask.
        For each connected segment within a slice, a ContourSequence entry is created.

        Args:
            roi_data (ROIData): The ROI data to be used for creating the ContourSequence.
            image_datasets (Tuple[Dataset, ...]): The referenced image datasets.

        Returns:
            Sequence: The created ContourSequence.
        """
        contour_sequence = Sequence()

        contours_coordinates = SegmentToRTSSConverter._get_contours_coordinates(roi_data, image_datasets)

        for series_slice, slice_contours in zip(image_datasets, contours_coordinates):
            for contour_data in slice_contours:
                if len(contour_data) <= 3:
                    continue
                contour_seq_entry = SegmentToRTSSConverter._create_contour_sequence_entry(series_slice, contour_data)
                contour_sequence.append(contour_seq_entry)

        return contour_sequence

    @staticmethod
    def _get_contours_coordinates(roi_data: ROIData,
                                  image_datasets: Tuple[Dataset, ...]
                                  ) -> List[List[List[float]]]:
        """Get the contour coordinates for each slice of the mask.

        Args:
            roi_data (ROIData): The ROI data to be used for creating the contour coordinates.
            image_datasets (Tuple[Dataset, ...]): The referenced image datasets.

        Returns:
            List[List[List[float]]]: The contour coordinates for each slice of the mask.

        """
        transform_matrix = SegmentToRTSSConverter._get_pixel_to_patient_transformation_matrix(image_datasets)

        series_contours = []
        for i in range(len(image_datasets)):
            mask_slice = roi_data.mask[i, :, :]

            # Do not add ROI's for blank slices
            if np.sum(mask_slice) == 0:
                series_contours.append([])
                continue

            # Create pinhole mask if specified
            if roi_data.use_pin_hole:
                mask_slice = SegmentToRTSSConverter._create_pin_hole_mask(mask_slice, roi_data.approximate_contours)

            # Get contours from mask
            contours, _ = SegmentToRTSSConverter._find_mask_contours(mask_slice, roi_data.approximate_contours)

            if not contours:
                raise Exception('Unable to find contour in non empty mask, please check your mask formatting!')

            # Format for DICOM
            formatted_contours = []
            for contour in contours:
                # Add z index
                contour = np.concatenate((np.array(contour), np.full((len(contour), 1), i)), axis=1)

                transformed_contour = SegmentToRTSSConverter._apply_transformation_to_3d_points(contour,
                                                                                                transform_matrix)
                dicom_formatted_contour = np.ravel(transformed_contour).tolist()
                formatted_contours.append(dicom_formatted_contour)

            series_contours.append(formatted_contours)

        return series_contours

    @staticmethod
    def _get_pixel_to_patient_transformation_matrix(image_datasets: Tuple[Dataset]) -> np.ndarray:
        """Get the pixel to patient transformation matrix according to the referenced image datasets.

        Notes:
            Description see: https://nipy.org/nibabel/dicom/dicom_orientation.html

        Args:
            image_datasets (Tuple[Dataset]): The image datasets with the necessary information to retrieve the pixel
             to patient transformation matrix.

        Returns:
            np.ndarray: The pixel to patient transformation matrix.
        """

        first_slice = image_datasets[0]

        offset = np.array(first_slice.ImagePositionPatient)
        row_spacing, column_spacing = first_slice.PixelSpacing
        slice_spacing = get_spacing_between_slices(image_datasets)
        row_direction, column_direction, slice_direction = get_slice_direction(first_slice)

        mat = np.identity(4, dtype=np.float32)
        mat[:3, 0] = row_direction * row_spacing
        mat[:3, 1] = column_direction * column_spacing
        mat[:3, 2] = slice_direction * slice_spacing
        mat[:3, 3] = offset

        return mat

    @staticmethod
    def _create_pin_hole_mask(mask: np.ndarray,
                              approximate_contours: bool):
        """Create masks with pinholes added to contour regions with holes. This is done so that a given region can
        be represented by a single contour.

        Args:
            mask (np.ndarray): The mask to be used for creating the pinhole mask.
            approximate_contours (bool): Whether to approximate the contours.

        Returns:
            np.ndarray: The pinhole mask.
        """
        contours, hierarchy = SegmentToRTSSConverter._find_mask_contours(mask, approximate_contours)
        pin_hole_mask = mask.copy()

        # Iterate through the hierarchy, for child nodes, draw a line upwards from the first point
        for i, array in enumerate(hierarchy):
            parent_contour_index = array[Hierarchy.PARENT_NODE]
            if parent_contour_index == -1:
                continue  # Contour is not a child

            child_contour = contours[i]

            line_start = tuple(child_contour[0])

            pin_hole_mask = SegmentToRTSSConverter._draw_line_upwards_from_point(pin_hole_mask, line_start,
                                                                                 fill_value=0)
        return pin_hole_mask

    # pylint: disable=too-many-locals
    # noinspection DuplicatedCode
    @staticmethod
    def _smoothen_contours(contours: Tuple[np.ndarray]) -> Tuple[np.ndarray]:
        """Smoothen the contours by applying a heuristic filter to the contours.

        Args:
            contours (Tuple[np.ndarray]): The contours to be smoothened.

        Returns:
            Tuple[np.ndarray]: The smoothened contours.
        """
        smoothened = []
        for contour in contours:
            x, y = contour.T
            x = x.tolist()[0]
            y = y.tolist()[0]

            num_points = len(x)

            if len(x) >= 1:
                while x[0] == x[-1] or y[0] == y[-1]:
                    if len(x) <= 1:
                        break
                    x = x[:-1]
                    y = y[:-1]

            if len(x) < 6:
                res_array = [[[int(i[0]), int(i[1])]] for i in zip(x, y)]
                smoothened.append(np.asarray(res_array, dtype=np.int32))
                continue

            # preprocess the points if there are sufficient to facilitate smoothing
            if len(x) > 10:
                x_pp = x[::2]
                y_pp = y[::2]

                # remove coordinate duplicates
                unique_coords = []
                for x_i, y_i in zip(x_pp, y_pp):
                    entry = [x_i, y_i]
                    if entry not in unique_coords:
                        unique_coords.append(entry)

                x_pp = [entry[0] for entry in unique_coords]
                y_pp = [entry[1] for entry in unique_coords]
            else:
                x_pp = x
                y_pp = y

            # perform the smoothing using a b-spline approach
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    # noinspection PyTupleAssignmentBalance
                    tck, u = splprep([x_pp, y_pp], u=None, s=1.0, per=1)
                    u_new = np.linspace(np.min(u), np.max(u), int(1.25 * num_points))
                    x_new, y_new = splev(u_new, tck, der=0)

                except ValueError:
                    # noinspection PyTupleAssignmentBalance
                    tck, u = splprep([x, y], u=None, s=1.0, per=1)
                    u_new = np.linspace(np.min(u), np.max(u), int(1.25 * num_points))
                    x_new, y_new = splev(u_new, tck, der=0)

            res_array = [[[float(i[0]), float(i[1])]] for i in zip(x_new, y_new)]
            smoothened.append(np.asarray(res_array, dtype=float))

        return tuple(smoothened)

    @staticmethod
    def _find_mask_contours(mask: np.ndarray,
                            approximate_contours: bool
                            ) -> Tuple[List[np.ndarray], List]:
        """Find the contours in the provided mask.

        Args:
            mask (np.ndarray): The mask to be used for finding the contours.
            approximate_contours (bool): Whether to approximate the contours.

        Returns:
            Tuple[List[np.ndarray], List]: The contours and the hierarchy.
        """
        approximation_method = cv.CHAIN_APPROX_SIMPLE if approximate_contours else cv.CHAIN_APPROX_NONE
        contours, hierarchy = cv.findContours(mask.astype(np.uint8), cv.RETR_TREE, approximation_method)
        contours = SegmentToRTSSConverter._smoothen_contours(contours)
        contours = list(contours)
        # Format extra array out of data
        for i, contour in enumerate(contours):
            contours[i] = [[pos[0][0], pos[0][1]] for pos in contour]
        hierarchy = hierarchy[0]  # Format extra array out of data

        return contours, hierarchy

    @staticmethod
    def _draw_line_upwards_from_point(mask: np.ndarray,
                                      start: Tuple[int, ...],
                                      fill_value: int
                                      ) -> np.ndarray:
        """Draw a line upwards from the given point in the mask.

        Args:
            mask (np.ndarray): The mask to be used for drawing the line.
            start (Tuple[int, ...]): The start point of the line.
            fill_value (int): The value to be used for filling the line.

        Returns:
            np.ndarray: The mask with the line drawn.
        """
        line_width = 2
        end = (start[0], start[1] - 1)
        mask = mask.astype(np.uint8)  # type that OpenCV expects

        # draw one point at a time until we hit a point that already has the desired value
        while mask[end] != fill_value:
            cv.line(mask, start, end, fill_value, line_width)

            # update start and end to the next positions
            start = end
            end = (start[0], start[1] - line_width)
        return mask.astype(bool)

    @staticmethod
    def _apply_transformation_to_3d_points(points: np.ndarray,
                                           transformation_matrix: np.ndarray
                                           ) -> np.ndarray:
        """Apply the given transformation matrix to the given 3D points.

        Notes:
            The transformation matrix is expected to be a 4x4 matrix (homogeneous coordinate transform).


        Args:
            points (np.ndarray): The points to be transformed.
            transformation_matrix (np.ndarray): The transformation matrix.

        Returns:
            np.ndarray: The transformed points.
        """
        vec = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
        return vec.dot(transformation_matrix.T)[:, :3]

    @staticmethod
    def _create_contour_sequence_entry(series_slice: Dataset,
                                       contour_data: List[float]
                                       ) -> Dataset:
        """Create a contour sequence entry for the given slice.

        Args:
            series_slice (Dataset): The slice to be used for creating the contour sequence entry.
            contour_data (List[float]): The contour data to be used for creating the contour sequence entry.

        Returns:
            Dataset: The contour sequence entry.
        """
        # create the ContourImageSequence entry
        contour_image = Dataset()
        contour_image.ReferencedSOPClassUID = series_slice.file_meta.MediaStorageSOPClassUID
        contour_image.ReferencedSOPInstanceUID = series_slice.file_meta.MediaStorageSOPInstanceUID

        # construct the ContourImageSequence
        contour_image_sequence = Sequence()
        contour_image_sequence.append(contour_image)

        # construct the ContourSequence entry
        contour = Dataset()
        contour.ContourImageSequence = contour_image_sequence
        contour.ContourGeometricType = 'CLOSED_PLANAR'
        contour.NumberOfContourPoints = len(contour_data) // 3  # each contour point consists of an x, y, and z value
        contour.ContourData = contour_data

        return contour

    @staticmethod
    def _create_structure_set_roi(roi_data: ROIData,
                                  rtss: Dataset
                                  ) -> None:
        """Create a StructureSetROISequence entry for the given ROI data.

        Args:
            roi_data (ROIData): The ROI data to be used for creating the StructureSetROISequence entry.
            rtss (Dataset): The RTSS to be used for creating the StructureSetROISequence entry.

        Returns:
            None
        """
        # generate the StructureSetROISequence entry
        structure_set_roi_entry = Dataset()
        structure_set_roi_entry.ROINumber = roi_data.number
        structure_set_roi_entry.ReferencedFrameOfReferenceUID = roi_data.frame_of_reference_uid
        structure_set_roi_entry.ROIName = roi_data.name
        structure_set_roi_entry.ROIDescription = roi_data.description
        structure_set_roi_entry.ROIGenerationAlgorithm = roi_data.roi_generation_algorithm

        # add the StructureSetROISequence entry to the RTSS
        rtss.StructureSetROISequence.append(structure_set_roi_entry)

    @staticmethod
    def _create_rt_roi_observation(roi_data: ROIData,
                                   rtss: Dataset
                                   ) -> None:
        """Create a RTROIObservationsSequence entry for the given ROI data.

        Args:
            roi_data (ROIData): The ROI data to be used for creating the RTROIObservationsSequence entry.
            rtss (Dataset): The RTSS to be used for creating the RTROIObservationsSequence entry.

        Returns:
            None
        """
        # generate the RTROIObservationsSequence entry
        rt_roi_observation = Dataset()
        rt_roi_observation.ObservationNumber = roi_data.number
        rt_roi_observation.ReferencedROINumber = roi_data.number
        rt_roi_observation.ROIObservationDescription = 'Type:Soft,Range:*/*,Fill:0,Opacity:0.0,Thickness:1,' \
                                                       'LineThickness:2,read-only:false'
        rt_roi_observation.private_creators = 'University of Bern, Switzerland'
        rt_roi_observation.RTROIInterpretedType = ''
        rt_roi_observation.ROIInterpreter = ''

        # add the RTROIObservationsSequence entry to the RTSS
        rtss.RTROIObservationsSequence.append(rt_roi_observation)

    def convert(self) -> Dataset:
        """Convert the provided :class:`SimpleITK.Image` instances to a DICOM-RTSS :class:`~pydicom.dataset.Dataset`
        instance.

        Returns:
            Dataset: The generated DICOM-RTSS :class:`~pydicom.dataset.Dataset`.
        """
        # generate the basic RTSS dataset
        rtss = self._generate_basic_rtss(self.image_datasets)

        # convert and add the ROIs to the RTSS
        frame_of_reference_uid = self.image_datasets[0].get('FrameOfReferenceUID')
        for idx, (label_image, label_name, color) in enumerate(zip(self.label_images, self.roi_names, self.colors)):
            mask = sitk.GetArrayFromImage(label_image)

            if not issubclass(mask.dtype.type, np.integer):
                raise TypeError('The label image must be of an integer type!')

            roi_data = ROIData(mask, list(color), idx + 1, label_name, frame_of_reference_uid)

            self._create_roi_contour(roi_data, self.image_datasets, rtss)
            self._create_structure_set_roi(roi_data, rtss)
            self._create_rt_roi_observation(roi_data, rtss)

        return rtss


def main_2():
    file_name = 'E:/conversion_experiment/nifti_0/ISAS_GBM_001/seg_ISAS_GBM_001_Daniel_Schmidhalter_Brain.nii.gz'
    dicom_dir = 'E:/DataBackups/2020_08_ISAS_OAR_test/ISAS_GBM_001/'
    import itk

    # image = itk.imread(file_name)
    image = sitk.ReadImage(file_name)

    from pyradise.fileio.crawling import SubjectDicomCrawler

    series_info = SubjectDicomCrawler(dicom_dir).execute()
    t1c_series_info = [info for info in series_info if isinstance(info, DicomSeriesImageInfo) and info.modality.name == 'T1c'][0]
    image_datasets = load_datasets(t1c_series_info.path)

    converter = SegmentToRTSSConverter((image,), image_datasets, ('Brain',), ((255, 0, 0),))
    converter.convert()

    print('finished')


def main():
    # file_name = 'D:/temp/oar_auto.nii.gz'
    # file_name = 'D:/DataBackupsConversion/20210105_ISAS_OAR_conversion_small/ISAS_GBM_009/' \
    #             'seg_ISAS_GBM_009_RP_Hippocampus_R.nii.gz'
    file_name = 'D:/DataBackupsConversion/20210105_ISAS_OAR_conversion_small/ISAS_GBM_009/' \
                'seg_ISAS_GBM_009_RP_Brainstem.nii.gz'
    file_name = 'E:/conversion_experiment/nifti_0/ISAS_GBM_001/seg_ISAS_GBM_001_Daniel_Schmidhalter_Brain.nii.gz'
    dicom_dir = 'E:/DataBackups/2020_08_ISAS_OAR_test/ISAS_GBM_001/'

    sitk_image = sitk.ReadImage(file_name)

    sitk_image = sitk.DICOMOrient(sitk_image, 'LPS')

    from pyradise.fileio.crawling import SubjectDicomCrawler

    series_info = SubjectDicomCrawler(dicom_dir).execute()
    t1c_series_info = \
    [info for info in series_info if isinstance(info, DicomSeriesImageInfo) and info.modality.name == 'T1c'][0]
    image_datasets = load_datasets(t1c_series_info.path)

    # reader = vtkNIFTIImageReader()
    # reader.SetFileName(str(file_name))
    # reader.Update(0)

    import itk
    image_itk = convert_to_itk_image(sitk_image)
    image_vtk = itk.vtk_image_from_image(image_itk)

    image_vtk.SetDirectionMatrix(sitk_image.GetDirection())

    select_tissue = vtkImageThreshold()
    select_tissue.ThresholdBetween(1, 1)
    select_tissue.SetInValue(255)
    select_tissue.SetOutValue(0)
    # select_tissue.SetInputConnection(reader.GetOutputPort())
    select_tissue.SetInputData(image_vtk)

    gaussian_radius = 1
    gaussian_standard_deviation = 2.0
    gaussian = vtkImageGaussianSmooth()
    gaussian.SetStandardDeviations(gaussian_standard_deviation, gaussian_standard_deviation,
                                   gaussian_standard_deviation)
    gaussian.SetRadiusFactors(gaussian_radius, gaussian_radius, gaussian_radius)
    gaussian.SetInputConnection(select_tissue.GetOutputPort())

    iso_surface = vtkFlyingEdges3D()
    iso_surface.SetInputConnection(gaussian.GetOutputPort())
    iso_surface.ComputeScalarsOff()
    iso_surface.ComputeGradientsOff()
    iso_surface.ComputeNormalsOff()
    iso_value = 127.5
    iso_surface.SetValue(0, iso_value)

    smoothing_iterations = 5
    pass_band = 0.001
    feature_angle = 60.0
    smoother = vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(iso_surface.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOff()
    smoother.Update()

    normals = vtkPolyDataNormals()
    normals.SetInputConnection(smoother.GetOutputPort())
    normals.SetFeatureAngle(feature_angle)

    stripper = vtkStripper()
    stripper.SetInputConnection(normals.GetOutputPort())
    stripper.Update()

    writer = vtkSTLWriter()
    writer.SetFileName('C:/temp/brain_lps.stl')
    writer.SetInputConnection(stripper.GetOutputPort())
    writer.Write()

    contour_actor = get_contours(stripper.GetOutput(), sitk_image, image_datasets)

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(stripper.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)

    renderer_left = vtkRenderer()
    renderer_left.SetViewport(0, 0, 0.5, 1)
    renderer_right = vtkRenderer()
    renderer_right.SetViewport(0.5, 0, 1, 1)
    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer_left)
    render_window.AddRenderer(renderer_right)
    render_window_interactor = vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    renderer_left.AddActor(actor)
    renderer_right.AddActor(contour_actor)

    render_window.SetSize(640, 480)
    render_window.SetWindowName('FrogBrain')
    render_window.Render()

    render_window_interactor.Start()


def get_contours(polydata: vtkPolyData, sitk_image: sitk.Image, image_datasets: Tuple[Dataset]) -> vtkActor:
    bounds = polydata.GetBounds()
    step_size = 0.5

    # minimum_value = bounds[2] - step_size
    # maximum_value = bounds[3] + step_size


    # for i, image_dataset in enumerate(image_datasets):
    #
    #     if i < 80:
    #         continue
    #
    #     plane = vtkPlane()
    #     plane.SetOrigin(image_dataset.get('ImagePositionPatient'))
    #
    #     normal = np.cross(np.array(image_dataset.get('ImageOrientationPatient')[0:3]),
    #                       np.array(image_dataset.get('ImageOrientationPatient')[3:6]))
    #
    #     plane.SetNormal(normal)
    #
    #     cutter = vtkCutter()
    #     cutter.SetInputData(polydata)
    #     cutter.SetCutFunction(plane)
    #     cutter.Update()
    #
    #     cutStrips = vtkStripper()
    #     cutStrips.SetInputConnection(cutter.GetOutputPort())
    #     cutStrips.Update()
    #
    #     cutPolyData = cutStrips.GetOutput()
    #     print(f'cutPolyData.GetNumberOfPoints() {i}: {cutPolyData.GetNumberOfPoints()}')
    #
    #     cleanDataFilter = vtkCleanPolyData()
    #     cleanDataFilter.AddInputData(cutStrips.GetOutput())
    #     cleanDataFilter.Update()
    #     cleanData = cleanDataFilter.GetOutput()
    #
    #
    #     print(f'Number of points in slice {i}: {cutter.GetOutput().GetNumberOfPoints()}')



    # position = minimum_value

    # while position < maximum_value:
    #     plane = vtkPlane()
    #     plane.SetOrigin((bounds[0] + bounds[1]) // 2,
    #                     (bounds[2] + bounds[3]) // 2,
    #                     position)
    #     plane.SetNormal(1, 0, 0)
    #
    #     cutter = vtkCutter()
    #     cutter.SetInputData(polydata)
    #     cutter.SetCutFunction(plane)
    #     cutter.Update()
    #     all_contours.append(cutter.GetOutput())
    #
    #     position += step_size

    # append = vtkAppendPolyData()
    # [append.AddInputData(contour) for contour in all_contours]
    # append.Update()

    first_pos = image_datasets[0].get('ImagePositionPatient')
    last_pos = image_datasets[-1].get('ImagePositionPatient')

    length = np.abs(np.linalg.norm(np.array(last_pos) - np.array(first_pos)))
    spacing = length / len(image_datasets)


    plane = vtkPlane()
    plane.SetOrigin(image_datasets[0].get('ImagePositionPatient'))

    normal = np.cross(np.array(image_datasets[0].get('ImageOrientationPatient')[0:3]),
                      np.array(image_datasets[0].get('ImageOrientationPatient')[3:6]))
    plane.SetNormal(normal)

    cutter = vtkCutter()
    cutter.SetInputData(polydata)
    cutter.SetCutFunction(plane)
    # cutter.GenerateValues(100, -125, 0)
    cutter.GenerateValues(len(image_datasets), 0, length)
    cutter.Update()

    # plane = vtkPlane()
    # plane.SetOrigin(bounds[0] - 1, bounds[2] - 1, bounds[4] - 1)
    # plane.SetNormal(normal)

    # cutter = vtkCutter()
    # cutter.SetInputData(polydata)
    # cutter.SetCutFunction(plane)
    # cutter.GenerateValues(50, minimum_value, maximum_value)
    # cutter.Update()

    cleaner = vtkCleanPolyData()
    cleaner.SetInputData(cutter.GetOutput())
    cleaner.Update()

    mapper = vtkPolyDataMapper()
    mapper.SetInputData(cleaner.GetOutput())
    mapper.ScalarVisibilityOff()

    colors = vtkNamedColors()

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d('Orange'))
    actor.GetProperty().SetLineWidth(2)

    return actor


# def get_dicom_series_info(path: str):



if __name__ == '__main__':
    main()
    # main_2()
