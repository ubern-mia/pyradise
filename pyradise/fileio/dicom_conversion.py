from abc import (
    ABC,
    abstractmethod)
from typing import (
    Any,
    Tuple,
    Optional,
    Union,
    List,
    Dict)
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass
from enum import IntEnum
import warnings

import numpy as np
import cv2 as cv
import SimpleITK as sitk
from scipy.interpolate import (
    splprep,
    splev)
from pydicom import (
    Dataset,
    FileDataset,
    Sequence)
from pydicom.uid import (
    generate_uid,
    ImplicitVRLittleEndian,
    PYDICOM_IMPLEMENTATION_UID)
from pydicom.dataset import FileMetaDataset
from pydicom.tag import Tag

from pyradise.data import (
    Subject,
    IntensityImage,
    SegmentationImage,
    Organ,
    Modality)
from pyradise.utils import (
    load_datasets,
    load_dataset,
    load_dataset_tag,
    chunkify,
    get_slice_position,
    get_slice_direction,
    get_spacing_between_slices)
from .series_info import (
    DicomSeriesInfo,
    DicomSeriesImageInfo,
    DicomSeriesRegistrationInfo,
    DicomSeriesRTSSInfo,
    RegistrationInfo)

__all__ = ['Converter', 'DicomImageSeriesConverter', 'DicomRTSSSeriesConverter', 'SubjectToRTSSConverter',
           'RTSSToSegmentConverter', 'SegmentToRTSSConverter']

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


# noinspection PyUnresolvedReferences
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


class Converter(ABC):
    """An abstract base class for all :class:`Converter` classes. Typically, the :class:`Converter` classes are used to
    convert DICOM data from and to other representations. For example, the :class:`DicomImageSeriesConverter` converts
    DICOM image series to :class:`~pyradise.data.image.IntensityImage` instances and applies the associated DICOM
    registration if provided.
    """

    @abstractmethod
    def convert(self) -> Any:
        """Convert the provided :class:`~pyradise.fileio.series_info.DicomSeriesInfo`.

        Returns:
            Any: The converted data.
        """
        raise NotImplementedError()


class RTSSToSegmentConverter(Converter):
    """A low-level DICOM-RTSS to SimpleITK image :class:`Converter` class converting the content of the DICOM-RTSS to
    one or multiple SimpleITK images. In contrast to the :class:`DicomRTSSSeriesConverter` this class generates a dict
    of binary :class:`SimpleITK.Image` instances and organ names instead of a tuple of
    :class:`~pyradise.data.image.SegmentationImage` s.

    Notes:
        Typically, this class is not used directly by the user but via the :class:`DicomRTSSSeriesConverter` which
        processes :class:`~pyradise.fileio.series_info.DicomSeriesInfo` entries directly.

        This class can be used with a DICOM registration which will be applied to the reference image and the structure
        set. However, the registration must reference the corresponding DICOM image, otherwise the registration will not
        be applied.

    Args:
        rtss_dataset (Union[str, Dataset]): The path to the DICOM-RTSS file or DICOM-RTSS
         :class:`~pydicom.dataset.Dataset`.
        image_datasets (Union[Tuple[str], Tuple[Dataset]]): The path to the DICOM image files or the DICOM image
         :class:`~pydicom.dataset.Dataset` entries which are referenced in the ``rtss_dataset``.
        registration_dataset (Union[str, Dataset, None]): The path to a DICOM registration file or a DICOM
         registration :class:`~pydicom.dataset.Dataset` entry which contains a reference to the DICOM image
         (default: None).
    """

    def __init__(self,
                 rtss_dataset: Union[str, Dataset],
                 image_datasets: Union[Tuple[str], Tuple[Dataset]],
                 registration_dataset: Union[str, Dataset, None] = None
                 ) -> None:
        super().__init__()

        # get the RTSS dataset and the referenced SeriesInstanceUID
        if isinstance(rtss_dataset, str):
            self.rtss_dataset: Dataset = load_dataset(rtss_dataset)
        else:
            self.rtss_dataset: Dataset = rtss_dataset
        ref_series_uid = self._get_ref_series_instance(self.rtss_dataset)

        # get the appropriate image datasets according to the RTSS
        if isinstance(image_datasets[0], str):
            self.rtss_image_datasets = self._load_ref_image_datasets(image_datasets, ref_series_uid)
        else:
            self.rtss_image_datasets = self._clean_image_datasets_for_rtss(image_datasets, ref_series_uid)

        # get the registration dataset and the appropriate images
        if not registration_dataset:
            self.reg_dataset: Optional[Dataset] = None
        elif isinstance(registration_dataset, str):
            self.reg_dataset: Optional[Dataset] = load_dataset(registration_dataset)
        elif isinstance(registration_dataset, Dataset):
            self.reg_dataset: Optional[Dataset] = registration_dataset
        self.reg_image_datasets = self._get_image_datasets_for_reg(image_datasets, self.reg_dataset)

        # validate the loaded data
        self._validate_rtss_dataset(self.rtss_dataset)
        self._validate_rtss_image_references(self.rtss_dataset, self.rtss_image_datasets)
        self.reg_dataset: Optional[Dataset] = self._validate_registration_dataset(self.reg_dataset, self.rtss_dataset)

    @staticmethod
    def _get_ref_series_instance(rtss_dataset: Dataset) -> str:
        """Get the referenced SeriesInstanceUID of the image series in the RTSS dataset.

        Args:
            rtss_dataset (Dataset): The RTSS dataset.

        Returns:
            str: The referenced SeriesInstanceUID of the image series.
        """
        ref_series_instance_uids = []
        for ref_frame_of_ref in rtss_dataset.get('ReferencedFrameOfReferenceSequence', []):
            for rt_ref_study in ref_frame_of_ref.get('RTReferencedStudySequence', []):
                for rt_ref_series in rt_ref_study.get('RTReferencedSeriesSequence', []):
                    si_uid = rt_ref_series.get('SeriesInstanceUID', None)
                    if si_uid is not None:
                        ref_series_instance_uids.append(str(si_uid))

        if len(ref_series_instance_uids) > 1:
            raise Exception(f'Multiple ({len(ref_series_instance_uids)}) referenced SeriesInstanceUIDs '
                            'have been retrieved from the RTSS but only one is allowed!')

        if not ref_series_instance_uids:
            raise Exception('No referenced SeriesInstanceUID could be retrieved!')

        return ref_series_instance_uids[0]

    @staticmethod
    def _load_ref_image_datasets(image_paths: Tuple[str],
                                 referenced_series_uid: str
                                 ) -> Tuple[Dataset]:
        """Load the appropriate image datasets which are referenced in the RTSS dataset.

        Args:
            image_paths (Tuple[str]): The paths to the image datasets.
            referenced_series_uid (str): The referenced SeriesInstanceUID.

        Returns:
            Tuple[Dataset]: The referenced image datasets.
        """
        # check all image file paths for the referenced SeriesInstanceUID
        ref_image_files = []
        for image_path in image_paths:
            image_dataset = load_dataset_tag(image_path, (Tag(0x0020, 0x000e),))
            if image_dataset.get('SeriesInstanceUID', '') == referenced_series_uid:
                ref_image_files.append(image_path)

        # load the appropriate image datasets
        ref_image_datasets = []
        for ref_image_file in ref_image_files:
            ref_image_datasets.append(load_dataset(ref_image_file))

        ref_image_datasets.sort(key=get_slice_position, reverse=False)

        return tuple(ref_image_datasets)

    @staticmethod
    def _clean_image_datasets_for_rtss(image_datasets: Tuple[Dataset],
                                       referenced_series_uid: str
                                       ) -> Tuple[Dataset]:
        """Clean the image datasets based on the referenced SeriesInstanceUID.

        Args:
            image_datasets (Tuple[Dataset]): The image Datasets which should be analyzed.
            referenced_series_uid (str): The referenced SeriesInstanceUID to identify the appropriate images.

        Returns:
            Tuple[Dataset]: The image datasets which are referenced by the RTSS Dataset.
        """
        selected = [image_dataset for image_dataset in image_datasets
                    if image_dataset.get('SeriesInstanceUID', None) == referenced_series_uid]

        selected.sort(key=get_slice_position, reverse=False)

        return tuple(selected)

    @staticmethod
    def _get_image_datasets_for_reg(image_datasets: Union[Tuple[Dataset, ...], Tuple[str, ...]],
                                    registration_dataset: Optional[Dataset]
                                    ) -> Optional[Tuple[Dataset, ...]]:
        """Get the image datasets for registration. This function outputs the datasets which are referenced within
         the registration dataset.

        Args:
            image_datasets (Union[Tuple[Dataset, ...], Tuple[str, ...]]): The image datasets to analyze.
            registration_dataset (Optional[Dataset]): The registration dataset containing the references.

        Returns:
            Optional[Tuple[Dataset, ...]]: The datasets being referenced within the registration.
        """
        if not registration_dataset:
            return None

        selected = []

        ref_image_series = DicomSeriesRegistrationInfo.get_referenced_series_info(registration_dataset)
        ref_series_uids = [ref_info.series_instance_uid for ref_info in ref_image_series]
        ref_study_uids = [ref_info.study_instance_uid for ref_info in ref_image_series]

        if isinstance(image_datasets[0], str):
            for path in image_datasets:
                dataset = load_dataset_tag(path, (Tag(0x0020, 0x000e), Tag(0x0020, 0x000d)))
                criteria = (str(dataset.get('SeriesInstanceUID', '')) in ref_series_uids,
                            str(dataset.get('StudyInstanceUID', '')) in ref_study_uids)
                if all(criteria):
                    selected.append(load_dataset(path))

        else:
            for image_dataset in image_datasets:
                criteria = (str(image_dataset.get('SeriesInstanceUID', '')) in ref_series_uids,
                            str(image_dataset.get('StudyInstanceUID', '')) in ref_study_uids)
                if all(criteria):
                    selected.append(image_dataset)

        return tuple(selected)

    @staticmethod
    def _validate_rtss_dataset(rtss_dataset: Dataset) -> None:
        """Validate if the RTSS dataset is containing the minimal data for conversion.

        Args:
            rtss_dataset (Dataset): The RTSS dataset to validate.

        Returns:
            None
        """
        criteria = (rtss_dataset.get('SOPClassUID', '') == '1.2.840.10008.5.1.4.1.1.481.3',
                    hasattr(rtss_dataset, 'ROIContourSequence'),
                    hasattr(rtss_dataset, 'StructureSetROISequence'),
                    hasattr(rtss_dataset, 'RTROIObservationsSequence'))

        if not all(criteria):
            raise Exception(f'The checked RTSS from subject {rtss_dataset.get("PatientID")} is invalid!')

    # pylint: disable=use-a-generator
    @staticmethod
    def _validate_registration_dataset(reg_dataset: Optional[Dataset],
                                       rtss_dataset: Dataset
                                       ) -> Optional[Dataset]:
        """Validate the registration dataset if it contains the minimally required data for conversion. If the provided
        information is insufficient this function will return None.

        Args:
            reg_dataset (Optional[Dataset]): The registration dataset to validate.
            rtss_dataset (Dataset): The RTSS dataset containing the references.

        Returns:
            Optional[Dataset]: The valid registration dataset or None.
        """

        if not reg_dataset:
            return None

        # search for references in the registration dataset
        reg_ref_instance_uids = []
        for ref_study_item in reg_dataset.get('StudiesContainingOtherReferencedInstancesSequence', []):
            for ref_series_item in ref_study_item.get('ReferencedSeriesSequence', []):
                ref_series_instance_uid = ref_series_item.get('SeriesInstanceUID', None)
                if ref_series_instance_uid:
                    reg_ref_instance_uids.append(ref_series_instance_uid)

        for ref_series_item in reg_dataset.get('ReferencedSeriesSequence', []):
            ref_series_uid = ref_series_item.get('SeriesInstanceUID', None)
            if ref_series_uid:
                reg_ref_instance_uids.append(ref_series_uid)

        # search for references in the rtss dataset
        rtss_ref_instance_uids = []
        for ref_frame_of_ref in rtss_dataset.get('ReferencedFrameOfReferenceSequence', []):
            for rt_ref_study in ref_frame_of_ref.get('RTReferencedStudySequence', []):
                for rt_ref_series in rt_ref_study.get('RTReferencedSeriesSequence', []):
                    ref_series_uid = rt_ref_series.get('SeriesInstanceUID', None)
                    if ref_series_uid:
                        rtss_ref_instance_uids.append(ref_series_uid)

        # check the criteria
        criteria = (reg_dataset.get('SOPClassUID', '') == '1.2.840.10008.5.1.4.1.1.66.1',
                    hasattr(reg_dataset, 'RegistrationSequence'),
                    hasattr(reg_dataset, 'ReferencedSeriesSequence'),
                    all([rt_reference in reg_ref_instance_uids for
                         rt_reference in rtss_ref_instance_uids]),
                    len(reg_ref_instance_uids) != 0)

        if not all(criteria):
            print(f'The checked registration from subject {rtss_dataset.get("PatientID", "n/a")} is invalid!')
            return None

        return reg_dataset

    @staticmethod
    def _validate_rtss_image_references(rtss_dataset: Dataset,
                                        image_datasets: Tuple[Dataset]
                                        ) -> None:
        """Validate if the ReferencedSOPInstanceUIDs of the RTSS dataset are contained in the image datasets.

        Args:
            rtss_dataset (Dataset): The RTSS dataset to validate.
            image_datasets (Tuple[Dataset]): The image datasets to be used for comparison.

        Returns:
            None
        """
        # get the ReferencedSOPInstanceUIDs from the RTSS dataset
        ref_sop_instance_uids = []
        for ref_frame_of_ref in rtss_dataset.get('ReferencedFrameOfReferenceSequence', []):
            for rt_ref_study in ref_frame_of_ref.get('RTReferencedStudySequence', []):
                for rt_ref_series in rt_ref_study.get('RTReferencedSeriesSequence', []):
                    for contour_image_entry in rt_ref_series.get('ContourImageSequence', []):
                        ref_sop_instance_uids.append(contour_image_entry.get('ReferencedSOPInstanceUID', ''))

        # get MediaStorageSOPInstanceUIDs from the image datasets
        image_sop_instance_uids = [entry.file_meta.get('MediaStorageSOPInstanceUID') for entry in image_datasets]

        # check if all ReferencedSOPInstanceUIDs are contained in the image datasets
        missing_sop_instance_uids = tuple(set(ref_sop_instance_uids) - set(image_sop_instance_uids))
        if missing_sop_instance_uids:
            raise ValueError('The following ReferencedSOPInstanceUIDs are missing in the image datasets: '
                             f'{missing_sop_instance_uids}')

    @staticmethod
    def _get_contour_sequence_by_roi_number(rtss_dataset: Dataset,
                                            roi_number: int
                                            ) -> Optional[Sequence]:
        """Get the ContourSequence by the ROINumber.

        Args:
            rtss_dataset (Dataset): The RT Structure Set dataset to retrieve the ContourSequence from.
            roi_number (int): The ROINumber for which the ContourSequence should be returned.

        Returns:
            Optional[Sequence]: The ContourSequence with the corresponding ROINumber if available, otherwise
             :class:`None`.
        """
        if rtss_dataset.get('ROIContourSequence', None) is None:
            return None

        for roi_contour_entry in rtss_dataset.get('ROIContourSequence', []):
            if str(roi_contour_entry.get('ReferencedROINumber', None)) == str(roi_number):
                if roi_contour_entry.get('ContourSequence', None) is None:
                    return None
                return roi_contour_entry.get('ContourSequence')

        raise Exception(f"Referenced ROI number '{roi_number}' not found")

    @staticmethod
    def _create_empty_series_mask(image_datasets: Tuple[Dataset, ...]) -> np.ndarray:
        """Create an empty numpy array with the shape according to the reference image.

        Returns:
            np.ndarray: The empty numpy array with appropriate shape.
        """
        rows = int(image_datasets[0].get('Rows'))
        columns = int(image_datasets[0].get('Columns'))
        num_slices = len(image_datasets)

        return np.zeros((columns, rows, num_slices)).astype(np.bool)

    @staticmethod
    def _create_empty_slice_mask(image_dataset: Dataset) -> np.ndarray:
        """Create an empty numpy array representing one slice of the output mask.

        Args:
            image_dataset (Dataset): The dataset providing the spatial information for the empty mask.

        Returns:
            np.ndarray: The empty slice numpy array.
        """
        columns = int(image_dataset.get('Columns'))
        rows = int(image_dataset.get('Rows'))
        return np.zeros((columns, rows)).astype(np.uint8)

    @staticmethod
    def _apply_transformation_to_3d_points(points: np.ndarray,
                                           transformation_matrix: np.ndarray
                                           ) -> np.ndarray:
        """Apply the provided transformation to multiple points in the 3D-space.

        Args:
            points (np.ndarray): The points to transform.
            transformation_matrix (np.ndarray): The transformation matrix.

        Notes:
            1. Augment each point with a '1' as the fourth coordinate for homogeneous coordinates.
            2. Multiply by a 4x4 transformation matrix
            3. Throw away the adaptation for homogeneous coordinates

        Returns:
            np.ndarray: The transformed points.
        """
        vec = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
        return vec.dot(transformation_matrix.T)[:, :3]

    @staticmethod
    def _get_contour_fill_mask(series_slice: Dataset,
                               contour_coords: List[float],
                               transformation_matrix: np.ndarray
                               ) -> np.ndarray:
        """Get the mask with the filled contour.

        Args:
            series_slice (Dataset): The corresponding slice dataset.
            contour_coords (List[float]): The contour points.
            transformation_matrix (np.ndarray): The transformation matrix.

        Returns:
            np.ndarray: The mask with the filled contour.
        """
        reshaped_contour_data = np.reshape(contour_coords, [len(contour_coords) // 3, 3])
        translated_contour_data = RTSSToSegmentConverter._apply_transformation_to_3d_points(reshaped_contour_data,
                                                                                            transformation_matrix)
        polygon = [np.around([translated_contour_data[:, :2]]).astype(np.int32)]

        # Create mask for the region. Fill with 1 for ROI
        fill_mask = RTSSToSegmentConverter._create_empty_slice_mask(series_slice)
        cv.fillPoly(img=fill_mask, pts=polygon, color=1)
        return fill_mask

    @staticmethod
    def _get_patient_to_pixel_transformation_matrix(image_datasets: Tuple[Dataset, ...]) -> np.ndarray:
        """Get the patient to pixel transformation matrix from the first image dataset.

        Args:
            image_datasets (Tuple[Dataset, ...]): The datasets to retrieve the transformation matrix.

        Returns:
            np.ndarray: The transformation matrix.
        """
        offset = np.array(image_datasets[0].get('ImagePositionPatient'))
        row_spacing, column_spacing = image_datasets[0].get('PixelSpacing')
        slice_spacing = get_spacing_between_slices(image_datasets)
        row_direction, column_direction, slice_direction = get_slice_direction(image_datasets[0])

        linear = np.identity(3, dtype=np.float32)
        linear[0, :3] = column_direction / column_spacing
        linear[1, :3] = row_direction / row_spacing
        linear[2, :3] = slice_direction / slice_spacing

        mat = np.identity(4, dtype=np.float32)
        mat[:3, :3] = linear
        mat[:3, 3] = offset.dot(-linear.T)

        return mat

    @staticmethod
    def _get_slice_contour_data(image_dataset: Dataset,
                                contour_sequence: Sequence
                                ) -> Tuple[Any, ...]:
        """Get the contour data from the corresponding ContourSequence.

        Args:
            image_dataset (Dataset): The referenced image dataset.
            contour_sequence (Sequence): The ContourSequence.

        Returns:
            Tuple[Any, ...]: The retrieved ContourData.
        """
        slice_contour_data = []

        for contour in contour_sequence:
            for contour_image in contour.get('ContourImageSequence', []):
                if contour_image.get('ReferencedSOPInstanceUID', None) == image_dataset.get('SOPInstanceUID', '1'):
                    slice_contour_data.append(contour.get('ContourData', []))

        return tuple(slice_contour_data)

    @staticmethod
    def _get_slice_mask_from_slice_contour_data(image_dataset: Dataset,
                                                contour_data: Tuple[Any, ...],
                                                transformation_matrix: np.ndarray
                                                ) -> np.ndarray:
        """Get the slice mask from the ContourData.

        Args:
            image_dataset (Dataset): The referenced image dataset.
            contour_data (Tuple[Any, ...]): The contour data.
            transformation_matrix (np.ndarray): The transformation matrix.

        Returns:
            np.ndarray: The discrete slice mask.
        """
        slice_mask = RTSSToSegmentConverter._create_empty_slice_mask(image_dataset)

        for contour_coords in contour_data:
            fill_mask = RTSSToSegmentConverter._get_contour_fill_mask(image_dataset, contour_coords,
                                                                      transformation_matrix)
            slice_mask[fill_mask == 1] = np.invert(slice_mask[fill_mask == 1])

        return slice_mask

    @staticmethod
    def _create_mask_from_contour_sequence(image_datasets: Tuple[Dataset, ...],
                                           contour_sequence: Sequence
                                           ) -> np.ndarray:
        """Create the whole 3D mask from the ContourSequence.

        Args:
            image_datasets (Tuple[Dataset, ...]): The image datasets to be used for mask creation.
            contour_sequence (Sequence): The ContourSequence to be discretized.

        Returns:
            np.ndarray: The discrete segmentation mask.
        """
        mask = RTSSToSegmentConverter._create_empty_series_mask(image_datasets)
        transformation_matrix = RTSSToSegmentConverter._get_patient_to_pixel_transformation_matrix(image_datasets)

        for i, image_dataset in enumerate(image_datasets):
            slice_contour_data = RTSSToSegmentConverter._get_slice_contour_data(image_dataset, contour_sequence)
            if slice_contour_data:
                mask[:, :, i] = RTSSToSegmentConverter._get_slice_mask_from_slice_contour_data(image_dataset,
                                                                                               slice_contour_data,
                                                                                               transformation_matrix)

        return mask

    @staticmethod
    def _create_image_from_mask(image_datasets: Tuple[Dataset, ...],
                                mask: np.ndarray
                                ) -> sitk.Image:
        """Create an image from the numpy segmentation mask with appropriate orientation.

        Args:
            image_datasets (Tuple[Dataset, ...]): The image datasets used to create the image.
            mask (np.ndarray): The mask to be converted into a sitk.Image.

        Returns:
            sitk.Image: The image generated.
        """
        mask = np.swapaxes(mask, 0, -1)

        image = sitk.GetImageFromArray(mask.astype(np.uint8))

        image.SetOrigin(image_datasets[0].get('ImagePositionPatient'))

        row_spacing, column_spacing = image_datasets[0].get('PixelSpacing')
        slice_spacing = get_spacing_between_slices(image_datasets)
        image.SetSpacing((float(row_spacing), float(column_spacing), float(slice_spacing)))

        slice_direction = np.stack(get_slice_direction(image_datasets[0]), axis=0).T.flatten().tolist()
        image.SetDirection(slice_direction)

        return image

    @staticmethod
    def _get_transform_from_registration_info(registration_infos: Tuple[RegistrationInfo, ...]) -> sitk.Transform:
        """Get the transformation from the registration infos.

        Args:
            registration_infos (Tuple[RegistrationInfo]): Registration info entries which hold transformations and
             references.

        Returns:
            sitk.Transform: The transformation which is not an identity transformation.
        """
        assert len(registration_infos) <= 2, 'The number of registration infos must be at max two but is ' \
                                             f'{len(registration_infos)}!'

        transforms = []
        for registration_info in registration_infos:
            if not registration_info.is_reference_image:
                transforms = registration_info.registration_info.transforms

        if len(transforms) != 1:
            raise NotImplementedError('The use of multiple sequential transformations is currently not supported!')

        return transforms[0]

    @staticmethod
    def _transform_image_datasets(image_datasets: Tuple[Dataset, ...],
                                  transform: sitk.Transform
                                  ) -> Tuple[Dataset, ...]:
        """Apply the transformation to multiple image datasets.

        Args:
            image_datasets (Tuple[Dataset, ...]): The image datasets to be transformed.
            transform (sitk.Transform): The transformation to be applied.

        Returns:
            Tuple[Dataset, ...]: The transformed image datasets.
        """

        transformed_image_datasets = []

        for image_dataset in image_datasets:
            # transform the patient position
            image_position_patient = image_dataset.get('ImagePositionPatient')
            position_transformed = list(transform.GetInverse().TransformPoint(image_position_patient))
            image_dataset['ImagePositionPatient'].value = position_transformed

            # transform the image orientation
            image_orientation_patient = image_dataset.get('ImageOrientationPatient')
            vector_0 = np.array(image_orientation_patient[:3])
            vector_1 = np.array(image_orientation_patient[3:6])

            vector_0_transformed = transform.GetInverse().TransformVector(vector_0, (0, 0, 0))
            vector_1_transformed = transform.GetInverse().TransformVector(vector_1, (0, 0, 0))

            new_direction = list(vector_0_transformed + vector_1_transformed)
            image_dataset['ImageOrientationPatient'].value = new_direction

            transformed_image_datasets.append(image_dataset)

        return image_datasets

    @staticmethod
    def _transform_rtss_dataset(rtss_dataset: Dataset,
                                transform: sitk.Transform
                                ) -> Dataset:
        """Apply the transformation to the RTSS.

        Args:
            rtss_dataset (Dataset): The dataset to be transformed.
            transform (sitk.Transform): The transformation to apply.

        Returns:
            Dataset: The transformed dataset.
        """
        dataset = deepcopy(rtss_dataset)

        for roi_contour in dataset.get('ROIContourSequence', []):
            for contour_sequence_item in roi_contour.get('ContourSequence', []):
                contour_data = contour_sequence_item.get('ContourData')

                if len(contour_data) % 3 != 0:
                    raise ValueError('The number of contour points must be a multiple of three!')

                transformed_points = []

                for chunk in chunkify(contour_data, 3):
                    transformed_point = list(transform.GetInverse().TransformPoint(np.array(chunk)))
                    transformed_points.extend(transformed_point)

                contour_sequence_item["ContourData"].value = transformed_points

        return dataset

    def convert(self) -> Dict[str, sitk.Image]:
        """Convert a DICOM-RTSS :class:`~pydicom.dataset.Dataset` instance into a dict of binary
        :class:`SimpleITK.Image` instances including their associated ROINames as a key in the dict.

        Returns:
            Dict[str, sitk.Image]: The ROINames and the corresponding binary segmented :class:`SimpleITK.Image`
            instances.
        """
        converted_images = {}

        # apply the registration if available
        if self.reg_dataset:
            reg_infos = DicomSeriesRegistrationInfo.get_registration_infos(self.reg_dataset,
                                                                           self.reg_image_datasets)
            transform = self._get_transform_from_registration_info(reg_infos)
            image_datasets = self._transform_image_datasets(self.rtss_image_datasets, transform)
            rtss_dataset = self._transform_rtss_dataset(self.rtss_dataset, transform)

        else:
            image_datasets = self.rtss_image_datasets
            rtss_dataset = self.rtss_dataset

        # convert the contours to images
        for ss_roi in rtss_dataset.get('StructureSetROISequence', []):
            roi_number = int(ss_roi.get('ROINumber'))
            roi_name = str(ss_roi.get('ROIName', ''))

            contour_sequence = self._get_contour_sequence_by_roi_number(rtss_dataset, roi_number)

            if contour_sequence is None:
                continue

            mask = self._create_mask_from_contour_sequence(image_datasets, contour_sequence)
            image = self._create_image_from_mask(image_datasets, mask)
            converted_images.update({roi_name: image})

        return converted_images


class SegmentToRTSSConverter(Converter):
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


class DicomImageSeriesConverter(Converter):
    """A :class:`Converter` class for converting DICOM image series to one or multiple
    :class:`~pyradise.data.image.IntensityImage` instances.

    Args:
        image_info (Tuple[DicomSeriesImageInfo, ...]): The :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo`
         entries of the images to convert.
        registration_info (Tuple[DicomSeriesRegistrationInfo, ...]): The
         :class:`~pyradise.fileio.series_info.DicomSeriesRegistrationInfo` entries (default: tuple()).
    """

    def __init__(self,
                 image_info: Tuple[DicomSeriesImageInfo, ...],
                 registration_info: Tuple[DicomSeriesRegistrationInfo, ...] = tuple()
                 ) -> None:
        super().__init__()
        self.image_info = image_info
        self.reg_info = registration_info

    def _get_image_info_by_series_instance_uid(self,
                                               series_instance_uid: str
                                               ) -> Optional[DicomSeriesImageInfo]:
        """Get the :class:`DicomSeriesImageInfo` entries which match with the specified SeriesInstanceUID.

        Args:
            series_instance_uid (str): The SeriesInstanceUID which must be contained in the returned
             :class:`DicomSeriesImageInfo` entries.

        Returns:
            Optional[DicomSeriesImageInfo]: The :class:`DicomSeriesImageInfo` entries which contains the specified
             SeriesInstanceUID or None
        """
        if not self.image_info:
            return None

        selected = [info for info in self.image_info if info.series_instance_uid == series_instance_uid]

        if len(selected) > 1:
            raise ValueError(f'Multiple image infos detected with the same SeriesInstanceUID ({series_instance_uid})!')

        if not selected:
            return None

        return selected[0]

    def _get_registration_info(self,
                               image_info: DicomSeriesImageInfo,
                               ) -> Optional[DicomSeriesRegistrationInfo]:
        """Get the :class:`DicomSeriesRegistrationInfo` instance which belongs to the specified
        :class:`DicomSeriesImageInfo`, if available. If no :class:`DicomSeriesRegistrationInfo` is available
        :class:`None` is returned.

        Args:
            image_info (DicomSeriesImageInfo): The :class:`DicomSeriesImageInfo` for which the
             :class:`DicomSeriesRegistrationInfo` is requested.

        Returns:
            Optional[DicomSeriesRegistrationInfo]: The :class:`DicomSeriesRegistrationInfo` which belongs to the
             specified :class:`DicomSeriesImageInfo` or :class:`None`.
        """
        if not self.reg_info:
            return None

        selected = []
        for reg_info in self.reg_info:
            reg_info.update() if not reg_info.is_updated() else None

            if reg_info.referenced_series_instance_uid_transform == image_info.series_instance_uid:
                selected.append(reg_info)

        if len(selected) > 1:
            raise ValueError(f'Multiple registration infos detected with the same referenced '
                             f'SeriesInstanceUID ({image_info.series_instance_uid})!')

        if not selected:
            return None

        return selected[0]

    @staticmethod
    def _transform_image(image: sitk.Image,
                         transform: sitk.Transform,
                         is_intensity: bool
                         ) -> sitk.Image:
        """Transform an :class:`sitk.Image` according to the provided :class:`sitk.Transform`.

        Args:
            image (sitk.Image): The image to transform.
            transform (sitk.Transform): The transform to be applied to the image.
            is_intensity (bool): If True the image will be resampled using a B-Spline interpolation function,
             otherwise a nearest neighbour interpolation function will be used.

        Returns:
            sitk.Image: The transformed image.
        """
        # select the appropriate interpolation function
        if is_intensity:
            interpolator = sitk.sitkBSpline
        else:
            interpolator = sitk.sitkNearestNeighbor

        image_np = sitk.GetArrayFromImage(image)
        default_pixel_value = np.min(image_np).astype(np.float)

        # compute the new origin
        new_origin = transform.GetInverse().TransformPoint(image.GetOrigin())

        # compute the new direction
        new_direction_0 = transform.TransformVector(image.GetDirection()[:3], image.GetOrigin())
        new_direction_1 = transform.TransformVector(image.GetDirection()[3:6], image.GetOrigin())
        new_direction_2 = transform.TransformVector(image.GetDirection()[6:], image.GetOrigin())
        new_direction = new_direction_0 + new_direction_1 + new_direction_2

        new_direction_matrix = np.array(new_direction).reshape(3, 3)
        original_direction_matrix = np.array(image.GetDirection()).reshape(3, 3)
        new_direction_corr = np.dot(np.dot(new_direction_matrix, original_direction_matrix).transpose(),
                                    original_direction_matrix).transpose()

        # resample the image
        resampled_image = sitk.Resample(image,
                                        image.GetSize(),
                                        transform=transform,
                                        interpolator=interpolator,
                                        outputOrigin=new_origin,
                                        outputSpacing=image.GetSpacing(),
                                        outputDirection=tuple(new_direction_corr.flatten()),
                                        defaultPixelValue=default_pixel_value,
                                        outputPixelType=image.GetPixelIDValue())

        return resampled_image

    def convert(self) -> Tuple[IntensityImage, ...]:
        """Convert the provided :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo` entries to one or multiple
        :class:`~pyradise.data.image.IntensityImage` instances.

        Returns:
            Tuple[IntensityImage, ...]: The converted :class:`~pyradise.data.image.IntensityImage` instances.
        """
        images = []

        for info in self.image_info:

            # read the image
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(info.path)
            image = reader.Execute()

            # get the registration info if available
            reg_info = self._get_registration_info(info)

            if not info.is_updated():
                info.update()

            # if no registration info is available, the image is added as is
            if reg_info is None:
                images.append(IntensityImage(image, info.modality))

            # else the image is transformed
            else:
                ref_series_instance_uid = reg_info.referenced_series_instance_uid_identity
                ref_image_info = self._get_image_info_by_series_instance_uid(ref_series_instance_uid)

                if ref_image_info is None:
                    raise ValueError(f'The reference image with SeriesInstanceUID {ref_series_instance_uid} '
                                     f'is missing for the registration!')

                image = self._transform_image(image, reg_info.transform, is_intensity=True)

                images.append(IntensityImage(image, info.modality))

        return tuple(images)


class DicomRTSSSeriesConverter(Converter):
    """A :class:`Converter` class for converting a DICOM-RTSS (i.e.
    :class:`~pyradise.fileio.series_info.DicomSeriesRTSSInfo`) to one or multiple
    :class:`~pyradise.data.image.SegmentationImage` instances.

    Notes:
        The user may provide all available :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo` and
        :class:`~pyradise.fileio.series_info.DicomSeriesRegistrationInfo` entries to the corresponding ``image_infos``
        and ``registration_infos``, respectively. In this case the :class:`DicomRTSSSeriesConverter` will sort out
        unused entries.

    Args:
        rtss_infos (Union[DicomSeriesRTSSInfo, Tuple[DicomSeriesRTSSInfo, ...]]): The
         :class:`~pyradise.fileio.series_info.DicomSeriesRTSSInfo` instance holding the information to be converted.
        image_infos (Tuple[DicomSeriesImageInfo, ...]): The :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo`
         entries which are referenced in the :class:`~pyradise.fileio.series_info.DicomSeriesRTSSInfo` instance.
        registration_infos (Optional[Tuple[DicomSeriesRegistrationInfo, ...]]): The
         :class:`~pyradise.fileio.series_info.DicomSeriesRegistrationInfo` entries referencing the DICOM image or
         DICOM-RTSS.
    """

    def __init__(self,
                 rtss_infos: Union[DicomSeriesRTSSInfo, Tuple[DicomSeriesRTSSInfo, ...]],
                 image_infos: Tuple[DicomSeriesImageInfo, ...],
                 registration_infos: Optional[Tuple[DicomSeriesRegistrationInfo, ...]]
                 ) -> None:
        super().__init__()

        if isinstance(rtss_infos, DicomSeriesRTSSInfo):
            self.rtss_infos = (rtss_infos,)
        else:
            self.rtss_infos = rtss_infos

        self.image_infos = image_infos
        self.reg_infos = registration_infos

    def _get_referenced_image_info(self, rtss_info: DicomSeriesRTSSInfo) -> Optional[DicomSeriesImageInfo]:
        """Get the :class:`DicomSeriesImageInfo` which is referenced by the provided :class:`DicomSeriesRTSSInfo`.

        Args:
            rtss_info (DicomSeriesRTSSInfo): The :class:`DicomSeriesRTSSInfo` containing the reference.

        Returns:
            Optional[DicomSeriesImageInfo]: The referenced :class:`DicomSeriesImageInfo` or :class:`None` if no
             reference is available.

        """
        if not self.image_infos:
            return None

        selected = [info for info in self.image_infos if info.series_instance_uid == rtss_info.referenced_instance_uid]

        if len(selected) > 1:
            raise ValueError(f'Multiple image infos detected with the same referenced '
                             f'SeriesInstanceUID ({rtss_info.referenced_instance_uid})!')

        if not selected:
            raise ValueError(f'The reference image with the SeriesInstanceUID '
                             f'{rtss_info.referenced_instance_uid} for the RTSS conversion is missing!')

        return selected[0]

    def _get_referenced_registration_info(self,
                                          rtss_info: DicomSeriesRTSSInfo,
                                          ) -> Optional[DicomSeriesRegistrationInfo]:
        """Get the :class:`DicomSeriesRegistrationInfo` which is referenced in the :class:`DicomSeriesRTSSInfo`.

        Args:
            rtss_info (DicomSeriesRTSSInfo): The :class:`DicomSeriesRTSSInfo` for which the referenced
             :class:`DicomSeriesRegistrationInfo` should be retrieved.

        Returns:
            Optional[DicomSeriesRegistrationInfo]: The :class:`DicomSeriesRegistrationInfo` instance referenced in the
             RTSS or None.
        """

        if not self.reg_infos:
            return None

        selected = []

        for registration_info in self.reg_infos:
            if registration_info.referenced_series_instance_uid_transform == rtss_info.referenced_instance_uid:
                selected.append(registration_info)

        if not selected:
            return None

        if len(selected) > 1:
            raise NotImplementedError('The number of referenced registrations is larger than one! '
                                      'The sequential application of registrations is not supported yet!')

        return selected[0]

    def convert(self) -> Tuple[SegmentationImage, ...]:
        """Convert the :class:`~pyradise.fileio.series_info.DicomSeriesRTSSInfo` instances into one or multiple
        :class:`~pyradise.data.image.SegmentationImage` instances.

        Returns:
            Tuple[SegmentationImage, ...]: The converted :class:`~pyradise.data.image.SegmentationImage` instances.
        """
        images = []

        for rtss_info in self.rtss_infos:

            ref_image_info = self._get_referenced_image_info(rtss_info)
            ref_reg_info = self._get_referenced_registration_info(rtss_info)

            if ref_reg_info:
                ref_reg_info.update() if not ref_reg_info.is_updated() else None
                reg_dataset = ref_reg_info.dataset
            else:
                reg_dataset = None

            structures = RTSSToSegmentConverter(rtss_info.dataset, ref_image_info.path, reg_dataset).convert()

            for roi_name, segmentation_image in structures.items():
                images.append(SegmentationImage(segmentation_image, Organ(roi_name), rtss_info.rater))

        return tuple(images)


class SubjectToRTSSConverter(Converter):
    """A :class:`Converter` class for converting the :class:`~pyradise.data.image.SegmentationImage` instances of a
    :class:`~pyradise.data.subject.Subject` instance to a :class:`~pydicom.dataset.Dataset` instance.

    Notes:
        This class is typically used at the end of a processing pipeline to output a DICOM-RTSS file containing the
        segmentation results of the pipeline.

    Args:
        subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be converted to a DICOM-RTSS
         :class:`~pydicom.dataset.Dataset` instance.
        infos (Tuple[DicomSeriesInfo]): The :class:`~pyradise.fileio.series_info.DicomSeriesInfo` entries provided for
         the conversion (only :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo` will be considered).
        reference_modality (Modality): The reference :class:`~pyradise.data.modality.Modality` of the images to be
         used for the conversion to DICOM-RTSS.
    """

    def __init__(self,
                 subject: Subject,
                 infos: Tuple[DicomSeriesInfo],
                 reference_modality: Modality
                 ) -> None:
        super().__init__()

        assert subject.segmentation_images, 'The subject must contain segmentation images!'
        self.subject = subject

        assert infos, 'There must be infos provided for the conversion!'

        image_infos = [entry for entry in infos if isinstance(entry, DicomSeriesImageInfo) and
                       entry.modality == reference_modality]

        assert image_infos, 'There must be image infos in the provided infos!'

        assert len(image_infos) == 1, 'There are multiple image infos fitting the reference modality!'
        self.image_info = image_infos[0]
        self.ref_modality = reference_modality

    def convert(self) -> Dataset:
        """Convert a :class:`~pyradise.data.subject.Subject` instance to a DICOM-RTSS :class:`~pydicom.dataset.Dataset`
        instance.

        Returns:
            Dataset: The DICOM-RTSS :class:`~pydicom.dataset.Dataset` instance generated from the provided
            :class:`~pyradise.data.subject.Subject` instance.
        """
        # get the image data and the label names
        sitk_images = []
        label_names = []
        for image in self.subject.segmentation_images:
            sitk_images.append(image.get_image(as_sitk=True))
            label_names.append(image.get_organ(as_str=True))

        # load the image datasets
        image_datasets = load_datasets(self.image_info.path)

        # convert the images to a rtss
        rtss = SegmentToRTSSConverter(tuple(sitk_images), image_datasets, tuple(label_names), None).convert()

        return rtss
