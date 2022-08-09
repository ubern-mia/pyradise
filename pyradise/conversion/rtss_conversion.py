from typing import (
    Tuple,
    Any,
    Optional,
    Union,
    List,
    Dict)
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass
from enum import IntEnum
import warnings

import SimpleITK as sitk
import numpy as np
import cv2 as cv
from pydicom.uid import (
    generate_uid,
    ImplicitVRLittleEndian,
    PYDICOM_IMPLEMENTATION_UID)
from pydicom.dataset import (
    Dataset,
    FileDataset,
    FileMetaDataset)
from pydicom.sequence import Sequence
from scipy.interpolate import (
    splprep,
    splev)

from .series_information import (
    DicomSeriesRegistrationInfo,
    RegistrationInfo)
from .base_conversion import Converter
from .utils import (
    load_dataset,
    load_datasets,
    chunkify
)

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


class RTSSToImageConverter(Converter):
    """A DICOM RT Structure Set converter which converts an RTSS Dataset into one or multiple SimpleITK images."""

    def __init__(self,
                 rtss_dataset: Union[str, Dataset],
                 image_datasets: Union[Tuple[str], Tuple[Dataset]],
                 registration_dataset: Union[str, Dataset, None] = None
                 ) -> None:
        super().__init__()

        if isinstance(rtss_dataset, str):
            self.rtss_dataset = load_dataset(rtss_dataset)
        else:
            self.rtss_dataset = rtss_dataset

        if isinstance(image_datasets[0], str):
            image_datasets = load_datasets(image_datasets)
            self.rtss_image_datasets = self._clean_image_datasets_for_rtss(image_datasets, self.rtss_dataset)
        else:
            self.rtss_image_datasets = self._clean_image_datasets_for_rtss(image_datasets, self.rtss_dataset)

        if not registration_dataset:
            self.registration_dataset = None
        elif isinstance(registration_dataset, str):
            self.registration_dataset = load_dataset(registration_dataset, True)
        elif isinstance(registration_dataset, Dataset):
            self.registration_dataset = registration_dataset

        self.registration_image_datasets = self._clean_image_datasets_for_registration(image_datasets,
                                                                                       self.registration_dataset)

        # Validation of the work data
        self._validate_rtss_dataset(self.rtss_dataset)
        self._validate_rtss_image_references(self.rtss_dataset, self.rtss_image_datasets)
        self.registration_dataset = self._validate_registration_dataset(self.registration_dataset, self.rtss_dataset)

    @staticmethod
    def get_slice_direction(image_dataset: Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gets the slice direction from the Dataset.

        Args:
            image_dataset (Dataset): The Dataset from which the slice direction should be determined.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The directions in all three dimensions.
        """
        orientation = image_dataset.get('ImageOrientationPatient')

        row_direction = np.array(orientation[:3])
        column_direction = np.array(orientation[3:])
        slice_direction = np.cross(row_direction, column_direction)

        validate_directions = (np.allclose(np.dot(row_direction, column_direction), 0.0, atol=1e-3),
                               np.allclose(np.linalg.norm(slice_direction), 1.0, atol=1e-3))

        if not all(validate_directions):
            raise Exception(f'Invalid ImageOrientationPatient attribute in {image_dataset.get("PatientID")}!')

        return row_direction, column_direction, slice_direction

    @staticmethod
    def _get_slice_position(image_dataset: Dataset) -> np.ndarray:
        """Gets the slice position from a Dataset.

        Args:
            image_dataset (Dataset): The Dataset from which the slice position should be determined.

        Returns:
            np.ndarray: The position of the slice in space.
        """
        orientation = image_dataset.get('ImagePositionPatient')

        _, _, slice_direction = RTSSToImageConverter.get_slice_direction(image_dataset)

        return np.dot(slice_direction, orientation)

    @staticmethod
    def _clean_image_datasets_for_rtss(image_datasets: Tuple[Dataset],
                                       rtss_dataset: Dataset
                                       ) -> Tuple[Dataset]:
        """Cleans the image Datasets based on the referenced SeriesInstanceUID.

        Args:
            image_datasets (Tuple[Dataset]): The image Datasets which should be analyzed.
            rtss_dataset (Dataset): The RTSS dataset which contains the image references.

        Returns:
            Tuple[Dataset]: The image Datasets which are referenced by the RTSS Dataset.
        """
        referenced_series_instance_uids = []

        for referenced_frame_of_ref in rtss_dataset.get('ReferencedFrameOfReferenceSequence', []):
            for rt_referenced_study in referenced_frame_of_ref.get('RTReferencedStudySequence', []):
                for rt_referenced_series in rt_referenced_study.get('RTReferencedSeriesSequence', []):
                    series_instance_uid = rt_referenced_series.get('SeriesInstanceUID', None)
                    if series_instance_uid:
                        referenced_series_instance_uids.append(str(series_instance_uid))

        if len(referenced_series_instance_uids) > 1:
            raise Exception(f'Multiple ({len(referenced_series_instance_uids)}) referenced SeriesInstanceUIDs '
                            f'have been retrieved from the RTSS but only one is allowed!')

        if not referenced_series_instance_uids:
            raise Exception('No valid referenced SeriesInstanceUID has been retrieved!')

        selected = [image_dataset for image_dataset in image_datasets
                    if image_dataset.get('SeriesInstanceUID', None) == referenced_series_instance_uids[0]]

        selected.sort(key=RTSSToImageConverter._get_slice_position, reverse=False)

        return tuple(selected)

    @staticmethod
    def _clean_image_datasets_for_registration(image_datasets: Tuple[Dataset, ...],
                                               registration_dataset: Optional[Dataset]
                                               ) -> Optional[Tuple[Dataset, ...]]:
        """Cleans the image datasets for registration. This function outputs the datasets which are referenced within
         the registration dataset.

        Args:
            image_datasets (Tuple[Dataset, ...]): The image datasets to analyze.
            registration_dataset (Optional[Dataset]): The registration dataset containing the references.

        Returns:
            Optional[Tuple[Dataset, ...]]: The datasets being referenced within the registration.
        """
        if not registration_dataset:
            return None

        selected = []

        referenced_image_series = DicomSeriesRegistrationInfo.get_referenced_series_info(registration_dataset)
        referenced_series_uids = [reference_info.series_instance_uid for reference_info in referenced_image_series]
        referenced_study_uids = [reference_info.study_instance_uid for reference_info in referenced_image_series]

        for image_dataset in image_datasets:
            criteria = (str(image_dataset.get('SeriesInstanceUID')) in referenced_series_uids,
                        str(image_dataset.get('StudyInstanceUID')) in referenced_study_uids)
            if all(criteria):
                selected.append(image_dataset)

        return tuple(selected)

    @staticmethod
    def _validate_rtss_dataset(rtss_dataset: Dataset) -> None:
        """Validates if the RTSS Dataset is containing the minimal data for conversion.

        Args:
            rtss_dataset (Dataset): The RTSS Dataset to validate.

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
    def _validate_registration_dataset(registration_dataset: Dataset,
                                       rtss_dataset: Dataset
                                       ) -> Optional[Dataset]:
        """Validates if the registration dataset is containing the minimal data for conversion and returns the
         registration dataset if it is valid (otherwise None).

        Args:
            registration_dataset (Dataset): The registration dataset to validate.
            rtss_dataset (Dataset): The RTSS dataset containing the references.

        Returns:
            Optional[Dataset]: The valid registration dataset or None.
        """

        if not registration_dataset:
            return None

        # search for references in the registration dataset
        registration_referenced_instance_uids = []
        for referenced_studies in registration_dataset.get('StudiesContainingOtherReferencedInstancesSequence', []):
            for referenced_series in referenced_studies.get('ReferencedSeriesSequence', []):
                referenced_series_instance_uid = referenced_series.get('SeriesInstanceUID', None)
                if referenced_series_instance_uid:
                    registration_referenced_instance_uids.append(referenced_series_instance_uid)

        for referenced_series in registration_dataset.get('ReferencedSeriesSequence', []):
            referenced_series_uid = referenced_series.get('SeriesInstanceUID', None)
            if referenced_series_uid:
                registration_referenced_instance_uids.append(referenced_series_uid)

        # search for references in the rtss dataset
        rtss_referenced_instance_uids = []
        for referenced_frame_of_ref in rtss_dataset.get('ReferencedFrameOfReferenceSequence', []):
            for rt_referenced_study in referenced_frame_of_ref.get('RTReferencedStudySequence', []):
                for rt_referenced_series in rt_referenced_study.get('RTReferencedSeriesSequence', []):
                    referenced_series_uid = rt_referenced_series.get('SeriesInstanceUID', None)
                    if referenced_series_uid:
                        rtss_referenced_instance_uids.append(referenced_series_uid)

        # check the criteria
        criteria = (registration_dataset.get('SOPClassUID', '') == '1.2.840.10008.5.1.4.1.1.66.1',
                    hasattr(registration_dataset, 'RegistrationSequence'),
                    hasattr(registration_dataset, 'ReferencedSeriesSequence'),
                    all([rt_reference in registration_referenced_instance_uids for
                         rt_reference in rtss_referenced_instance_uids]),
                    len(registration_referenced_instance_uids) != 0)

        if not all(criteria):
            print(f'The checked registration from subject {rtss_dataset.get("PatientID", "n/a")} is invalid!')
            return None

        return registration_dataset

    @staticmethod
    def _validate_rtss_image_references(rtss_dataset: Dataset,
                                        image_datasets: Tuple[Dataset]
                                        ) -> None:
        """Validates if all RTSS Dataset's ReferencedSOPInstanceUIDs are contained in the image Datasets.

        Args:
            rtss_dataset (Dataset): The RTSS Dataset to validate.
            image_datasets (Tuple[Dataset]): The image Datasets to be used for comparison.

        Returns:
            None
        """
        for referenced_frame_of_ref in rtss_dataset.get('ReferencedFrameOfReferenceSequence', []):
            for rt_referenced_study in referenced_frame_of_ref.get('RTReferencedStudySequence', []):
                for rt_referenced_series in rt_referenced_study.get('RTReferencedSeriesSequence', []):
                    for contour_image in rt_referenced_series.get('ContourImageSequence', []):
                        RTSSToImageConverter._validate_referenced_contour_image(contour_image,
                                                                                image_datasets)

    @staticmethod
    def _validate_referenced_contour_image(contour_image_sq_entry: Dataset,
                                           image_datasets: Tuple[Dataset]
                                           ) -> None:
        """Validates if a certain RTSS ContourSequence entry's reference is provided in the image Datasets.

        Args:
            contour_image_sq_entry (Dataset): The ContourSequence entry to be checked.
            image_datasets (Tuple[Dataset]): The candidate image Datasets.

        Returns:
            None
        """
        for dataset in image_datasets:
            if contour_image_sq_entry.get('ReferencedSOPInstanceUID', '') == dataset.file_meta. \
                    get('MediaStorageSOPInstanceUID', None):
                return

        raise Exception('Loaded RTSS reference image(s) detected that are not contained in the provided image(s)! \n'
                        'Missing image has SOPInstanceUID: '
                        f'{contour_image_sq_entry.get("ReferencedSOPInstanceUID", "")}.')

    @staticmethod
    def _get_contour_sequence_by_roi_number(rtss_dataset: Dataset,
                                            roi_number: int
                                            ) -> Optional[Sequence]:
        """Gets the ContourSequence by the ROINumber.

        Args:
            rtss_dataset (Dataset): The RT Structure Set dataset to retrieve the ContourSequence from.
            roi_number (int): The ROINumber for which the ContourSequence should be returned.

        Returns:
            Optional[Sequence]: The ContourSequence with the corresponding ROINumber.
        """
        if rtss_dataset.get('ROIContourSequence', None) is None:
            return None

        for roi_contour in rtss_dataset.get('ROIContourSequence', []):

            # Ensure same type
            if str(roi_contour.get('ReferencedROINumber', None)) == str(roi_number):
                if roi_contour.get('ContourSequence', None) is None:
                    return None

                return roi_contour.get('ContourSequence')

        raise Exception(f"Referenced ROI number '{roi_number}' not found")

    @staticmethod
    def _create_empty_series_mask(image_datasets: Tuple[Dataset, ...]) -> np.ndarray:
        """Creates an empty numpy array with the shape according to the reference image.

        Returns:
            np.ndarray: The empty numpy array with appropriate shape.
        """
        rows = int(image_datasets[0].get('Rows'))
        columns = int(image_datasets[0].get('Columns'))
        num_slices = len(image_datasets)

        return np.zeros((columns, rows, num_slices)).astype(np.bool)

    @staticmethod
    def _create_empty_slice_mask(image_dataset: Dataset) -> np.ndarray:
        """Creates an empty numpy array representing one slice of the output mask.

        Args:
            image_dataset (Dataset): The Dataset providing the spatial information for the empty mask.

        Returns:
            np.ndarray: The empty slice numpy array.
        """
        columns = int(image_dataset.get('Columns'))
        rows = int(image_dataset.get('Rows'))
        return np.zeros((columns, rows)).astype(bool)

    @staticmethod
    def _apply_transformation_to_3d_points(points: np.ndarray,
                                           transformation_matrix: np.ndarray
                                           ) -> np.ndarray:
        """Applies the provided transformation to multiple points in the 3D-space.

        Args:
            points (np.ndarray): The points to transform.
            transformation_matrix (np.ndarray): The transformation matrix.

        Notes:
            1. Augment each point with a '1' as the fourth coordinate to allow translation
            2. Multiply by a 4x4 transformation matrix
            3. Throw away added '1's

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
        """Gets the mask with the filled contour.

        Args:
            series_slice (Dataset): The corresponding slice Dataset.
            contour_coords (List[float]): The contour points.
            transformation_matrix (np.ndarray): The transformation matrix.

        Returns:
            np.ndarray: The mask with the filled contour.
        """
        reshaped_contour_data = np.reshape(contour_coords, [len(contour_coords) // 3, 3])
        translated_contour_data = RTSSToImageConverter._apply_transformation_to_3d_points(reshaped_contour_data,
                                                                                          transformation_matrix)
        polygon = [np.around([translated_contour_data[:, :2]]).astype(np.int32)]

        # Create mask for the region. Fill with 1 for ROI
        fill_mask = RTSSToImageConverter._create_empty_slice_mask(series_slice).astype(np.uint8)
        cv.fillPoly(img=fill_mask, pts=polygon, color=1)
        return fill_mask

    @staticmethod
    def get_spacing_between_slices(image_datasets: Tuple[Dataset, ...]) -> float:
        """Gets the spacing between the slices based on the first and last slice positions.

        Args:
            image_datasets (Tuple[Dataset, ...]): The datasets to get the spacing from

        Returns:
            float: The spacing between the slices.
        """
        if len(image_datasets) > 1:
            first = RTSSToImageConverter._get_slice_position(image_datasets[0])
            last = RTSSToImageConverter._get_slice_position(image_datasets[-1])
            return (last - first) / (len(image_datasets) - 1)

        return 1.0

    @staticmethod
    def _get_patient_to_pixel_transformation_matrix(image_datasets: Tuple[Dataset, ...]) -> np.ndarray:
        """Gets the patient to pixel transformation matrix from the first image Dataset.

        Args:
            image_datasets (Tuple[Dataset, ...]): The datasets to retrieve the transformation matrix.

        Returns:
            np.ndarray: The transformation matrix.
        """
        offset = np.array(image_datasets[0].get('ImagePositionPatient'))
        row_spacing, column_spacing = image_datasets[0].get('PixelSpacing')
        slice_spacing = RTSSToImageConverter.get_spacing_between_slices(image_datasets)
        row_direction, column_direction, slice_direction = RTSSToImageConverter.get_slice_direction(image_datasets[0])

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
        """Gets the contour data from the corresponding ContourSequence.

        Args:
            image_dataset (Dataset): The referenced image Dataset.
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
        """Gets the slice mask from the ContourData.

        Args:
            image_dataset (Dataset): The referenced image Dataset.
            contour_data (Tuple[Any, ...]): The contour data.
            transformation_matrix (np.ndarray): The transformation matrix.

        Returns:
            np.ndarray: The discrete slice mask.
        """
        slice_mask = RTSSToImageConverter._create_empty_slice_mask(image_dataset)

        for contour_coords in contour_data:
            fill_mask = RTSSToImageConverter._get_contour_fill_mask(image_dataset, contour_coords,
                                                                    transformation_matrix)
            slice_mask[fill_mask == 1] = np.invert(slice_mask[fill_mask == 1])

        return slice_mask

    @staticmethod
    def _create_mask_from_contour_sequence(image_datasets: Tuple[Dataset, ...],
                                           contour_sequence: Sequence
                                           ) -> np.ndarray:
        """Creates the whole 3D mask from the ContourSequence.

        Args:
            image_datasets (Tuple[Dataset, ...]): The image datasets to be used for mask creation.
            contour_sequence (Sequence): The ContourSequence to be discretized.

        Returns:
            np.ndarray: The discrete segmentation mask.
        """
        mask = RTSSToImageConverter._create_empty_series_mask(image_datasets)
        transformation_matrix = RTSSToImageConverter._get_patient_to_pixel_transformation_matrix(image_datasets)

        for i, image_dataset in enumerate(image_datasets):
            slice_contour_data = RTSSToImageConverter._get_slice_contour_data(image_dataset, contour_sequence)
            if slice_contour_data:
                mask[:, :, i] = RTSSToImageConverter._get_slice_mask_from_slice_contour_data(image_dataset,
                                                                                             slice_contour_data,
                                                                                             transformation_matrix)

        return mask

    @staticmethod
    def _create_image_from_mask(image_datasets: Tuple[Dataset, ...],
                                mask: np.ndarray
                                ) -> sitk.Image:
        """Creates a sitk.Image from the numpy segmentation mask with appropriate orientation.

        Args:
            image_datasets (Tuple[Dataset, ...]): The image datasets used to create the image.
            mask (np.ndarray): The mask to be converted into a sitk.Image.

        Returns:
            sitk.Image: The sitk.Image generated.
        """
        mask = np.swapaxes(mask, 0, -1)

        image = sitk.GetImageFromArray(mask.astype(np.uint8))

        image.SetOrigin(image_datasets[0].get('ImagePositionPatient'))

        row_spacing, column_spacing = image_datasets[0].get('PixelSpacing')
        slice_spacing = RTSSToImageConverter.get_spacing_between_slices(image_datasets)
        image.SetSpacing((float(row_spacing), float(column_spacing), float(slice_spacing)))

        slice_direction = np.stack(RTSSToImageConverter.get_slice_direction(image_datasets[0]), axis=0).T. \
            flatten().tolist()
        image.SetDirection(slice_direction)

        return image

    @staticmethod
    def _get_transform_from_registration_info(registration_infos: Tuple[RegistrationInfo, ...]) -> sitk.Transform:
        """Gets the transformation from the registration infos.

        Args:
            registration_infos (Tuple[RegistrationInfo]): Multiple registration infos which hold transformations and
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
    def _apply_transform_to_images(image_datasets: Tuple[Dataset, ...],
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
    def _apply_transform_to_rtss(rtss_dataset: Dataset,
                                 transform: sitk.Transform
                                 ) -> Dataset:
        """Apply the transformation to the RT Structure Set.

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
        """Converts a RTSS into binary sitk.Images.

        Returns:
            Dict[str, sitk.Image]: The RoiNames and the corresponding binary segmented sitk.Images.
        """
        converted_images = {}

        # apply the registration if available
        if self.registration_dataset:
            registration_infos = DicomSeriesRegistrationInfo.get_registration_infos(self.registration_dataset,
                                                                                    self.registration_image_datasets)
            transform = self._get_transform_from_registration_info(registration_infos)
            image_datasets = self._apply_transform_to_images(self.rtss_image_datasets, transform)
            rtss_dataset = self._apply_transform_to_rtss(self.rtss_dataset, transform)

        else:
            image_datasets = self.rtss_image_datasets
            rtss_dataset = self.rtss_dataset

        # convert the contours to images
        for ss_roi_entry in rtss_dataset.get('StructureSetROISequence', []):
            roi_number = int(ss_roi_entry.get('ROINumber'))
            roi_name = str(ss_roi_entry.get('ROIName', ''))

            contour_sequence = self._get_contour_sequence_by_roi_number(rtss_dataset, roi_number)

            if contour_sequence is None:
                continue

            mask = self._create_mask_from_contour_sequence(image_datasets, contour_sequence)
            image = self._create_image_from_mask(image_datasets, mask)
            converted_images.update({roi_name: image})

        return converted_images


@dataclass
class ROIData:
    """Data class to easily pass ROI data to helper methods."""
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
            self.name = f"ROI-{self.number}"

    def _validate_color(self):
        if self.color is None:
            return

        # Validating list eg: [0, 0, 0]
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
    """
    Enum class for what the positions in the OpenCV hierarchy array mean
    """

    NEXT_NODE = 0
    PREVIOUS_NODE = 1
    FIRST_CHILD = 2
    PARENT_NODE = 3


class ImageToRTSSConverter(Converter):
    """A class for converting one or multiple label images into a DICOM RT Structure Set file."""

    def __init__(self,
                 label_images: Union[Tuple[str, ...], Tuple[sitk.Image, ...]],
                 referenced_image_datasets: Union[Tuple[str, ...], Tuple[Dataset, ...]],
                 label_names: Union[Tuple[str, ...], Dict[int, str], None],
                 colors: Optional[Tuple[Tuple[int, int, int], ...]],
                 ) -> None:
        # pylint: disable=consider-using-generator
        super().__init__()

        if isinstance(label_images[0], str):
            self.label_images = tuple([sitk.ReadImage(path, sitk.sitkUInt8) for path in label_images])
        else:
            self.label_images = label_images

        if isinstance(referenced_image_datasets[0], str):
            self.image_datasets: Tuple[Dataset, ...] = load_datasets(referenced_image_datasets)
        else:
            self.image_datasets: Tuple[Dataset, ...] = referenced_image_datasets

        if isinstance(label_names, dict):
            sorted_keys = sorted(label_names.keys())
            self.label_names = tuple([str(label_names.get(key)) for key in sorted_keys])
        elif not label_names:
            self.label_names = tuple([f'Structure_{i}' for i in range(len(self.label_images))])
        else:
            self.label_names = label_names

        if not colors:
            self.colors = COLOR_PALETTE
        else:
            self.colors = colors

        assert len(self.label_names) >= len(self.label_images), 'The number of label names must be equal or larger ' \
                                                                'than the number of label images!'
        assert len(self.colors) >= len(self.label_images), 'The number of colors must be equal or larger ' \
                                                           'than the number of label images!'

    @staticmethod
    def _generate_file_meta() -> FileMetaDataset:
        meta = FileMetaDataset()
        meta.FileMetaInformationGroupLength = 202
        meta.FileMetaInformationVersion = b'\x00\x01'
        meta.TransferSyntaxUID = ImplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
        return meta

    @staticmethod
    def _add_required_elements_to_dataset(dataset: FileDataset) -> None:
        now = datetime.now()

        dataset.SpecificCharacterSet = 'ISO_IR 100'
        dataset.InstanceCreationDate = now.strftime('%Y%m%d')
        dataset.InstanceCreationTime = now.strftime('%H%M%S.%f')
        dataset.StructureSetLabel = 'Autogenerated'
        dataset.StructureSetDate = now.strftime('%Y%m%d')
        dataset.StructureSetTime = now.strftime('%H%M%S.%f')
        dataset.Modality = 'RTSTRUCT'
        dataset.Manufacturer = 'University of Bern, Switzerland'
        dataset.ManufacturerModelName = 'ISAS'
        dataset.InstitutionName = 'University of Bern, Switzerland'
        dataset.OperatorsName = 'InnoSuisse Autosegmentation Algorithm'

        dataset.is_little_endian = True
        dataset.is_implicit_VR = True

        # set values already defined in the file meta
        dataset.SOPClassUID = dataset.file_meta.MediaStorageSOPClassUID
        dataset.SOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID

        dataset.ApprovalStatus = 'UNAPPROVED'

    @staticmethod
    def _add_sequence_lists_to_dataset(dataset: FileDataset) -> None:
        dataset.StructureSetROISequence = Sequence()
        dataset.ROIContourSequence = Sequence()
        dataset.RTROIObservationsSequence = Sequence()

    @staticmethod
    def _add_study_and_series_information(dataset: FileDataset,
                                          image_datasets: Tuple[Dataset, ...]
                                          ) -> None:
        reference_dataset = image_datasets[0]
        dataset.StudyDate = reference_dataset.StudyDate
        dataset.SeriesDate = getattr(reference_dataset, 'SeriesDate', '')
        dataset.StudyTime = reference_dataset.StudyTime
        dataset.SeriesTime = getattr(reference_dataset, 'SeriesTime', '')
        dataset.StudyDescription = getattr(reference_dataset, 'StudyDescription', '')
        dataset.SeriesDescription = getattr(reference_dataset, 'SeriesDescription', '')
        dataset.StudyInstanceUID = reference_dataset.StudyInstanceUID
        dataset.SeriesInstanceUID = generate_uid()
        dataset.StudyID = reference_dataset.StudyID
        dataset.SeriesNumber = "99"
        dataset.ReferringPhysicianName = "UNKNOWN"
        dataset.AccessionNumber = "0"

    @staticmethod
    def _add_patient_information(dataset: FileDataset,
                                 image_datasets: Tuple[Dataset, ...]
                                 ) -> None:
        reference_dataset = image_datasets[0]
        dataset.PatientName = getattr(reference_dataset, 'PatientName', '')
        dataset.PatientID = getattr(reference_dataset, 'PatientID', '')
        dataset.PatientBirthDate = getattr(reference_dataset, 'PatientBirthDate', '')
        dataset.PatientSex = getattr(reference_dataset, 'PatientSex', '')
        dataset.PatientAge = getattr(reference_dataset, 'PatientAge', '')
        dataset.PatientSize = getattr(reference_dataset, 'PatientSize', '')
        dataset.PatientWeight = getattr(reference_dataset, 'PatientWeight', '')

    @staticmethod
    def _add_referenced_frame_of_ref_sequence(dataset: FileDataset,
                                              image_datasets: Tuple[Dataset, ...]
                                              ) -> None:
        referenced_frame_of_ref = Dataset()
        referenced_frame_of_ref.FrameOfReferenceUID = image_datasets[0].FrameOfReferenceUID
        referenced_frame_of_ref.RTReferencedStudySequence = ImageToRTSSConverter. \
            _create_frame_of_ref_study_sequence(image_datasets)

        # Add to sequence
        dataset.ReferencedFrameOfReferenceSequence = Sequence()
        dataset.ReferencedFrameOfReferenceSequence.append(referenced_frame_of_ref)

    @staticmethod
    def _create_frame_of_ref_study_sequence(image_datasets: Tuple[Dataset, ...]) -> Sequence:
        reference_ds = image_datasets[0]  # All elements in series should have the same data
        rt_referenced_series = Dataset()
        rt_referenced_series.SeriesInstanceUID = reference_ds.SeriesInstanceUID
        rt_referenced_series.ContourImageSequence = ImageToRTSSConverter._create_contour_image_sequence(image_datasets)

        rt_referenced_series_sequence = Sequence()
        rt_referenced_series_sequence.append(rt_referenced_series)

        rt_referenced_study = Dataset()
        rt_referenced_study.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.1'
        rt_referenced_study.ReferencedSOPInstanceUID = reference_ds.StudyInstanceUID
        rt_referenced_study.RTReferencedSeriesSequence = rt_referenced_series_sequence

        rt_referenced_study_sequence = Sequence()
        rt_referenced_study_sequence.append(rt_referenced_study)
        return rt_referenced_study_sequence

    @staticmethod
    def _create_contour_image_sequence(image_datasets: Tuple[Dataset, ...]) -> Sequence:
        contour_image_sequence = Sequence()

        # Add each referenced image
        for image_dataset in image_datasets:
            contour_image = Dataset()
            contour_image.ReferencedSOPClassUID = image_dataset.file_meta.MediaStorageSOPClassUID
            contour_image.ReferencedSOPInstanceUID = image_dataset.file_meta.MediaStorageSOPInstanceUID
            contour_image_sequence.append(contour_image)
        return contour_image_sequence

    @staticmethod
    def generate_basic_rtss(reference_image_datasets: Tuple[Dataset, ...],
                            file_name: str = 'rt_struct'
                            ) -> FileDataset:
        """Generate the basic RT Structure Set.

        Args:
            reference_image_datasets (Tuple[Dataset, ...]): The referenced image datasets.
            file_name (str): The file name.

        Returns:
            FileDataset: The basic RT Structure Set.
        """
        file_meta = ImageToRTSSConverter._generate_file_meta()
        rtss = FileDataset(file_name, {}, file_meta=file_meta, preamble=b"\0" * 128)

        ImageToRTSSConverter._add_required_elements_to_dataset(rtss)

        ImageToRTSSConverter._add_study_and_series_information(rtss, reference_image_datasets)
        ImageToRTSSConverter._add_patient_information(rtss, reference_image_datasets)
        ImageToRTSSConverter._add_referenced_frame_of_ref_sequence(rtss, reference_image_datasets)

        return rtss

    @staticmethod
    def _create_roi_contour(roi_data: ROIData, image_datasets: Tuple[Dataset, ...]) -> Dataset:
        roi_contour = Dataset()
        roi_contour.ROIDisplayColor = roi_data.color
        roi_contour.ContourSequence = ImageToRTSSConverter._create_contour_sequence(roi_data, image_datasets)
        roi_contour.ReferencedROINumber = str(roi_data.number)
        return roi_contour

    @staticmethod
    def _create_contour_sequence(roi_data: ROIData, image_datasets: Tuple[Dataset, ...]) -> Sequence:
        """
        Iterate through each slice of the mask
        For each connected segment within a slice, create a contour
        """

        contour_sequence = Sequence()

        contours_coords = ImageToRTSSConverter._get_contours_coords(roi_data, image_datasets)

        for series_slice, slice_contours in zip(image_datasets, contours_coords):
            for contour_data in slice_contours:
                if len(contour_data) <= 3:
                    continue
                contour = ImageToRTSSConverter._create_contour(series_slice, contour_data)
                contour_sequence.append(contour)

        return contour_sequence

    @staticmethod
    def _get_contours_coords(roi_data: ROIData, image_datasets: Tuple[Dataset, ...]) -> List[List[List[float]]]:
        transformation_matrix = ImageToRTSSConverter._get_pixel_to_patient_transformation_matrix(image_datasets)

        series_contours = []
        for i, _ in enumerate(image_datasets):
            mask_slice = roi_data.mask[i, :, :]

            # Do not add ROI's for blank slices
            if np.sum(mask_slice) == 0:
                series_contours.append([])
                # print("Skipping empty mask layer")
                continue

            # Create pinhole mask if specified
            if roi_data.use_pin_hole:
                mask_slice = ImageToRTSSConverter._create_pin_hole_mask(mask_slice, roi_data.approximate_contours)

            # Get contours from mask
            contours, _ = ImageToRTSSConverter._find_mask_contours(mask_slice, roi_data.approximate_contours)

            if not contours:
                raise Exception('Unable to find contour in non empty mask, please check your mask formatting!')

            # Format for DICOM
            formatted_contours = []
            for contour in contours:
                # Add z index
                contour = np.concatenate((np.array(contour), np.full((len(contour), 1), i)), axis=1)

                transformed_contour = ImageToRTSSConverter._apply_transformation_to_3d_points(contour,
                                                                                              transformation_matrix)
                dicom_formatted_contour = np.ravel(transformed_contour).tolist()
                formatted_contours.append(dicom_formatted_contour)

            series_contours.append(formatted_contours)

        return series_contours

    @staticmethod
    def _get_pixel_to_patient_transformation_matrix(series_data):
        """
        https://nipy.org/nibabel/dicom/dicom_orientation.html
        """

        first_slice = series_data[0]

        offset = np.array(first_slice.ImagePositionPatient)
        row_spacing, column_spacing = first_slice.PixelSpacing
        slice_spacing = RTSSToImageConverter.get_spacing_between_slices(series_data)
        row_direction, column_direction, slice_direction = RTSSToImageConverter.get_slice_direction(first_slice)

        # matrix adjusted for SimpleITK work data
        mat = np.identity(4, dtype=np.float32)
        mat[:3, 0] = row_direction * row_spacing
        mat[:3, 1] = column_direction * column_spacing
        mat[:3, 2] = slice_direction * slice_spacing
        mat[:3, 3] = offset

        return mat

    @staticmethod
    def _create_pin_hole_mask(mask: np.ndarray, approximate_contours: bool):
        """
        Creates masks with pinholes added to contour regions with holes.
        This is done so that a given region can be represented by a single contour.
        """

        contours, hierarchy = ImageToRTSSConverter._find_mask_contours(mask, approximate_contours)
        pin_hole_mask = mask.copy()

        # Iterate through the hierarchy, for child nodes, draw a line upwards from the first point
        for i, array in enumerate(hierarchy):
            parent_contour_index = array[Hierarchy.PARENT_NODE]
            if parent_contour_index == -1:
                continue  # Contour is not a child

            child_contour = contours[i]

            line_start = tuple(child_contour[0])

            pin_hole_mask = ImageToRTSSConverter._draw_line_upwards_from_point(pin_hole_mask, line_start, fill_value=0)
        return pin_hole_mask

    # pylint: disable=too-many-locals
    # noinspection DuplicatedCode
    @staticmethod
    def _smoothen_contours(contours: Tuple[np.ndarray]):
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

        return smoothened

    @staticmethod
    def _find_mask_contours(mask: np.ndarray, approximate_contours: bool):
        approximation_method = cv.CHAIN_APPROX_SIMPLE if approximate_contours else cv.CHAIN_APPROX_NONE
        contours, hierarchy = cv.findContours(mask.astype(np.uint8), cv.RETR_TREE, approximation_method)
        contours = ImageToRTSSConverter._smoothen_contours(contours)
        contours = list(contours)
        # Format extra array out of data
        for i, contour in enumerate(contours):
            contours[i] = [[pos[0][0], pos[0][1]] for pos in contour]
        hierarchy = hierarchy[0]  # Format extra array out of data

        return contours, hierarchy

    @staticmethod
    def _draw_line_upwards_from_point(mask: np.ndarray, start, fill_value: int) -> np.ndarray:
        line_width = 2
        end = (start[0], start[1] - 1)
        mask = mask.astype(np.uint8)  # Type that OpenCV expects
        # Draw one point at a time until we hit a point that already has the desired value
        while mask[end] != fill_value:
            cv.line(mask, start, end, fill_value, line_width)

            # Update start and end to the next positions
            start = end
            end = (start[0], start[1] - line_width)
        return mask.astype(bool)

    @staticmethod
    def _apply_transformation_to_3d_points(points: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
        """
            * Augment each point with a '1' as the fourth coordinate to allow translation
            * Multiply by a 4x4 transformation matrix
            * Throw away added '1's
        """
        vec = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
        return vec.dot(transformation_matrix.T)[:, :3]

    @staticmethod
    def _create_contour(series_slice: Dataset, contour_data: List[float]) -> Dataset:
        contour_image = Dataset()
        contour_image.ReferencedSOPClassUID = series_slice.file_meta.MediaStorageSOPClassUID
        contour_image.ReferencedSOPInstanceUID = series_slice.file_meta.MediaStorageSOPInstanceUID

        # Contour Image Sequence
        contour_image_sequence = Sequence()
        contour_image_sequence.append(contour_image)

        contour = Dataset()
        contour.ContourImageSequence = contour_image_sequence
        contour.ContourGeometricType = 'CLOSED_PLANAR'
        contour.NumberOfContourPoints = len(contour_data) / 3  # Each point has an x, y, and z value
        contour.ContourData = contour_data

        return contour

    @staticmethod
    def _create_structure_set_roi(roi_data: ROIData) -> Dataset:
        # Structure Set ROI Sequence: Structure Set ROI 1
        structure_set_roi = Dataset()
        structure_set_roi.ROINumber = roi_data.number
        structure_set_roi.ReferencedFrameOfReferenceUID = roi_data.frame_of_reference_uid
        structure_set_roi.ROIName = roi_data.name
        structure_set_roi.ROIDescription = roi_data.description
        structure_set_roi.ROIGenerationAlgorithm = roi_data.roi_generation_algorithm
        return structure_set_roi

    @staticmethod
    def _create_rt_roi_observation(roi_data: ROIData) -> Dataset:
        rt_roi_observation = Dataset()
        rt_roi_observation.ObservationNumber = roi_data.number
        rt_roi_observation.ReferencedROINumber = roi_data.number
        rt_roi_observation.ROIObservationDescription = 'Type:Soft,Range:*/*,Fill:0,Opacity:0.0,Thickness:1,' \
                                                       'LineThickness:2,read-only:false'
        rt_roi_observation.private_creators = 'University of Bern, Switzerland'
        rt_roi_observation.RTROIInterpretedType = ''
        rt_roi_observation.ROIInterpreter = ''
        return rt_roi_observation

    def convert(self) -> FileDataset:
        """Convert the provided subject's segmentations to an RT Structure Set.

        Returns:
            FileDataset: The RT Structure Set.
        """
        rtss = ImageToRTSSConverter.generate_basic_rtss(self.image_datasets)
        rtss.ROIContourSequence = Sequence()
        rtss.StructureSetROISequence = Sequence()
        rtss.RTROIObservationsSequence = Sequence()

        frame_of_reference_uid = self.image_datasets[0].get('FrameOfReferenceUID')

        for idx, (label_image, label_name, color) in enumerate(zip(self.label_images, self.label_names, self.colors)):
            roi_idx = idx + 1

            mask = sitk.GetArrayFromImage(label_image)
            roi_data = ROIData(mask, list(color), roi_idx, label_name, frame_of_reference_uid)

            rtss.ROIContourSequence.append(self._create_roi_contour(roi_data, self.image_datasets))
            rtss.StructureSetROISequence.append(self._create_structure_set_roi(roi_data))
            rtss.RTROIObservationsSequence.append(self._create_rt_roi_observation(roi_data))

        return rtss
