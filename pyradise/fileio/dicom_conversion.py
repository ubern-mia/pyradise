import itertools
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2 as cv
import itk
import numpy as np
import scipy
import SimpleITK as sitk
import vtkmodules.vtkCommonCore as vtk_ccore
import vtkmodules.vtkCommonDataModel as vtk_dm
import vtkmodules.vtkFiltersCore as vtk_fcore
import vtkmodules.vtkFiltersHybrid as vtk_fhybrid
import vtkmodules.vtkFiltersModeling as vtk_fmodel
import vtkmodules.vtkImagingCore as vtk_icore
import vtkmodules.vtkImagingGeneral as vtk_igen
from pydicom import Dataset, FileDataset, Sequence
from pydicom.dataset import FileMetaDataset
from pydicom.tag import Tag
from pydicom.uid import (PYDICOM_IMPLEMENTATION_UID, ImplicitVRLittleEndian,
                         generate_uid)

from pyradise.data import (IntensityImage, Modality, Organ, SegmentationImage,
                           Subject, str_to_modality)
from pyradise.utils import (chunkify, convert_to_itk_image,
                            get_slice_direction, get_slice_position,
                            get_spacing_between_slices, load_dataset,
                            load_dataset_tag, load_datasets)

from .series_info import (DicomSeriesImageInfo, DicomSeriesRegistrationInfo,
                          DicomSeriesRTSSInfo, RegistrationInfo, SeriesInfo)

__all__ = [
    "Converter",
    "DicomImageSeriesConverter",
    "DicomRTSSSeriesConverter",
    "SubjectToRTSSConverter",
    "RTSSToSegmentConverter",
    "SegmentToRTSSConverter2D",
    "SegmentToRTSSConverter3D",
    "RTSSMetaData",
    "RTSSConverter2DConfiguration",
    "RTSSConverter3DConfiguration",
]

ROI_GENERATION_ALGORITHMS = ["AUTOMATIC", "SEMIAUTOMATIC", "MANUAL"]

COLOR_PALETTE = [
    [255, 0, 255],
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
    [255, 225, 0],
]


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
        fill_hole_search_distance (int): The search distance for the hole filling algorithm. If the search distance is
         set to zero the hole filling algorithm is omitted. The search distance must be an odd number larger than 1
         (default: 0).
    """

    def __init__(
        self,
        rtss_dataset: Union[str, Dataset],
        image_datasets: Union[Tuple[str], Tuple[Dataset]],
        registration_dataset: Union[str, Dataset, None] = None,
        fill_hole_search_distance: int = 0,
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

        # store the fill hole search distance
        if fill_hole_search_distance == 0:
            self.fill_hole_distance = 0
        elif fill_hole_search_distance % 2 == 0:
            raise ValueError("The fill hole search distance must be an odd number.")
        elif fill_hole_search_distance == 1:
            raise ValueError("The fill hole search distance must be larger than 1.")
        else:
            self.fill_hole_distance = fill_hole_search_distance

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
        for ref_frame_of_ref in rtss_dataset.get("ReferencedFrameOfReferenceSequence", []):
            for rt_ref_study in ref_frame_of_ref.get("RTReferencedStudySequence", []):
                for rt_ref_series in rt_ref_study.get("RTReferencedSeriesSequence", []):
                    si_uid = rt_ref_series.get("SeriesInstanceUID", None)
                    if si_uid is not None:
                        ref_series_instance_uids.append(str(si_uid))

        if len(ref_series_instance_uids) > 1:
            raise Exception(
                f"Multiple ({len(ref_series_instance_uids)}) referenced SeriesInstanceUIDs "
                "have been retrieved from the RTSS but only one is allowed!"
            )

        if not ref_series_instance_uids:
            raise Exception("No referenced SeriesInstanceUID could be retrieved!")

        return ref_series_instance_uids[0]

    @staticmethod
    def _load_ref_image_datasets(image_paths: Tuple[str], referenced_series_uid: str) -> Tuple[Dataset]:
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
            image_dataset = load_dataset_tag(image_path, (Tag(0x0020, 0x000E),))
            if image_dataset.get("SeriesInstanceUID", "") == referenced_series_uid:
                ref_image_files.append(image_path)

        # load the appropriate image datasets
        ref_image_datasets = []
        for ref_image_file in ref_image_files:
            ref_image_datasets.append(load_dataset(ref_image_file))

        ref_image_datasets.sort(key=get_slice_position, reverse=False)

        return tuple(ref_image_datasets)

    @staticmethod
    def _clean_image_datasets_for_rtss(image_datasets: Tuple[Dataset], referenced_series_uid: str) -> Tuple[Dataset]:
        """Clean the image datasets based on the referenced SeriesInstanceUID.

        Args:
            image_datasets (Tuple[Dataset]): The image Datasets which should be analyzed.
            referenced_series_uid (str): The referenced SeriesInstanceUID to identify the appropriate images.

        Returns:
            Tuple[Dataset]: The image datasets which are referenced by the RTSS Dataset.
        """
        selected = [
            image_dataset
            for image_dataset in image_datasets
            if image_dataset.get("SeriesInstanceUID", None) == referenced_series_uid
        ]

        selected.sort(key=get_slice_position, reverse=False)

        return tuple(selected)

    @staticmethod
    def _get_image_datasets_for_reg(
        image_datasets: Union[Tuple[Dataset, ...], Tuple[str, ...]], registration_dataset: Optional[Dataset]
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
                dataset = load_dataset_tag(path, (Tag(0x0020, 0x000E), Tag(0x0020, 0x000D)))
                criteria = (
                    str(dataset.get("SeriesInstanceUID", "")) in ref_series_uids,
                    str(dataset.get("StudyInstanceUID", "")) in ref_study_uids,
                )
                if all(criteria):
                    selected.append(load_dataset(path))

        else:
            for image_dataset in image_datasets:
                criteria = (
                    str(image_dataset.get("SeriesInstanceUID", "")) in ref_series_uids,
                    str(image_dataset.get("StudyInstanceUID", "")) in ref_study_uids,
                )
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
        criteria = (
            rtss_dataset.get("SOPClassUID", "") == "1.2.840.10008.5.1.4.1.1.481.3",
            hasattr(rtss_dataset, "ROIContourSequence"),
            hasattr(rtss_dataset, "StructureSetROISequence"),
            hasattr(rtss_dataset, "RTROIObservationsSequence"),
        )

        if not all(criteria):
            raise Exception(f'The checked RTSS from subject {rtss_dataset.get("PatientID")} is invalid!')

    # pylint: disable=use-a-generator
    @staticmethod
    def _validate_registration_dataset(reg_dataset: Optional[Dataset], rtss_dataset: Dataset) -> Optional[Dataset]:
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
        for ref_study_item in reg_dataset.get("StudiesContainingOtherReferencedInstancesSequence", []):
            for ref_series_item in ref_study_item.get("ReferencedSeriesSequence", []):
                ref_series_instance_uid = ref_series_item.get("SeriesInstanceUID", None)
                if ref_series_instance_uid:
                    reg_ref_instance_uids.append(ref_series_instance_uid)

        for ref_series_item in reg_dataset.get("ReferencedSeriesSequence", []):
            ref_series_uid = ref_series_item.get("SeriesInstanceUID", None)
            if ref_series_uid:
                reg_ref_instance_uids.append(ref_series_uid)

        # search for references in the rtss dataset
        rtss_ref_instance_uids = []
        for ref_frame_of_ref in rtss_dataset.get("ReferencedFrameOfReferenceSequence", []):
            for rt_ref_study in ref_frame_of_ref.get("RTReferencedStudySequence", []):
                for rt_ref_series in rt_ref_study.get("RTReferencedSeriesSequence", []):
                    ref_series_uid = rt_ref_series.get("SeriesInstanceUID", None)
                    if ref_series_uid:
                        rtss_ref_instance_uids.append(ref_series_uid)

        # check the criteria
        criteria = (
            reg_dataset.get("SOPClassUID", "") == "1.2.840.10008.5.1.4.1.1.66.1",
            hasattr(reg_dataset, "RegistrationSequence"),
            hasattr(reg_dataset, "ReferencedSeriesSequence"),
            all([rt_reference in reg_ref_instance_uids for rt_reference in rtss_ref_instance_uids]),
            len(reg_ref_instance_uids) != 0,
        )

        if not all(criteria):
            print(f'The checked registration from subject {rtss_dataset.get("PatientID", "n/a")} is invalid!')
            return None

        return reg_dataset

    @staticmethod
    def _validate_rtss_image_references(rtss_dataset: Dataset, image_datasets: Tuple[Dataset]) -> None:
        """Validate if the ReferencedSOPInstanceUIDs of the RTSS dataset are contained in the image datasets.

        Args:
            rtss_dataset (Dataset): The RTSS dataset to validate.
            image_datasets (Tuple[Dataset]): The image datasets to be used for comparison.

        Returns:
            None
        """
        # get the ReferencedSOPInstanceUIDs from the RTSS dataset
        ref_sop_instance_uids = []
        for ref_frame_of_ref in rtss_dataset.get("ReferencedFrameOfReferenceSequence", []):
            for rt_ref_study in ref_frame_of_ref.get("RTReferencedStudySequence", []):
                for rt_ref_series in rt_ref_study.get("RTReferencedSeriesSequence", []):
                    for contour_image_entry in rt_ref_series.get("ContourImageSequence", []):
                        ref_sop_instance_uids.append(contour_image_entry.get("ReferencedSOPInstanceUID", ""))

        # get MediaStorageSOPInstanceUIDs from the image datasets
        image_sop_instance_uids = [entry.file_meta.get("MediaStorageSOPInstanceUID") for entry in image_datasets]

        # check if all ReferencedSOPInstanceUIDs are contained in the image datasets
        missing_sop_instance_uids = tuple(set(ref_sop_instance_uids) - set(image_sop_instance_uids))
        if missing_sop_instance_uids:
            raise ValueError(
                "The following ReferencedSOPInstanceUIDs are missing in the image datasets: "
                f"{missing_sop_instance_uids}"
            )

    @staticmethod
    def _get_contour_sequence_by_roi_number(rtss_dataset: Dataset, roi_number: int) -> Optional[Sequence]:
        """Get the ContourSequence by the ROINumber.

        Args:
            rtss_dataset (Dataset): The RT Structure Set dataset to retrieve the ContourSequence from.
            roi_number (int): The ROINumber for which the ContourSequence should be returned.

        Returns:
            Optional[Sequence]: The ContourSequence with the corresponding ROINumber if available, otherwise
             :class:`None`.
        """
        if rtss_dataset.get("ROIContourSequence", None) is None:
            return None

        for roi_contour_entry in rtss_dataset.get("ROIContourSequence", []):
            if str(roi_contour_entry.get("ReferencedROINumber", None)) == str(roi_number):
                if roi_contour_entry.get("ContourSequence", None) is None:
                    return None
                return roi_contour_entry.get("ContourSequence")

        raise Exception(f"Referenced ROI number '{roi_number}' not found")

    @staticmethod
    def _create_empty_series_mask(image_datasets: Tuple[Dataset, ...]) -> np.ndarray:
        """Create an empty numpy array with the shape according to the reference image.

        Returns:
            np.ndarray: The empty numpy array with appropriate shape.
        """
        rows = int(image_datasets[0].get("Rows"))
        columns = int(image_datasets[0].get("Columns"))
        num_slices = len(image_datasets)

        return np.zeros((columns, rows, num_slices)).astype(bool)

    @staticmethod
    def _create_empty_slice_mask(image_dataset: Dataset) -> np.ndarray:
        """Create an empty numpy array representing one slice of the output mask.

        Args:
            image_dataset (Dataset): The dataset providing the spatial information for the empty mask.

        Returns:
            np.ndarray: The empty slice numpy array.
        """
        columns = int(image_dataset.get("Columns"))
        rows = int(image_dataset.get("Rows"))
        return np.zeros((columns, rows)).astype(np.uint8)

    @staticmethod
    def _apply_transformation_to_3d_points(points: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
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
    def _get_patient_to_pixel_transformation_matrix(image_datasets: Tuple[Dataset, ...]) -> np.ndarray:
        """Get the patient to pixel transformation matrix from the first image dataset.

        Args:
            image_datasets (Tuple[Dataset, ...]): The datasets to retrieve the transformation matrix.

        Returns:
            np.ndarray: The transformation matrix.
        """
        offset = np.array(image_datasets[0].get("ImagePositionPatient"))
        row_spacing, column_spacing = image_datasets[0].get("PixelSpacing")
        slice_spacing = get_spacing_between_slices(image_datasets)
        row_direction, column_direction, slice_direction = get_slice_direction(image_datasets[0])

        linear = np.identity(3, dtype=float)
        linear[0, :3] = row_direction / row_spacing
        linear[1, :3] = column_direction / column_spacing
        linear[2, :3] = slice_direction / slice_spacing

        mat = np.identity(4, dtype=float)
        mat[:3, :3] = linear
        mat[:3, 3] = offset.dot(-linear.T)

        return mat

    @staticmethod
    def _get_slice_contour_data(image_dataset: Dataset, contour_sequence: Sequence) -> Tuple[Any, ...]:
        """Get the contour data from the corresponding ContourSequence.

        Args:
            image_dataset (Dataset): The referenced image dataset.
            contour_sequence (Sequence): The ContourSequence.

        Returns:
            Tuple[Any, ...]: The retrieved ContourData.
        """
        slice_contour_data = []

        for contour in contour_sequence:
            for contour_image in contour.get("ContourImageSequence", []):
                if contour_image.get("ReferencedSOPInstanceUID", None) == image_dataset.get("SOPInstanceUID", "1"):
                    slice_contour_data.append(contour.get("ContourData", []))

        return tuple(slice_contour_data)

    @staticmethod
    def _get_slice_mask_from_slice_contour_data(
        image_dataset: Dataset, contour_data: Tuple[Any, ...], transformation_matrix: np.ndarray
    ) -> np.ndarray:
        """Get the slice mask from the ContourData.

        Args:
            image_dataset (Dataset): The referenced image dataset.
            contour_data (Tuple[Any, ...]): The contour data.
            transformation_matrix (np.ndarray): The transformation matrix.

        Returns:
            np.ndarray: The discrete slice mask.
        """
        raw_polygons = []
        for contour_coords in contour_data:
            reshaped_contour_data = np.reshape(contour_coords, [len(contour_coords) // 3, 3])
            translated_contour_data = RTSSToSegmentConverter._apply_transformation_to_3d_points(
                reshaped_contour_data, transformation_matrix
            )
            polygon = [np.around([translated_contour_data[:, :2]]).astype(int)]
            polygon = np.array(polygon).squeeze()
            if polygon.shape[0] > 2:
                raw_polygons.append(polygon.astype(int))

        slice_mask = RTSSToSegmentConverter._create_empty_slice_mask(image_dataset)
        cv.fillPoly(img=slice_mask, pts=raw_polygons, color=1)
        return slice_mask

    @staticmethod
    def _create_mask_from_contour_sequence(
        image_datasets: Tuple[Dataset, ...], contour_sequence: Sequence
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
                mask[:, :, i] = RTSSToSegmentConverter._get_slice_mask_from_slice_contour_data(
                    image_dataset, slice_contour_data, transformation_matrix
                )

        return mask

    @staticmethod
    def _create_image_from_mask(image_datasets: Tuple[Dataset, ...], mask: np.ndarray) -> sitk.Image:
        """Create an image from the numpy segmentation mask with appropriate orientation.

        Args:
            image_datasets (Tuple[Dataset, ...]): The image datasets used to create the image.
            mask (np.ndarray): The mask to be converted into a sitk.Image.

        Returns:
            sitk.Image: The image generated.
        """
        mask = np.swapaxes(mask, 0, 2)
        mask = np.swapaxes(mask, 1, 2)

        image = sitk.GetImageFromArray(mask.astype(np.uint8))
        image = sitk.Cast(image, sitk.sitkUInt8)

        image.SetOrigin(image_datasets[0].get("ImagePositionPatient"))

        row_spacing, column_spacing = image_datasets[0].get("PixelSpacing")
        slice_spacing = get_spacing_between_slices(image_datasets)
        image.SetSpacing((float(row_spacing), float(column_spacing), float(slice_spacing)))

        slice_direction = np.stack(get_slice_direction(image_datasets[0]), axis=0).astype(float).T.flatten()
        slice_direction = slice_direction.tolist()
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
        assert len(registration_infos) <= 2, (
            "The number of registration infos must be at max two but is " f"{len(registration_infos)}!"
        )

        transforms = []
        for registration_info in registration_infos:
            if not registration_info.is_reference_image:
                transforms = registration_info.registration_info.transforms

        if len(transforms) != 1:
            raise NotImplementedError("The use of multiple sequential transformations is currently not supported!")

        return transforms[0]

    @staticmethod
    def _transform_image_datasets(
        image_datasets: Tuple[Dataset, ...], transform: sitk.Transform
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
            image_position_patient = image_dataset.get("ImagePositionPatient")
            position_transformed = list(transform.GetInverse().TransformPoint(image_position_patient))
            image_dataset["ImagePositionPatient"].value = position_transformed

            # transform the image orientation
            image_orientation_patient = image_dataset.get("ImageOrientationPatient")
            vector_0 = np.array(image_orientation_patient[:3])
            vector_1 = np.array(image_orientation_patient[3:6])

            vector_0_transformed = transform.GetInverse().TransformVector(vector_0, (0, 0, 0))
            vector_1_transformed = transform.GetInverse().TransformVector(vector_1, (0, 0, 0))

            new_direction = list(vector_0_transformed + vector_1_transformed)
            image_dataset["ImageOrientationPatient"].value = new_direction

            transformed_image_datasets.append(image_dataset)

        return image_datasets

    @staticmethod
    def _transform_rtss_dataset(rtss_dataset: Dataset, transform: sitk.Transform) -> Dataset:
        """Apply the transformation to the RTSS.

        Args:
            rtss_dataset (Dataset): The dataset to be transformed.
            transform (sitk.Transform): The transformation to apply.

        Returns:
            Dataset: The transformed dataset.
        """
        dataset = deepcopy(rtss_dataset)

        for roi_contour in dataset.get("ROIContourSequence", []):
            for contour_sequence_item in roi_contour.get("ContourSequence", []):
                contour_data = contour_sequence_item.get("ContourData")

                if len(contour_data) % 3 != 0:
                    raise ValueError("The number of contour points must be a multiple of three!")

                transformed_points = []

                for chunk in chunkify(contour_data, 3):
                    transformed_point = list(transform.GetInverse().TransformPoint(np.array(chunk)))
                    transformed_points.extend(transformed_point)

                contour_sequence_item["ContourData"].value = transformed_points

        return dataset

    @staticmethod
    def _fill_holes_in_mask(
        mask: np.ndarray,
        search_radius: int = 5,
    ) -> np.ndarray:
        """Fill holes in a binary mask using morphological closing.

        Args:
            mask (np.ndarray): The mask to fill holes in.
            search_radius (int): The radius to search for holes.

        Returns:
            np.ndarray: The mask with filled holes.
        """

        # get the bounding box of the mask
        bb = []
        for ax in itertools.combinations(reversed(range(mask.ndim)), mask.ndim - 1):
            nonzero = np.any(mask, axis=ax)
            bb.extend(np.where(nonzero)[0][[0, -1]])

        # extend the bounding box by the search radius
        bb = np.array(bb)
        bb[::2] = np.maximum(bb[::2] - search_radius, 0)
        bb[1::2] = np.minimum(bb[1::2] + search_radius, mask.shape)

        if mask.ndim == 3:
            inner_mask = mask[bb[0] : bb[1], bb[2] : bb[3], bb[4] : bb[5]]
        else:
            inner_mask = mask[bb[0] : bb[1], bb[2] : bb[3]]

        # fill the holes in the inner mask
        structure = np.ones(tuple([search_radius for _ in range(mask.ndim)]), dtype=int)
        inner_mask = scipy.ndimage.binary_fill_holes(inner_mask, structure).astype(np.bool)

        # remove additional unconnected single voxel segmentations
        inner_mask2 = np.copy(inner_mask).astype(np.uint8)
        id_regions, num_ids = scipy.ndimage.label(inner_mask, structure=np.ones(tuple([3 for _ in range(mask.ndim)])))
        id_sizes = np.array(scipy.ndimage.sum(inner_mask, id_regions, range(num_ids + 1)))
        area_mask = id_sizes == 1
        inner_mask2[area_mask[id_regions]] = 0
        inner_mask = inner_mask2.astype(np.bool)

        # insert the inner mask into the original mask
        new_mask = np.zeros_like(mask, dtype=np.uint8)
        if mask.ndim == 3:
            new_mask[bb[0] : bb[1], bb[2] : bb[3], bb[4] : bb[5]] = inner_mask
        else:
            new_mask[bb[0] : bb[1], bb[2] : bb[3]] = inner_mask

        return new_mask

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
            reg_infos = DicomSeriesRegistrationInfo.get_registration_infos(self.reg_dataset, self.reg_image_datasets)
            transform = self._get_transform_from_registration_info(reg_infos)
            image_datasets = self._transform_image_datasets(self.rtss_image_datasets, transform)
            rtss_dataset = self._transform_rtss_dataset(self.rtss_dataset, transform)

        else:
            image_datasets = self.rtss_image_datasets
            rtss_dataset = self.rtss_dataset

        # convert the contours to images
        for ss_roi in rtss_dataset.get("StructureSetROISequence", []):
            roi_number = int(ss_roi.get("ROINumber"))
            roi_name = str(ss_roi.get("ROIName", ""))

            contour_sequence = self._get_contour_sequence_by_roi_number(rtss_dataset, roi_number)

            if contour_sequence is None:
                continue

            # create the mask from the contours
            mask = self._create_mask_from_contour_sequence(image_datasets, contour_sequence)

            # fill holes in the mask
            if self.fill_hole_distance > 0:
                mask = self._fill_holes_in_mask(mask, search_radius=5)

            # create the image from the mask
            image = self._create_image_from_mask(image_datasets, mask)
            converted_images.update({roi_name: image})

        return converted_images


@dataclass
class RTSSMetaData:
    """A class to define metadata of a new DICOM-RTSS dataset.

    Note:
        Some attributes can take ``None`` as a value. This means that the attribute will be copied from the reference
        DICOM image dataset.

    Note:
        For some attributes, the value must follow the value representation of the DICOM standard. For example, the
        ``PatientSex`` attribute must be either ``'M'``, ``'F'``, or ``'O'``. For more information, we refer to the
        `DICOM standard part 5 chapter 6.2 <https://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.2.html>`_.

    Args:
        patient_name (Optional[str]): The patient name.
        patient_id (Optional[str]): The patient ID.
        patient_birth_date (Optional[str]): The patient birth date (format: YYYYMMDD).
        patient_sex (Optional[str]): The patient sex (valid values: 'F' (female), 'M' (male), 'O' (other)).
        patient_weight (Optional[str]): The patient weight (unit: kilograms with decimals).
        patient_size (Optional[str]): The patient size (unit: meters with decimals).
        study_description (Optional[str]): The study description.
        series_description (Optional[str]): The series description.
        series_number (Optional[str]): The series number (default: '99').
        structure_set_label (str): The structure set label (default: 'Autogenerated').
        operators_name (str): The operator's name (default: 'NA').
        manufacturer (str): The manufacturer (default: 'University of Bern, Switzerland').
        manufacturer_model_name (str): The manufacturer model name (default: 'PyRaDiSe Package').
        institution_name (str): The institution name (default: 'Unknown Institution').
        referring_physician_name (str): The referring physician name (default: 'NA').
        approval_status (str): The approval status (valid values: 'APPROVED', 'UNAPPROVED', 'REJECTED',
         default: 'UNAPPROVED').
        roi_gen_algorithm (str): The ROI generation algorithm (valid values: 'AUTOMATIC', 'SEMIAUTOMATIC', 'MANUAL',
         default: 'AUTOMATIC').

    """

    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    patient_birth_date: Optional[str] = None
    patient_sex: Optional[str] = None
    patient_age: Optional[str] = None
    patient_weight: Optional[str] = None
    patient_size: Optional[str] = None
    study_description: Optional[str] = None
    series_description: Optional[str] = None
    series_number: str = "99"
    structure_set_label: str = "Autogenerated"
    operators_name: str = "NA"
    manufacturer: str = "University of Bern, Switzerland"
    manufacturer_model_name: str = "PyRaDiSe Package"
    institution_name: str = "Unknown Institution"
    referring_physician_name: str = "NA"
    approval_status: str = "UNAPPROVED"
    roi_gen_algorithm: str = "AUTOMATIC"

    def __post_init__(self) -> None:
        # validate entries
        criteria = (
            isinstance(self.patient_name, str) or self.patient_name is None,
            isinstance(self.patient_id, str) or self.patient_id is None,
            isinstance(self.patient_birth_date, str) or self.patient_birth_date is None,
            isinstance(self.patient_sex, str) or self.patient_sex is None,
            isinstance(self.patient_age, str) or self.patient_age is None,
            isinstance(self.patient_weight, str) or self.patient_weight is None,
            isinstance(self.patient_size, str) or self.patient_size is None,
            isinstance(self.study_description, str) or self.study_description is None,
            isinstance(self.series_description, str) or self.series_description is None,
            isinstance(self.series_number, str),
            isinstance(self.structure_set_label, str),
            isinstance(self.operators_name, str),
            isinstance(self.manufacturer, str),
            isinstance(self.manufacturer_model_name, str),
            isinstance(self.institution_name, str),
            isinstance(self.referring_physician_name, str),
            isinstance(self.approval_status, str) and self.approval_status in ("APPROVED", "UNAPPROVED", "REJECTED"),
            isinstance(self.roi_gen_algorithm, str)
            and self.roi_gen_algorithm
            in (
                "AUTOMATIC",
                "SEMIAUTOMATIC",
                "MANUAL",
            ),
        )

        if not all(criteria):
            raise ValueError("The RTSS meta data is not valid! Please check the input values.")


class RTSSConverterConfiguration(ABC):
    """An abstract base class to parameterize a segment to DICOM-RTSS converter."""

    def __init__(self) -> None:
        super().__init__()
        self.general_params: Dict[str, Any] = {}
        self.image_specific_params: Dict[str, Dict[str, Any]] = {}

    @abstractmethod
    def set_general_params(self, **kwargs: Any) -> None:
        """Set the general parameters of the converter.

        Args:
            **kwargs: The parameters of the converter.

        Returns:
            None
        """
        raise NotImplementedError()

    def get_general_params(self, name: Optional[str] = None) -> Optional[Union[Dict[str, Any], Any]]:
        """Get all or a specific general parameter for the converter.

        Args:
            name (Optional[str]): The name of the parameter to get. If ``None``, all parameters are returned
             (default: None).

        Returns:
            Optional[Union[Dict[str, Any], Any]]: All or the selected parameter.
        """
        if name is None:
            if self.general_params:
                return self.general_params
            return None

        return self.general_params.get(name, None)

    @abstractmethod
    def set_image_params(self, **kwargs: Any) -> None:
        """Set the parameters of the converter for a specific image using an identifier.
        instance.

        Args:
            **kwargs: The parameters of the converter.

        Returns:
            None
        """
        raise NotImplementedError()

    def get_image_params(
        self, image_identifier: str, name: Optional[str] = None
    ) -> Optional[Union[Dict[str, Any], Any]]:
        """Get all parameters for a specific image using its identifier.

        Args:
            image_identifier (str): The identifier of the image
            name (Optional[str]): The name of the parameter to get. If ``None``, all parameters are returned (default:
             None).

        Returns:
            Optional[Union[Dict[str, Any], Any]]: All or the selected parameter belonging to the specific image

        """
        if image_identifier not in self.image_specific_params:
            return None

        specific = self.image_specific_params[image_identifier]
        if name is None:
            if specific:
                return specific
            return None
        return specific.get(name, None)


class RTSSConverter2DConfiguration(RTSSConverterConfiguration):
    """A configuration class to parameterize a :class:`SegmentToRTSSConverter2D` instance.

    The configuration can be used to set the general and image specific conversion parameters of the converter.
    The general parameters are applied to all images except if image specific parameters are provided.
    used for the image.

    The parameters define the following:

    * ``smoothing``: Indicates if Gaussian smoothing is applied.

    * ``smoothing_sigma``: The variance of the discrete Gaussian smoothing kernel.

    * ``smoothing_kernel_size``: The size of the discrete Gaussian smoothing kernel.

    Args:
        smoothing (bool): Whether to smooth the contours or not (default: True).
        smoothing_sigma (float): The variance of the Gaussian smoothing (default: 1.0).
        smoothing_kernel_size (int): The size of the Gaussian smoothing kernel (default: 8).
    """

    def __init__(
        self,
        smoothing: bool = True,
        smoothing_sigma: float = 1.0,
        smoothing_kernel_size: int = 8,
    ) -> None:
        super().__init__()

        self._validate_entries(smoothing, smoothing_sigma, smoothing_kernel_size)

        self.set_general_params(smoothing, smoothing_sigma, smoothing_kernel_size)

    @staticmethod
    def _validate_entries(
        smoothing: bool,
        smoothing_sigma: float,
        smoothing_kernel_size: int,
    ) -> None:
        """Validate the entries of the configuration.

        Args:
            smoothing (bool): Whether to smooth the contours or not.
            smoothing_sigma (float): The variance of the Gaussian smoothing.
            smoothing_kernel_size (int): The size of the Gaussian smoothing kernel.

        Raises:
            ValueError: If the entries are not valid.

        Returns:
            None
        """
        criteria = (
            isinstance(smoothing, bool),
            isinstance(smoothing_sigma, float) and smoothing_sigma > 0,
            isinstance(smoothing_kernel_size, int) and smoothing_kernel_size > 0,
        )

        if not all(criteria):
            raise ValueError("The RTSS converter configuration is not valid! Please check the input values.")

    def set_general_params(
        self,
        smoothing: bool,
        smoothing_sigma: float,
        smoothing_kernel_size: int,
    ) -> None:
        """Set the general parameters for all images except those that have specific parameters.

        Args:
            smoothing (bool): Whether to apply Gaussian smoothing.
            smoothing_sigma (float): The sigma of the Gaussian filter.
            smoothing_kernel_size (int): The kernel size of the Gaussian filter.

        Returns:
            None
        """
        self._validate_entries(smoothing, smoothing_sigma, smoothing_kernel_size)

        self.general_params["smoothing"] = smoothing
        self.general_params["smoothing_sigma"] = smoothing_sigma
        self.general_params["smoothing_kernel_size"] = smoothing_kernel_size

    def set_image_params(
        self, image_identifier: str, smoothing: bool, smoothing_sigma: float, smoothing_kernel_size: int
    ) -> None:
        """Set the parameters of the converter for a specific image using the :class:`~pyradise.data.organ.Organ`
        instance.

        Args:
            image_identifier (str): The image to set the parameter(s) for.
            smoothing (bool): Whether to apply Gaussian smoothing to the image.
            smoothing_sigma (float): The sigma value for the Gaussian smoothing kernel.
            smoothing_kernel_size (int): The kernel size for the Gaussian smoothing kernel.

        Returns:
            None
        """
        self._validate_entries(smoothing, smoothing_sigma, smoothing_kernel_size)

        self.image_specific_params[image_identifier] = {
            "smoothing": smoothing,
            "smoothing_sigma": smoothing_sigma,
            "smoothing_kernel_size": smoothing_kernel_size,
        }


class RTSSConverter3DConfiguration(RTSSConverterConfiguration):
    """A configuration class to parameterize a :class:`SegmentToRTSSConverter3D` instance.

    The configuration can be used to set the general and image specific conversion parameters of the converter.
    The general parameters are applied to all images except if image specific parameters are provided.
    used for the image.

    The parameters define the following:

    * ``image_smoothing``: Whether to apply Gaussian smoothing to the image before 3D model construction.

    * ``image_smoothing_sigma``: The standard deviation value for the Gaussian smoothing kernel.

    * ``image_smoothing_radius``: The radius of the Gaussian smoothing kernel.

    * ``image_smoothing_threshold``: The threshold value for the Gaussian smoothing. All segmentation masks that
      have less foreground voxels than this threshold are not smoothed to avoid deletion.

    * ``decimate_reduction``: The reduction factor for the decimation of the 3D model. This factor defines how much
      vertices are removed from the model during the decimation process (0 = none, 1: all).

    * ``decimate_threshold``: The threshold value for the decimation of the 3D model. All models that arise from a
      segmentation mask with less than the threshold number of foreground voxels are not decimated to avoid
      deletion.

    * ``model_smoothing_iterations``: The number of iterations for the smoothing of the 3D model (Sinc-Filter).
      Typically, 10 to 20 iterations are sufficient for smoothing.

    * ``model_smoothing_pass_band``: The pass band for the smoothing of the 3D model (Sinc-Filter). The closer this
      value is to zero (e.g., 0.001) the stronger the smoothing is. The higher the value (e.g., 0.4) the less the
      smoothing is.

    * ``min_segment_lines``: The minimum number of lines that a segment must have to be considered for the RTSS. All
      segments smaller than this value are discarded.


    Args:
        image_smoothing (bool): Whether to smooth the image before 3D model construction or not (default: False).
        image_smoothing_sigma (float): The standard deviation of the Gaussian smoothing before 3D model construction
         (default: 2.).
        image_smoothing_radius (float): The radius of the Gaussian smoothing before 3D model construction (default: 1.).
        image_smoothing_threshold (float): The minimum number of foreground voxels that must be contained in the
         segmentation mask to trigger Gaussian smoothing (default: 0).
        decimate_reduction (float): The reduction factor for the 3D decimation. The decimation factor is valid
         between 0 and 1 and the lower it is the more smoothing is applied (default: 0.5).
        decimate_threshold (float): The minimum number of foreground voxels that must be contained in the
         segmentation mask to trigger 3D decimation (default: 0).
        model_smoothing_iterations (int): The number of 3D smoothing steps (typically 10 - 20 steps) (default: 10).
        model_smoothing_pass_band (bool): The strength of the 3D smoothing (0.001 - 0.1 = strong smoothing,
         0.1 - 0.5 = intermediate smoothing, 0.5 - 1 = almost no smoothing) (default: 0.25).
        min_segment_lines (int): The minimum number of lines that a segment must have to be considered for the RTSS
         (default: 0).
    """

    def __init__(
        self,
        image_smoothing: bool = False,
        image_smoothing_sigma: float = 2.0,
        image_smoothing_radius: float = 1.0,
        image_smoothing_threshold: float = 0.0,
        decimate_reduction: float = 0.5,
        decimate_threshold: float = 0.0,
        model_smoothing_iterations: int = 10,
        model_smoothing_pass_band: float = 0.25,
        min_segment_lines: int = 0,
    ) -> None:
        super().__init__()

        self.set_general_params(
            image_smoothing,
            image_smoothing_sigma,
            image_smoothing_radius,
            image_smoothing_threshold,
            decimate_reduction,
            decimate_threshold,
            model_smoothing_iterations,
            model_smoothing_pass_band,
            min_segment_lines,
        )

    @staticmethod
    def _validate_entries(
        image_smoothing: bool,
        image_smoothing_sigma: float,
        image_smoothing_radius: float,
        image_smoothing_threshold: float,
        decimate_reduction: float,
        decimate_threshold: float,
        model_smoothing_iterations: int,
        model_smoothing_pass_band: float,
        min_segment_lines: int,
    ) -> None:
        """Validate the entries of the configuration.

        Args:
            image_smoothing (bool): Whether to smooth the image before 3D model construction or not.
            image_smoothing_sigma (float): The standard deviation of the Gaussian smoothing before 3D model
             construction.
            image_smoothing_radius (float): The radius of the Gaussian smoothing before 3D model construction.
            image_smoothing_threshold (float): The minimum number of foreground voxels that must be contained in the
             segmentation mask to trigger Gaussian smoothing.
            decimate_reduction (float): The reduction factor for the 3D decimation. The decimation factor is valid
             between 0 and 1 and the lower it is the more smoothing is applied.
            decimate_threshold (float): The minimum number of foreground voxels that must be contained in the
             segmentation mask to trigger 3D decimation.
            model_smoothing_iterations (int): The number of 3D smoothing steps (typically 15 - 20 steps).
            model_smoothing_pass_band (float): The strength of the 3D smoothing (0.001 - 0.1 = strong smoothing,
             0.5 - 1 = almost no smoothing).
            min_segment_lines (int): The minimum number of lines that a segment must have to be considered for the RTSS.

        Raises:
            ValueError: If the entries are not valid.

        Returns:
            None
        """
        criteria = (
            isinstance(image_smoothing, bool),
            isinstance(image_smoothing_sigma, float) and image_smoothing_sigma > 0,
            isinstance(image_smoothing_radius, float) and image_smoothing_radius > 0,
            isinstance(image_smoothing_threshold, (float, int)) and image_smoothing_threshold >= 0,
            isinstance(decimate_reduction, float) and 0 < decimate_reduction < 0.99,
            isinstance(decimate_threshold, (float, int)) and decimate_threshold >= 0,
            isinstance(model_smoothing_iterations, int) and model_smoothing_iterations >= 0,
            isinstance(model_smoothing_pass_band, float),
            isinstance(min_segment_lines, int),
        )

        if not all(criteria):
            raise ValueError("The RTSS converter configuration is not valid! Please check the input values.")

    def set_general_params(
        self,
        image_smoothing: bool,
        image_smoothing_sigma: float,
        image_smoothing_radius: float,
        image_smoothing_threshold: float,
        decimate_reduction: float,
        decimate_threshold: float,
        model_smoothing_iterations: int,
        model_smoothing_pass_band: float,
        min_segment_lines: int,
    ) -> None:
        """Set the general parameters for all images except those that have specific parameters.

        Args:
            image_smoothing (bool): Whether to smooth the image before 3D model construction or not.
            image_smoothing_sigma (float): The standard deviation of the Gaussian smoothing before 3D model
             construction.
            image_smoothing_radius (float): The radius of the Gaussian smoothing before 3D model construction.
            image_smoothing_threshold (float): The minimum number of foreground voxels that must be contained in the
             segmentation mask to trigger Gaussian smoothing.
            decimate_reduction (float): The reduction factor for the 3D decimation. The decimation factor is valid
             between 0 and 1 and the lower it is the more smoothing is applied.
            decimate_threshold (float): The minimum number of foreground voxels that must be contained in the
             segmentation mask to trigger 3D decimation.
            model_smoothing_iterations (int): The number of 3D smoothing steps (typically 15 - 20 steps).
            model_smoothing_pass_band (bool): The strength of the 3D smoothing (0.001 - 0.1 = strong smoothing,
             0.5 - 1 = almost no smoothing).
            min_segment_lines (int): The minimum number of lines that a segment must have to be considered for the RTSS.

        Returns:
            None
        """
        self._validate_entries(
            image_smoothing,
            image_smoothing_sigma,
            image_smoothing_radius,
            image_smoothing_threshold,
            decimate_reduction,
            decimate_threshold,
            model_smoothing_iterations,
            model_smoothing_pass_band,
            min_segment_lines,
        )

        self.general_params["image_smoothing"] = image_smoothing
        self.general_params["image_smoothing_sigma"] = image_smoothing_sigma
        self.general_params["image_smoothing_radius"] = image_smoothing_radius
        self.general_params["image_smoothing_threshold"] = image_smoothing_threshold
        self.general_params["decimate_reduction"] = decimate_reduction
        self.general_params["decimate_threshold"] = decimate_threshold
        self.general_params["model_smoothing_iterations"] = model_smoothing_iterations
        self.general_params["model_smoothing_pass_band"] = model_smoothing_pass_band
        self.general_params["min_segment_lines"] = min_segment_lines

    def set_image_params(
        self,
        image_identifier: str,
        image_smoothing: bool,
        image_smoothing_sigma: float,
        image_smoothing_radius: float,
        image_smoothing_threshold: float,
        decimate_reduction: float,
        decimate_threshold: float,
        model_smoothing_iterations: int,
        model_smoothing_pass_band: float,
        min_segment_lines: int,
    ) -> None:
        """Set the parameters of the converter for a specific image using an identifier.

        Args:
            image_identifier (str): The identifier that identifies the segmentation mask for which the parameters are
             set.
            image_smoothing (bool): Whether to smooth the image before 3D model construction or not.
            image_smoothing_sigma (float): The variance of the Gaussian smoothing before 3D model construction.
            image_smoothing_radius (float): The radius of the Gaussian smoothing before 3D model construction.
            image_smoothing_threshold (float): The minimum number of foreground voxels that must be contained in the
             segmentation mask to trigger Gaussian smoothing.
            decimate_reduction (float): The reduction factor for the 3D decimation. The decimation factor is valid
             between 0 and 1 and the lower it is the more smoothing is applied.
            decimate_threshold (float): The minimum number of foreground voxels that must be contained in the
             segmentation mask to trigger 3D decimation.
            model_smoothing_iterations (int): The number of 3D smoothing steps (typically 15 - 20 steps).
            model_smoothing_pass_band (bool): The strength of the 3D smoothing (0.001 - 0.1 = strong smoothing,
             0.5 - 1 = almost no smoothing).
            min_segment_lines (int): The minimum number of lines that a segment must have to be considered for the RTSS.

        Returns:
            None
        """

        self._validate_entries(
            image_smoothing,
            image_smoothing_sigma,
            image_smoothing_radius,
            image_smoothing_threshold,
            decimate_reduction,
            decimate_threshold,
            model_smoothing_iterations,
            model_smoothing_pass_band,
            min_segment_lines,
        )

        self.image_specific_params[image_identifier] = {
            "image_smoothing": image_smoothing,
            "image_smoothing_sigma": image_smoothing_sigma,
            "image_smoothing_radius": image_smoothing_radius,
            "image_smoothing_threshold": image_smoothing_threshold,
            "decimate_reduction": decimate_reduction,
            "decimate_threshold": decimate_threshold,
            "model_smoothing_iterations": model_smoothing_iterations,
            "model_smoothing_pass_band": model_smoothing_pass_band,
            "min_segment_lines": min_segment_lines,
        }


class SegmentToRTSSConverterBase(Converter):
    """A base class for low-level segmentation mask to DICOM-RTSS converters. This class is not intended to be used
    directly but rather as a base class for more specific conversion algorithm implementations.

    Args:
        label_images (Union[Tuple[str, ...], Tuple[sitk.Image, ...]]): The path to the images or a sequence of
         :class:`SimpleITK.Image` instances.
        ref_image_datasets (Union[Tuple[str, ...], Tuple[Dataset, ...]]): The referenced DICOM image
         :class:`~pydicom.dataset.Dataset` instances.
        roi_names (Union[Tuple[str, ...], Dict[int, str], None]): The label names which will be assigned to the ROIs.
        colors (Optional[Tuple[Tuple[int, int, int], ...]]): The colors which will be assigned to the ROIs.
        meta_data (RTSSMetaData): The configuration to specify certain DICOM attributes (default: RTSSMetaData()).
    """

    def __init__(
        self,
        label_images: Union[Tuple[str, ...], Tuple[sitk.Image, ...]],
        ref_image_datasets: Union[Tuple[str, ...], Tuple[Dataset, ...]],
        roi_names: Union[Tuple[str, ...], Dict[int, str], None],
        colors: Optional[Tuple[Tuple[int, int, int], ...]],
        config: RTSSConverterConfiguration,
        meta_data: RTSSMetaData = RTSSMetaData(),
    ) -> None:
        super().__init__()

        # get or load the label images
        self.label_images: Tuple[sitk.Image, ...] = (
            label_images
            if isinstance(label_images[0], sitk.Image)
            else tuple([sitk.ReadImage(path, sitk.sitkUInt8) for path in label_images])
        )

        # get or load the reference image datasets
        self.image_datasets: Tuple[Dataset, ...] = (
            ref_image_datasets if isinstance(ref_image_datasets[0], Dataset) else load_datasets(ref_image_datasets)
        )
        self.image_datasets = self._sort_datasets(self.image_datasets)

        # get the ROI names
        if isinstance(roi_names, dict):
            sorted_keys = sorted(roi_names.keys())
            self.roi_names = tuple([str(roi_names.get(key)) for key in sorted_keys])
        elif not roi_names:
            self.roi_names = tuple([f"Structure_{i}" for i in range(len(self.label_images))])
        else:
            self.roi_names = roi_names

        # get the colors
        if not colors:
            self.colors = COLOR_PALETTE
        else:
            self.colors = colors

        # check if the correct number of ROI names and colors are provided
        if len(self.roi_names) < len(self.label_images):
            raise ValueError("The number of ROI names must be equal or larger than the number of label images!")

        if len(self.colors) < len(self.label_images):
            raise ValueError("The number of colors must be equal or larger than the number of label images!")

        # get the meta data
        self.meta_data = meta_data

        # get the configuration
        self.config = config

    @staticmethod
    def _sort_datasets(datasets: Tuple[Dataset, ...]) -> Tuple[Dataset, ...]:
        """Sort the datasets by their patient image position.

        Args:
            datasets (Tuple[Dataset, ...]): The datasets to sort.

        Returns:
            Tuple[Dataset, ...]: The sorted datasets.
        """
        # get the principal axes of the image orientation
        direction = np.array(get_slice_direction(datasets[0]))[2]

        principal_component = np.argmax(np.abs(direction))
        principal_direction = np.sign(direction[principal_component])

        datasets_ = tuple(
            sorted(
                datasets,
                key=lambda dataset: float(datasets[0].ImagePositionPatient[principal_component]),
                reverse=principal_direction < 0,
            )
        )
        return datasets_

    def _validate_label_images(self) -> None:
        """Validate the label images.

        Raises:
            ValueError: If the label images are not binary or the pixel type is not an integer.
        """
        # check if the label images are binary and that they have an integer pixel type
        for image in self.label_images:
            if np.unique(sitk.GetArrayFromImage(image)).size > 2:
                raise ValueError("The label images must be binary!")

            if "float" in image.GetPixelIDTypeAsString():
                raise ValueError("The label images must have an integer pixel type!")

    def _generate_basic_rtss(self) -> FileDataset:
        """Generate the basic RTSS skeleton.

        Returns:
            FileDataset: The basic RT Structure Set.
        """
        # create the file meta for the dataset
        file_meta = FileMetaDataset()
        file_meta.FileMetaInformationGroupLength = 202
        file_meta.FileMetaInformationVersion = b"\x00\x01"
        file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

        # create the dataset
        rtss = FileDataset("rt_struct", {}, file_meta=file_meta, preamble=b"\0" * 128)

        # add the basic information
        now = datetime.now()
        rtss.SpecificCharacterSet = "ISO_IR 100"
        rtss.InstanceCreationDate = now.strftime("%Y%m%d")
        rtss.InstanceCreationTime = now.strftime("%H%M%S.%f")
        rtss.StructureSetLabel = self.meta_data.structure_set_label
        rtss.StructureSetDate = now.strftime("%Y%m%d")
        rtss.StructureSetTime = now.strftime("%H%M%S.%f")
        rtss.Modality = "RTSTRUCT"
        rtss.Manufacturer = self.meta_data.manufacturer
        rtss.ManufacturerModelName = self.meta_data.manufacturer_model_name
        rtss.InstitutionName = self.meta_data.institution_name
        rtss.OperatorsName = self.meta_data.operators_name
        rtss.ApprovalStatus = self.meta_data.approval_status

        rtss.is_little_endian = True
        rtss.is_implicit_VR = True

        # set values already defined in the file meta
        rtss.SOPClassUID = rtss.file_meta.MediaStorageSOPClassUID
        rtss.SOPInstanceUID = rtss.file_meta.MediaStorageSOPInstanceUID

        # add study and series information
        reference_dataset = self.image_datasets[0]
        rtss.StudyDate = reference_dataset.StudyDate
        rtss.SeriesDate = getattr(reference_dataset, "SeriesDate", "")
        rtss.StudyTime = reference_dataset.StudyTime
        rtss.SeriesTime = getattr(reference_dataset, "SeriesTime", "")

        if self.meta_data.study_description is None:
            rtss.StudyDescription = getattr(reference_dataset, "StudyDescription", "")
        else:
            rtss.StudyDescription = self.meta_data.study_description

        if self.meta_data.series_description is None:
            rtss.SeriesDescription = getattr(reference_dataset, "SeriesDescription", "")
        else:
            rtss.SeriesDescription = self.meta_data.series_description

        rtss.StudyInstanceUID = reference_dataset.StudyInstanceUID
        rtss.SeriesInstanceUID = generate_uid()
        rtss.StudyID = reference_dataset.StudyID
        rtss.SeriesNumber = self.meta_data.series_number
        rtss.ReferringPhysicianName = self.meta_data.referring_physician_name
        rtss.AccessionNumber = "0"

        # add the patient information
        if self.meta_data.patient_name is None:
            rtss.PatientName = getattr(reference_dataset, "PatientName", "")
        else:
            rtss.PatientName = self.meta_data.patient_name

        if self.meta_data.patient_id is None:
            rtss.PatientID = getattr(reference_dataset, "PatientID", "")
        else:
            rtss.PatientID = self.meta_data.patient_id

        if self.meta_data.patient_birth_date is None:
            rtss.PatientBirthDate = getattr(reference_dataset, "PatientBirthDate", "")
        else:
            rtss.PatientBirthDate = self.meta_data.patient_birth_date

        if self.meta_data.patient_sex is None:
            rtss.PatientSex = getattr(reference_dataset, "PatientSex", "")
        else:
            rtss.PatientSex = self.meta_data.patient_sex

        if self.meta_data.patient_age is None:
            rtss.PatientAge = getattr(reference_dataset, "PatientAge", "")
        else:
            rtss.PatientAge = self.meta_data.patient_age

        if self.meta_data.patient_weight is None:
            rtss.PatientWeight = getattr(reference_dataset, "PatientWeight", "")
        else:
            rtss.PatientWeight = self.meta_data.patient_weight

        if self.meta_data.patient_size is None:
            rtss.PatientSize = getattr(reference_dataset, "PatientSize", "")
        else:
            rtss.PatientSize = self.meta_data.patient_size

        # construct the ContourImageSequence
        contour_image_sequence = Sequence()
        for image_dataset in self.image_datasets:
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
        rt_referenced_study_entry.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"  # RT Structure Set Storage
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
    def _append_rt_roi_observation(roi_number: int, rtss: Dataset) -> None:
        """Create a RTROIObservationsSequence entry for the given ROI data.

        Args:
            roi_number (int): The ROI data to be used for creating the RTROIObservationsSequence entry.
            rtss (Dataset): The RTSS to be used for creating the RTROIObservationsSequence entry.

        Returns:
            None
        """
        # generate the RTROIObservationsSequence entry
        rt_roi_observation = Dataset()
        rt_roi_observation.ObservationNumber = roi_number
        rt_roi_observation.ReferencedROINumber = roi_number
        rt_roi_observation.ROIObservationDescription = (
            "Type:Soft,Range:*/*,Fill:0,Opacity:0.0,Thickness:1," "LineThickness:2,read-only:false"
        )
        rt_roi_observation.private_creators = "University of Bern, Switzerland"
        rt_roi_observation.RTROIInterpretedType = ""
        rt_roi_observation.ROIInterpreter = ""

        # add the RTROIObservationsSequence entry to the RTSS
        rtss.RTROIObservationsSequence.append(rt_roi_observation)

    def _append_structure_set_roi_sequence_entry(self, roi_name: str, roi_number: int, rtss: Dataset) -> None:
        """Append a StructureSetROISequence entry to the given RTSS.

        Args:
            roi_name (str): The name of the ROI.
            roi_number (int): The number of the ROI.
            rtss (Dataset): The RTSS to be used for creating the StructureSetROISequence entry.

        Returns:
            None
        """
        # generate the StructureSetROISequence entry
        structure_set_roi = Dataset()
        structure_set_roi.ROINumber = roi_number
        structure_set_roi.ReferencedFrameOfReferenceUID = self.image_datasets[0].get("FrameOfReferenceUID")
        structure_set_roi.ROIName = roi_name
        structure_set_roi.ROIDescription = ""
        structure_set_roi.ROIGenerationAlgorithm = self.meta_data.roi_gen_algorithm

        # add the StructureSetROISequence entry to the RTSS
        rtss.StructureSetROISequence.append(structure_set_roi)

    @abstractmethod
    def convert(self) -> Any:
        """Abstract method for starting the conversion procedure.

        Returns:
            Any: The converted data.
        """
        raise NotImplementedError()


class SegmentToRTSSConverter2D(SegmentToRTSSConverterBase):
    """A low-level 2D-based :class:`Converter` class for converting one or multiple
    :class:`~pyradise.data.image.SegmentationImage` instances to a DICOM-RTSS :class:`~pydicom.dataset.Dataset`.
    In contrast to the :class:`SegmentToRTSSConverter3D` class, this class generates the DICOM-RTSS contours using a
    two-dimensional approach. This reduces the computation time and leads to a more robust conversion procedure.
    However, this class has limitations such as the inability to smooth contours in all three dimensions. Furthermore,
    the resulting contours may appear to be artificially generated and not as smooth as the ones generated by the
    :class:`SegmentToRTSSConverter3D` class.

    Warning:
        The provided ``label_images`` must be binary, otherwise the conversion will fail.

    Note:
        Typically, this class is not used directly by the used but via the :class:`SubjectToRTSSConverter` which
        processes :class:`~pyradise.fileio.series_info.DicomSeriesInfo` entries and thus provides a more suitable
        interface.

    Note:
        This class can take a :class:`RTSSMetaData` instance as input to specify certain DICOM attributes of the
        output DICOM-RTSS. If no instance is provided, the default values will be used.

    Args:
        label_images (Union[Tuple[str, ...], Tuple[sitk.Image, ...]]): The path to the images or a sequence of
         :class:`SimpleITK.Image` instances.
        ref_image_datasets (Union[Tuple[str, ...], Tuple[Dataset, ...]]): The referenced DICOM image
         :class:`~pydicom.dataset.Dataset` instances.
        roi_names (Union[Tuple[str, ...], Dict[int, str], None]): The label names which will be assigned to the ROIs.
        colors (Optional[Tuple[Tuple[int, int, int], ...]]): The colors which will be assigned to the ROIs.
        meta_data (RTSSMetaData): The configuration to specify certain DICOM attributes (default: RTSSMetaData()).
        config (RTSSConverter2DConfiguration): The configuration to specify certain conversion parameters (default:
         RTSSConverter2DConfiguration()).
    """

    def __init__(
        self,
        label_images: Union[Tuple[str, ...], Tuple[sitk.Image, ...]],
        ref_image_datasets: Union[Tuple[str, ...], Tuple[Dataset, ...]],
        roi_names: Union[Tuple[str, ...], Dict[int, str], None],
        colors: Optional[Tuple[Tuple[int, int, int], ...]],
        meta_data: RTSSMetaData = RTSSMetaData(),
        config: RTSSConverter2DConfiguration = RTSSConverter2DConfiguration(),
    ) -> None:
        super().__init__(label_images, ref_image_datasets, roi_names, colors, config, meta_data)

        self.config = config

    @staticmethod
    def _append_roi_contour(
        mask: np.ndarray,
        image_datasets: Tuple[Dataset, ...],
        rtss: Dataset,
        roi_color: Tuple[int, int, int],
        roi_number: int,
    ) -> None:
        """Create a ROIContourSequence entry for the given ROI data.

        Args:
            mask (np.ndarray): The ROI mask to generate the contours from.
            image_datasets (Tuple[Dataset, ...]): The referenced image datasets.
            rtss (Dataset): The RTSS dataset.

        Returns:
            None
        """
        roi_contour = Dataset()
        roi_contour.ROIDisplayColor = [str(color) for color in roi_color]
        roi_contour.ContourSequence = SegmentToRTSSConverter2D._create_contour_sequence(mask, image_datasets)
        roi_contour.ReferencedROINumber = str(roi_number)

        # add the ROIContourSequence entry to the RTSS
        rtss.ROIContourSequence.append(roi_contour)

    @staticmethod
    def _create_contour_sequence(mask: np.ndarray, image_datasets: Tuple[Dataset, ...]) -> Sequence:
        """Create a ContourSequence for the given ROI data by iterating through each slice of the mask.
        For each connected segment within a slice, a ContourSequence entry is created.

        Args:
            mask (np.ndarray): The ROI mask for generating the contours.
            image_datasets (Tuple[Dataset, ...]): The referenced image datasets.

        Returns:
            Sequence: The created ContourSequence.
        """
        contour_sequence = Sequence()

        contours_coordinates = SegmentToRTSSConverter2D._get_contours_coordinates(mask, image_datasets)

        for series_slice, slice_contours in zip(image_datasets, contours_coordinates):
            for contour_data in slice_contours:
                if len(contour_data) <= 3:
                    continue
                contour_seq_entry = SegmentToRTSSConverter2D._create_contour_sequence_entry(series_slice, contour_data)
                contour_sequence.append(contour_seq_entry)

        return contour_sequence

    @staticmethod
    def _get_contours_coordinates(mask: np.ndarray, image_datasets: Tuple[Dataset, ...]) -> List[List[List[float]]]:
        """Get the contour coordinates for each slice of the mask.

        Args:
            mask (np.ndarray): The ROI mask for generating the contours.
            image_datasets (Tuple[Dataset, ...]): The referenced image datasets.

        Returns:
            List[List[List[float]]]: The contour coordinates for each slice of the mask.
        """
        transform_matrix = SegmentToRTSSConverter2D._get_pixel_to_patient_transformation_matrix(image_datasets)

        series_contours = []
        for i in range(len(image_datasets)):
            mask_slice = mask[i, :, :]

            # Do not add ROI's for blank slices
            if np.sum(mask_slice) == 0:
                series_contours.append([])
                continue

            # Get contours from mask
            contours, _ = SegmentToRTSSConverter2D._find_mask_contours(mask_slice)

            if not contours:
                raise Exception("Unable to find contour in non empty mask, please check your mask formatting!")

            # Format for DICOM
            formatted_contours = []
            for contour in contours:
                # Add z index
                contour = np.concatenate((np.array(contour), np.full((len(contour), 1), i)), axis=1)

                transformed_contour = SegmentToRTSSConverter2D._apply_transformation_to_3d_points(
                    contour, transform_matrix
                )
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

        mat = np.identity(4, dtype=float)
        mat[:3, 0] = row_direction * row_spacing
        mat[:3, 1] = column_direction * column_spacing
        mat[:3, 2] = slice_direction * slice_spacing
        mat[:3, 3] = offset

        return mat

    @staticmethod
    def _find_mask_contours(mask: np.ndarray, approximate_contours: bool = True) -> Tuple[List[np.ndarray], List]:
        """Find the contours in the provided mask.

        Args:
            mask (np.ndarray): The mask to be used for finding the contours.
            approximate_contours (bool): Whether to approximate the contours (default: True).

        Returns:
            Tuple[List[np.ndarray], List]: The contours and the hierarchy.
        """
        method = cv.CHAIN_APPROX_SIMPLE if approximate_contours else cv.CHAIN_APPROX_NONE
        contours, hierarchy = cv.findContours(mask.astype(np.uint8), cv.RETR_TREE, method)
        contours = list(contours)

        for i, contour in enumerate(contours):
            contours[i] = [[pos[0][0], pos[0][1]] for pos in contour]
        hierarchy = hierarchy[0]

        return contours, hierarchy

    @staticmethod
    def _apply_transformation_to_3d_points(points: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
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
    def _create_contour_sequence_entry(series_slice: Dataset, contour_data: List[float]) -> Dataset:
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
        contour.ContourGeometricType = "CLOSED_PLANAR"
        contour.NumberOfContourPoints = len(contour_data) // 3  # each contour point consists of an x, y, and z value
        contour.ContourData = contour_data

        return contour

    @staticmethod
    def _adjust_label_image_to_dicom(label_image: sitk.Image, image_datasets: Tuple[Dataset, ...]) -> sitk.Image:
        """Adjust the given label image to the DICOM image.

        Args:
            label_image (sitk.Image): The label image to be adjusted.
            image_datasets (Tuple[Dataset, ...]): The DICOM image datasets.

        Returns:
            sitk.Image: The adjusted label image.
        """
        # construct a reference image from the DICOM datasets

        # compute the direction from the DICOM datasets
        vec_0 = image_datasets[0].ImageOrientationPatient[:3]
        vec_1 = image_datasets[0].ImageOrientationPatient[3:]
        vec_2 = np.cross(np.array(vec_0, dtype=float), np.array(vec_1, dtype=float))
        dicom_direction = np.stack((vec_0, vec_1, vec_2)).astype(float).T.reshape(-1).tolist()

        # compute the origin from the DICOM datasets
        dicom_origin = image_datasets[0].ImagePositionPatient

        # compute the spacing from the DICOM datasets
        dicom_spacing = [
            float(image_datasets[0].PixelSpacing[0]),
            float(image_datasets[0].PixelSpacing[1]),
            get_spacing_between_slices(image_datasets),
        ]

        # compute the size from the DICOM datasets
        # note: sizes must be in z, x, y order because numpy reverses axes
        dicom_size = (len(image_datasets), image_datasets[0].Rows, image_datasets[0].Columns)

        # construct the reference image via numpy
        reference_image_np = np.zeros(dicom_size)
        reference_image_sitk = sitk.GetImageFromArray(reference_image_np)
        reference_image_sitk.SetDirection(dicom_direction)
        reference_image_sitk.SetOrigin(dicom_origin)
        reference_image_sitk.SetSpacing(dicom_spacing)

        # orient the label image according to the reference image
        reference_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            reference_image_sitk.GetDirection()
        )
        label_image_0 = sitk.DICOMOrient(label_image, reference_orientation)

        criteria = []
        criteria.extend(
            [
                reference_image_sitk.GetSpacing()[i] == label_image_0.GetSpacing()[i]
                for i in range(label_image.GetDimension())
            ]
        )
        criteria.extend(
            [
                reference_image_sitk.GetOrigin()[i] == label_image_0.GetOrigin()[i]
                for i in range(label_image.GetDimension())
            ]
        )
        criteria.extend(
            [
                reference_image_sitk.GetDirection()[i] == label_image_0.GetDirection()[i]
                for i in range(label_image.GetDimension())
            ]
        )
        criteria.extend(
            [reference_image_sitk.GetSize()[i] == label_image_0.GetSize()[i] for i in range(label_image.GetDimension())]
        )

        if all(criteria):
            return label_image_0

        # resample the oriented label image to the reference image
        label_image_1 = sitk.Resample(
            label_image_0, reference_image_sitk, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8
        )
        return label_image_1

    @staticmethod
    def _smooth_label_image(label_image: sitk.Image, kernel_size: int, sigma: float) -> sitk.Image:
        """Apply Gaussian smoothing to a label image.

        Args:
            label_image (sitk.Image): The image to be smoothened.
            kernel_size (int): The maximum kernel size for the Gaussian kernel.
            sigma (float): The sigma for the Gaussian Kernel.

        Returns:
             sitk.Image: The smoothened image.
        """
        label_image_0 = sitk.Cast(label_image, sitk.sitkFloat32)

        num_dims = label_image.GetDimension()
        label_image_1 = sitk.DiscreteGaussian(
            label_image_0, variance=[sigma] * num_dims, maximumKernelWidth=kernel_size
        )
        label_image_2 = sitk.BinaryThreshold(label_image_1, 0.5, 1.0, 1, 0)

        return label_image_2

    def convert(self) -> Dataset:
        """Convert the provided :class:`SimpleITK.Image` instances to a DICOM-RTSS :class:`~pydicom.dataset.Dataset`
        instance using a two-dimensional reconstruction algorithm.

        Returns:
            Dataset: The generated DICOM-RTSS :class:`~pydicom.dataset.Dataset`.
        """
        # generate the basic RTSS dataset
        rtss = self._generate_basic_rtss()

        # convert and add the ROIs to the RTSS
        for idx, (label, name, color) in enumerate(zip(self.label_images, self.roi_names, self.colors)):
            # check if a specific parameterization must be used for this image
            if self.config.get_image_params(name) is not None:
                smooth = self.config.get_image_params(name, "smoothing")
                kernel = self.config.get_image_params(name, "smoothing_kernel_size")
                sigma = self.config.get_image_params(name, "smoothing_sigma")
            else:
                config = self.config.get_general_params()
                smooth = config.get("smoothing")
                kernel = config.get("smoothing_kernel_size")
                sigma = config.get("smoothing_sigma")

            # adjust the label image to have the same properties as the DICOM datasets
            label_processed = self._adjust_label_image_to_dicom(label, self.image_datasets)

            # smooth the label image
            if smooth:
                label_processed = self._smooth_label_image(label_processed, kernel, sigma)

            # get the binary image data from the label image
            mask = sitk.GetArrayFromImage(label_processed)

            self._append_roi_contour(mask, self.image_datasets, rtss, color, idx + 1)
            self._append_structure_set_roi_sequence_entry(name, idx + 1, rtss)
            self._append_rt_roi_observation(idx + 1, rtss)

        return rtss


class SegmentToRTSSConverter3D(SegmentToRTSSConverterBase):
    """A low-level 3D-based :class:`Converter` class for converting one or multiple
    :class:`~pyradise.data.image.SegmentationImage` instances to a DICOM-RTSS :class:`~pydicom.dataset.Dataset`.
    In contrast to the :class:`SegmentToRTSSConverter2D` class, this class generates the DICOM-RTSS contours using a
    three-dimensional approach. This reduces spatial inconsistencies but comes at the cost of a longer computation time
    and higher memory consumption. Furthermore, this converter is less robust than its two-dimensional counterpart.
    However, the resulting contours are more accurate and appear more natural it the converter is applied with
    appropriate parameterization.

    Warning:
        The provided ``label_images`` must be binary, otherwise the conversion will fail.

    Note:
        Typically, this class is not used directly by the used but via the :class:`SubjectToRTSSConverter` which
        processes :class:`~pyradise.fileio.series_info.DicomSeriesInfo` entries and thus provides a more suitable
        interface.

    Note:
        This class can take a :class:`RTSSMetaData` instance as input to specify certain DICOM attributes of the
        output DICOM-RTSS. If no instance is provided, the default values will be used.

    Args:
        label_images (Union[Tuple[str, ...], Tuple[sitk.Image, ...]]): The path to the images or a sequence of
         :class:`SimpleITK.Image` instances.
        ref_image_datasets (Union[Tuple[str, ...], Tuple[Dataset, ...]]): The referenced DICOM image
         :class:`~pydicom.dataset.Dataset` instances.
        roi_names (Union[Tuple[str, ...], Dict[int, str], None]): The label names which will be assigned to the ROIs.
        colors (Optional[Tuple[Tuple[int, int, int], ...]]): The colors which will be assigned to the ROIs.
        meta_data (RTSSMetaData): The configuration to specify certain DICOM attributes (default: RTSSMetaData()).
        config (RTSSConverter3DConfiguration): The configuration to specify certain conversion parameters (default:
         RTSSConverter3DConfiguration()).
    """

    def __init__(
        self,
        label_images: Union[Tuple[str, ...], Tuple[sitk.Image, ...]],
        ref_image_datasets: Union[Tuple[str, ...], Tuple[Dataset, ...]],
        roi_names: Union[Tuple[str, ...], Dict[int, str], None],
        colors: Optional[Tuple[Tuple[int, int, int], ...]],
        meta_data: RTSSMetaData = RTSSMetaData(),
        config: RTSSConverter3DConfiguration = RTSSConverter3DConfiguration(),
    ):
        super().__init__(label_images, ref_image_datasets, roi_names, colors, config, meta_data)

        self.config = config

    @staticmethod
    def _has_foreground_on_borders(image: sitk.Image):
        """Check if the provided image has foreground pixels on the borders.

        Args:
            image (sitk.Image): The image to check.

        Returns:
            bool: True if the image has foreground pixels on the borders, False otherwise.
        """
        image_array = sitk.GetArrayFromImage(image)

        return any(
            (
                np.any(image_array[0, :, :]),
                np.any(image_array[-1, :, :]),
                np.any(image_array[:, 0, :]),
                np.any(image_array[:, -1, :]),
                np.any(image_array[:, :, 0]),
                np.any(image_array[:, :, -1]),
            )
        )

    def preprocess_image(self, image_sitk: sitk.Image) -> sitk.Image:
        """Preprocess the provided :class:`SimpleITK.Image` instance such that the image has the same properties as
        the referenced DICOM image series.

        Args:
            image_sitk (sitk.Image): The :class:`SimpleITK.Image` instance to preprocess.

        Returns:
            sitk.Image: The preprocessed :class:`SimpleITK.Image` instance.
        """
        # orient the image to the LPS coordinate system (-> DICOM standard)
        image_sitk = sitk.DICOMOrient(image_sitk, "LPS")

        # resample to match the DICOM image
        origin = self.image_datasets[0].ImagePositionPatient
        slice_spacing = get_spacing_between_slices(self.image_datasets)
        spacing = (
            float(self.image_datasets[0].PixelSpacing[0]),
            float(self.image_datasets[0].PixelSpacing[1]),
            slice_spacing,
        )
        direction = np.array(get_slice_direction(self.image_datasets[0])).T.flatten()
        size = (self.image_datasets[0].Rows, self.image_datasets[0].Columns, len(self.image_datasets))

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputOrigin(origin)
        resampler.SetOutputSpacing(spacing)
        resampler.SetOutputDirection(direction)
        resampler.SetSize(size)
        resampler.SetTransform(sitk.Transform())
        resampler.SetOutputPixelType(sitk.sitkUInt8)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        image_sitk = resampler.Execute(image_sitk)

        # convert to binary
        image_sitk = sitk.BinaryThreshold(image_sitk, 1, 255, 255, 0)

        return image_sitk

    def _get_3d_model(
        self,
        sitk_image: sitk.Image,
        image_smoothing: bool,
        smooth_sigma: float,
        smooth_radius: float,
        smooth_threshold: float,
        decimate_reduction: float,
        decimate_threshold: float,
        model_smooth_iter: int,
        model_smooth_pass_band: float,
    ) -> vtk_dm.vtkPolyData:
        """Generate a 3D model of the provided :class:`SimpleITK.Image` instance.

        Args:
            sitk_image (sitk.Image): The :class:`SimpleITK.Image` instance to generate the 3D model for.
            image_smoothing (bool): Whether to smooth the image before generating the 3D model.
            smooth_sigma (float): The sigma value for the Gaussian smoothing.
            smooth_radius (float): The radius value for the Gaussian smoothing.
            smooth_threshold (float): The threshold value for the Gaussian smoothing.
            decimate_reduction (float): The reduction value for the decimation.
            decimate_threshold (float): The threshold value for the decimation.
            model_smooth_iter (int): The number of iterations for the model smoothing.
            model_smooth_pass_band (float): The pass band value for the model smoothing.

        Returns:
            vtk_dm.vtkPolyData: The 3D model of the provided :class:`SimpleITK.Image` instance.
        """
        # cast the image to vtkImageData
        itk_image = convert_to_itk_image(sitk_image)
        vtk_image = itk.vtk_image_from_image(itk_image)

        # set the direction matrix
        vtk_image.SetDirectionMatrix(sitk_image.GetDirection())

        # pad the image to avoid boundary effects if necessary
        if self._has_foreground_on_borders(sitk_image):
            margin = 10
            extent = vtk_image.GetExtent()
            new_extent = (
                extent[0] - margin,
                extent[1] + margin,
                extent[2] - margin,
                extent[3] + margin,
                extent[4] - margin,
                extent[5] + margin,
            )
            padder = vtk_icore.vtkImageConstantPad()
            padder.SetConstant(0)
            padder.SetInputDataObject(0, vtk_image)
            padder.SetOutputWholeExtent(new_extent)
            padder.Update(0)
            vtk_image = padder.GetOutput()

            # update the image properties
            vtk_image.SetDirectionMatrix(sitk_image.GetDirection())
            vtk_image.SetOrigin(sitk_image.GetOrigin())
            vtk_image.SetSpacing(sitk_image.GetSpacing())

        # apply gaussian smoothing
        foreground_amount = sitk.GetArrayFromImage(sitk_image).sum() / 255
        if foreground_amount > smooth_threshold and image_smoothing:
            gaussian = vtk_igen.vtkImageGaussianSmooth()
            gaussian.SetInputDataObject(0, vtk_image)
            gaussian.SetStandardDeviation(smooth_sigma)
            gaussian.SetRadiusFactor(smooth_radius)
            gaussian.Update(0)
            vtk_image = gaussian.GetOutputDataObject(0)

            # apply the thresholding
            image_threshold = vtk_icore.vtkImageThreshold()
            image_threshold.ThresholdByUpper(127.5)
            image_threshold.SetInValue(255)
            image_threshold.SetOutValue(0)
            image_threshold.SetInputDataObject(0, vtk_image)
            image_threshold.Update(0)
            vtk_image = image_threshold.GetOutputDataObject(0)

            # update the image properties
            vtk_image.SetDirectionMatrix(sitk_image.GetDirection())
            vtk_image.SetOrigin(sitk_image.GetOrigin())
            vtk_image.SetSpacing(sitk_image.GetSpacing())

        # apply flying edges
        flying_edges = vtk_fcore.vtkFlyingEdges3D()
        flying_edges.SetInputDataObject(0, vtk_image)
        flying_edges.SetValue(0, 255)
        flying_edges.ComputeGradientsOff()
        flying_edges.ComputeNormalsOff()
        flying_edges.Update(0)
        model = flying_edges.GetOutputDataObject(0)

        # apply decimation
        if foreground_amount > decimate_threshold:
            decimate = vtk_fcore.vtkDecimatePro()
            decimate.SetInputConnection(0, flying_edges.GetOutputPort(0))
            decimate.SetTargetReduction(decimate_reduction)
            decimate.PreserveTopologyOn()
            decimate.SetMaximumError(1)
            decimate.SplittingOff()
            decimate.SetFeatureAngle(60.0)
            decimate.Update(0)
            model = decimate.GetOutputDataObject(0)

        # smooth via the sinc filter
        if model_smooth_iter > 0:
            smoother = vtk_fcore.vtkWindowedSincPolyDataFilter()
            smoother.SetInputDataObject(0, model)
            smoother.SetNumberOfIterations(model_smooth_iter)
            smoother.BoundarySmoothingOff()
            smoother.FeatureEdgeSmoothingOff()
            smoother.SetFeatureAngle(60.0)
            smoother.SetPassBand(model_smooth_pass_band)
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            smoother.Update(0)
            model = smoother.GetOutputDataObject(0)

        # get normals
        normals = vtk_fcore.vtkPolyDataNormals()
        normals.SetInputDataObject(0, model)
        normals.SetFeatureAngle(60.0)

        # strip the polydata
        stripper = vtk_fcore.vtkStripper()
        stripper.SetInputConnection(0, normals.GetOutputPort(0))
        stripper.JoinContiguousSegmentsOn()
        stripper.Update(0)
        stripped = stripper.GetOutput()

        return stripped

    def _get_2d_contours(
        self, polydata: vtk_dm.vtkPolyData, min_segment_lines: int = 0
    ) -> List[Optional[List[List[List[float]]]]]:
        """Get the 2D contours of the provided :class:`vtk.vtkPolyData` instance that correspond with the referenced
        DICOM image series.

        Args:
            polydata (vtk.vtkPolyData): The :class:`vtk.vtkPolyData` instance to get the 2D contours for.
            min_segment_lines (int): The minimum number of lines that a contour segment must have to be considered
             (default: 0).

        Returns:
            List[Optional[List[List[List[float]]]]]: The 2D contours of the provided :class:`vtk.vtkPolyData` instance
            that correspond with the referenced DICOM image series.
        """
        origin = self.image_datasets[0].ImagePositionPatient
        origin = [float(val) for val in origin]
        first_pos = self.image_datasets[0].ImagePositionPatient
        last_pos = self.image_datasets[-1].ImagePositionPatient
        length = np.abs(np.linalg.norm(np.array(last_pos) - np.array(first_pos)))
        slice_spacing = length / (len(self.image_datasets) - 1)
        normal = tuple(
            np.cross(
                np.array(self.image_datasets[0].get("ImageOrientationPatient")[0:3]),
                np.array(self.image_datasets[0].get("ImageOrientationPatient")[3:6]),
            )
        )

        # create the initial cutting plane
        plane = vtk_dm.vtkPlane()
        plane.SetOrigin(*origin)
        plane.SetNormal(*normal)

        # create the cutter
        cutter = vtk_fcore.vtkCutter()
        cutter.SetInputDataObject(0, polydata)
        cutter.SetCutFunction(plane)
        cutter.GenerateTrianglesOn()
        cutter.GenerateValues(len(self.image_datasets), 0, length)

        # create the cleaner
        cleaner = vtk_fcore.vtkCleanPolyData()
        cleaner.SetInputConnection(0, cutter.GetOutputPort(0))
        cleaner.SetAbsoluteTolerance(0.01)
        cleaner.PointMergingOn()
        cleaner.Update(0)

        # get the polylines
        loop = vtk_fmodel.vtkContourLoopExtraction()
        loop.SetInputConnection(0, cleaner.GetOutputPort(0))
        loop.SetOutputModeToPolylines()
        loop.SetNormal(*normal)
        loop.Update(0)
        looped = loop.GetOutput()

        # sort the polydata
        sorter = vtk_fhybrid.vtkDepthSortPolyData()
        sorter.SetInputDataObject(0, looped)
        sorter.SetVector(*tuple(np.array(normal) * -1))
        sorter.SetOrigin(*origin)
        sorter.SetSortScalars(True)
        sorter.SetDirectionToSpecifiedVector()
        sorter.Update(0)
        looped = sorter.GetOutput()

        # get the polylines for each slice if there are any
        cells = looped.GetLines()
        points = looped.GetPoints()

        indices = vtk_ccore.vtkIdList()
        cell_indicator = cells.GetNextCell(indices)

        # if there are no cells return an empty list
        if cell_indicator == 0:
            return [None for _ in range(len(self.image_datasets))]

        # get the slice planes
        slice_planes = {}
        for slice_idx, dataset in enumerate(self.image_datasets):
            slice_plane = vtk_dm.vtkPlane()
            slice_plane.SetOrigin(*dataset.ImagePositionPatient)
            slice_plane.SetNormal(*normal)

            slice_planes.update({slice_idx: slice_plane})

        # get the contours for the appropriate slices
        contours_points: List[Optional[List[List[float]]]] = [None for _ in range(len(self.image_datasets))]
        while cell_indicator == 1:
            if indices.GetNumberOfIds() <= min_segment_lines:
                cell_indicator = cells.GetNextCell(indices)
                continue

            reference_point = points.GetPoint(indices.GetId(0))

            for slice_idx, slice_plane in slice_planes.items():
                distance = slice_plane.DistanceToPlane(reference_point)

                if distance <= slice_spacing * 0.5:
                    contour_points = []

                    for idx in range(indices.GetNumberOfIds()):
                        contour_points.append(points.GetPoint(indices.GetId(idx)))

                    if isinstance(contours_points[slice_idx], list):
                        contours_points[slice_idx].append(contour_points)
                    else:
                        contours_points[slice_idx] = [contour_points]

                    cell_indicator = cells.GetNextCell(indices)
                    break

        return contours_points

    def _append_roi_contour_sequence_entry(
        self, contours: List[List[List[List[float]]]], color: Tuple[int, int, int], roi_number: int, rtss: Dataset
    ) -> None:
        """Append a ROIContourSequence entry to the given DICOM-RTSS.

        Args:
            contours (List[List[List[List[float]]]]): The 2D contours of the ROI.
            color (Tuple[int, int, int]): The color of the ROI.
            roi_number (int): The ROI number.
            rtss (Dataset): The DICOM-RTSS to append the ROIContourSequence entry to.

        Returns:
            None
        """
        roi_contour = Dataset()
        roi_contour.ROIDisplayColor = [str(color_) for color_ in color]
        roi_contour.ReferencedROINumber = str(roi_number)

        # create the contour sequence
        contour_sequence = Sequence()
        for slice_dataset, slice_coords in zip(self.image_datasets, contours):
            if slice_coords is None:
                continue

            for slice_coord_set in slice_coords:
                # create the contour image sequence
                contour_image = Dataset()
                contour_image.ReferencedSOPClassUID = str(slice_dataset.file_meta.MediaStorageSOPClassUID)
                contour_image.ReferencedSOPInstanceUID = str(slice_dataset.SOPInstanceUID)

                contour_image_sequence = Sequence()
                contour_image_sequence.append(contour_image)

                # append to the contour sequence
                contour = Dataset()
                contour.ContourImageSequence = contour_image_sequence
                contour.ContourGeometricType = "CLOSED_PLANAR"
                contour.NumberOfContourPoints = len(slice_coord_set)
                contour.ContourData = [coord for point in slice_coord_set for coord in point]
                contour_sequence.append(contour)

        roi_contour.ContourSequence = contour_sequence

        # append to the ROIContourSequence to the RTSS
        rtss.ROIContourSequence.append(roi_contour)

    def convert(self) -> Dataset:
        """Convert the provided :class:`SimpleITK.Image` instances to a DICOM-RTSS :class:`~pydicom.dataset.Dataset`
        instance using a three-dimensional reconstruction algorithm.

        Returns:
            Dataset: The generated DICOM-RTSS :class:`~pydicom.dataset.Dataset`.
        """
        rtss = self._generate_basic_rtss()

        for idx, (label, name, color) in enumerate(zip(self.label_images, self.roi_names, self.colors)):
            # check if the image is empty
            label_image_np = sitk.GetArrayFromImage(label)
            if np.sum(label_image_np) == 0:
                contours = [None for _ in range(len(self.image_datasets))]

            else:
                # check if specific parameters are given for this ROI
                if self.config.get_image_params(name) is not None:
                    image_params = self.config.get_image_params(name)
                    image_smooth = image_params.get("image_smoothing")
                    image_smooth_sigma = image_params.get("image_smoothing_sigma")
                    image_smooth_radius = image_params.get("image_smoothing_radius")
                    image_threshold = image_params.get("image_smoothing_threshold")
                    decimate_reduction = image_params.get("decimate_reduction")
                    decimate_threshold = image_params.get("decimate_threshold")
                    model_smoothing_iter = image_params.get("model_smoothing_iterations")
                    model_smoothing_pass_band = image_params.get("model_smoothing_pass_band")
                    min_segment_lines = image_params.get("min_segment_lines")
                else:
                    image_params = self.config.get_general_params()
                    image_smooth = image_params.get("image_smoothing")
                    image_smooth_sigma = image_params.get("image_smoothing_sigma")
                    image_smooth_radius = image_params.get("image_smoothing_radius")
                    image_threshold = image_params.get("image_smoothing_threshold")
                    decimate_reduction = image_params.get("decimate_reduction")
                    decimate_threshold = image_params.get("decimate_threshold")
                    model_smoothing_iter = image_params.get("model_smoothing_iterations")
                    model_smoothing_pass_band = image_params.get("model_smoothing_pass_band")
                    min_segment_lines = image_params.get("min_segment_lines")

                # preprocess the image
                label = self.preprocess_image(label)

                # Get polydata
                polydata = self._get_3d_model(
                    label,
                    image_smooth,
                    image_smooth_sigma,
                    image_smooth_radius,
                    image_threshold,
                    decimate_reduction,
                    decimate_threshold,
                    model_smoothing_iter,
                    model_smoothing_pass_band,
                )

                # Get the contour data
                contours = self._get_2d_contours(polydata, min_segment_lines)

            # enhance the rtss with the data
            self._append_structure_set_roi_sequence_entry(name, idx + 1, rtss)
            self._append_roi_contour_sequence_entry(contours, color, idx + 1, rtss)
            self._append_rt_roi_observation(idx + 1, rtss)

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

    def __init__(
        self,
        image_info: Tuple[DicomSeriesImageInfo, ...],
        registration_info: Tuple[DicomSeriesRegistrationInfo, ...] = tuple(),
    ) -> None:
        super().__init__()
        self.image_info = image_info
        self.reg_info = registration_info

    def _get_image_info_by_series_instance_uid(self, series_instance_uid: str) -> Optional[DicomSeriesImageInfo]:
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
            raise ValueError(f"Multiple image infos detected with the same SeriesInstanceUID ({series_instance_uid})!")

        if not selected:
            return None

        return selected[0]

    def _get_registration_info(
        self,
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
            raise ValueError(
                f"Multiple registration infos detected with the same referenced "
                f"SeriesInstanceUID ({image_info.series_instance_uid})!"
            )

        if not selected:
            return None

        return selected[0]

    @staticmethod
    def _transform_image(image: sitk.Image, transform: sitk.Transform, is_intensity: bool) -> sitk.Image:
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
        default_pixel_value = np.min(image_np).astype(float)

        # compute the new origin
        new_origin = transform.GetInverse().TransformPoint(image.GetOrigin())

        # compute the new direction
        new_direction_0 = transform.TransformVector(image.GetDirection()[:3], image.GetOrigin())
        new_direction_1 = transform.TransformVector(image.GetDirection()[3:6], image.GetOrigin())
        new_direction_2 = transform.TransformVector(image.GetDirection()[6:], image.GetOrigin())
        new_direction = new_direction_0 + new_direction_1 + new_direction_2

        new_direction_matrix = np.array(new_direction).reshape(3, 3)
        original_direction_matrix = np.array(image.GetDirection()).reshape(3, 3)
        new_direction_corr = np.dot(
            np.dot(new_direction_matrix, original_direction_matrix).transpose(), original_direction_matrix
        ).transpose()

        # resample the image
        resampled_image = sitk.Resample(
            image,
            image.GetSize(),
            transform=transform,
            interpolator=interpolator,
            outputOrigin=new_origin,
            outputSpacing=image.GetSpacing(),
            outputDirection=tuple(new_direction_corr.flatten()),
            defaultPixelValue=default_pixel_value,
            outputPixelType=image.GetPixelIDValue(),
        )

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
                image_ = IntensityImage(image, info.modality)
                image_.add_data({"SeriesInstanceUID": info.series_instance_uid})
                images.append(image_)

            # else the image is transformed
            else:
                ref_series_instance_uid = reg_info.referenced_series_instance_uid_identity
                ref_image_info = self._get_image_info_by_series_instance_uid(ref_series_instance_uid)

                if ref_image_info is None:
                    raise ValueError(
                        f"The reference image with SeriesInstanceUID {ref_series_instance_uid} "
                        f"is missing for the registration!"
                    )

                image = self._transform_image(image, reg_info.transform, is_intensity=True)
                image_ = IntensityImage(image, info.modality)
                image_.add_data({"SeriesInstanceUID": info.series_instance_uid})
                images.append(image_)

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
        fill_hole_search_distance (int): The search distance for the hole filling algorithm. If the search distance is
         set to zero the hole filling algorithm is omitted. The search distance must be an odd number larger than 1
         (default: 0).
    """

    def __init__(
        self,
        rtss_infos: Union[DicomSeriesRTSSInfo, Tuple[DicomSeriesRTSSInfo, ...]],
        image_infos: Tuple[DicomSeriesImageInfo, ...],
        registration_infos: Optional[Tuple[DicomSeriesRegistrationInfo, ...]],
        fill_hole_search_distance: int = 0,
    ) -> None:
        super().__init__()

        if isinstance(rtss_infos, DicomSeriesRTSSInfo):
            self.rtss_infos = (rtss_infos,)
        else:
            self.rtss_infos = rtss_infos

        self.image_infos = image_infos
        self.reg_infos = registration_infos

        # store the fill hole search distance
        if fill_hole_search_distance == 0:
            self.fill_hole_distance = 0
        elif fill_hole_search_distance % 2 == 0:
            raise ValueError("The fill hole search distance must be an odd number.")
        elif fill_hole_search_distance == 1:
            raise ValueError("The fill hole search distance must be larger than 1.")
        else:
            self.fill_hole_distance = fill_hole_search_distance

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
            raise ValueError(
                f"Multiple image infos detected with the same referenced "
                f"SeriesInstanceUID ({rtss_info.referenced_instance_uid})!"
            )

        if not selected:
            raise ValueError(
                f"The reference image with the SeriesInstanceUID "
                f"{rtss_info.referenced_instance_uid} for the RTSS conversion is missing!"
            )

        return selected[0]

    def _get_referenced_registration_info(
        self,
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
            raise NotImplementedError(
                "The number of referenced registrations is larger than one! "
                "The sequential application of registrations is not supported yet!"
            )

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

            dataset = load_dataset(rtss_info.get_path()[0])
            structures = RTSSToSegmentConverter(
                dataset, ref_image_info.path, reg_dataset, self.fill_hole_distance
            ).convert()

            for roi_name, segmentation_image in structures.items():
                segmentation = SegmentationImage(segmentation_image, Organ(roi_name), rtss_info.get_annotator())
                segmentation.add_data(
                    {"SeriesInstanceUID": rtss_info.referenced_instance_uid, "ROINames": rtss_info.roi_names}
                )
                images.append(segmentation)

        return tuple(images)


class SubjectToRTSSConverter(Converter):
    """A :class:`Converter` class for converting the :class:`~pyradise.data.image.SegmentationImage` instances of a
    :class:`~pyradise.data.subject.Subject` instance to a :class:`~pydicom.dataset.Dataset` instance.

    Note:
        This class is typically used at the end of a processing pipeline to output a DICOM-RTSS file containing the
        segmentation results of the pipeline.

    Note:
        This class can take a :class:`RTSSMetaData` instance as input to specify certain DICOM attributes of the
        output DICOM-RTSS. If no instance is provided, the default values will be used.

    Important:
        The ``config`` parameter defines the type of conversion algorithm. If specific conversion parameters are
        required for a certain :class:`~pyradise.data.image.SegmentationImage` instance, they must be provided in the
        appropriate :class:`RTSSConverterConfiguration` (i.e., :class:`RTSSConverter2DConfiguration` or
        :class:`RTSSConverter3DConfiguration`). Furthermore, the ``image_identifier`` parameter must be set to the
        organ name of the :class:`~pyradise.data.image.SegmentationImage` instance. Otherwise, segmentation masks with
        specific parameters can not be identified.

    Args:
        subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be converted to a DICOM-RTSS
         :class:`~pydicom.dataset.Dataset` instance.
        infos (Tuple[SeriesInfo, ...]): The :class:`~pyradise.fileio.series_info.DicomSeriesInfo` entries provided for
         the conversion (only :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo` will be considered).
        reference_modality (Union[Modality, str]): The reference :class:`~pyradise.data.modality.Modality` of the
         images to be used for the conversion to DICOM-RTSS.
        config (Union[RTSSConverter2DConfiguration, RTSSConverter3DConfiguration]): The configuration for the conversion
         procedure. The type of conversion configuration determines also the conversion algorithm (2D or 3D) that is
         used.
        meta_data (RTSSMetaData): The configuration to specify certain DICOM attributes (default: RTSSMetaData()).
        colors (Optional[Tuple[Tuple[int, int, int], ...]]): The colors to be used for the segmentation masks. If None,
         the default colors will be used (default: None).
    """

    def __init__(
        self,
        subject: Subject,
        infos: Tuple[SeriesInfo, ...],
        reference_modality: Union[Modality, str],
        config: Union[RTSSConverter2DConfiguration, RTSSConverter3DConfiguration],
        meta_data: RTSSMetaData = RTSSMetaData(),
        colors: Optional[Tuple[Tuple[int, int, int], ...]] = None,
    ) -> None:
        super().__init__()

        assert subject.segmentation_images, "The subject must contain segmentation images!"
        self.subject = subject

        assert infos, "There must be infos provided for the conversion!"

        reference_modality_: Modality = str_to_modality(reference_modality)
        image_infos = [
            entry
            for entry in infos
            if isinstance(entry, DicomSeriesImageInfo) and entry.modality == reference_modality_
        ]

        assert image_infos, "There must be image infos in the provided infos!"

        assert len(image_infos) == 1, "There are multiple image infos fitting the reference modality!"
        self.image_info = image_infos[0]
        self.ref_modality = reference_modality_

        if not isinstance(config, (RTSSConverter2DConfiguration, RTSSConverter3DConfiguration)):
            raise ValueError(f"The config type {type(config)} is not supported!")

        self.config: Union[RTSSConverter2DConfiguration, RTSSConverter3DConfiguration] = config
        self.meta_data = meta_data

        self._validate_colors(colors)
        self.colors = colors

    @staticmethod
    def _validate_colors(colors: Optional[Tuple[Tuple[int, int, int], ...]]) -> None:
        """Validate the provided colors.

        Args:
            colors (Optional[Tuple[Tuple[int, int, int], ...]]): The colors to be validated.

        Raises:
            ValueError: If the provided colors are not valid.

        Returns:
            None: If the provided colors are valid.
        """
        if not colors:
            return

        for color in colors:
            if len(color) != 3:
                raise ValueError(f"The color {color} is not valid!")

            for value in color:
                if not isinstance(value, int) or value < 0 or value > 255:
                    raise ValueError(f"The color {color} is not valid!")

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
            sitk_image = image.get_image_data()
            if "int" not in sitk_image.GetPixelIDTypeAsString():
                try:
                    sitk_image = sitk.Cast(sitk_image, sitk.sitkUInt8)
                except Exception:  # noqa
                    warnings.warn(
                        f"Could not cast SegmentationImage for organ {image.get_organ(True)} to an integer "
                        "type. The SegmentationImage will be skipped!"
                    )
                    continue

            sitk_images.append(sitk_image)
            label_names.append(image.get_organ(as_str=True))

        # load the image datasets
        image_datasets = load_datasets(self.image_info.path)

        # convert the images to a rtss
        if isinstance(self.config, RTSSConverter2DConfiguration):
            rtss = SegmentToRTSSConverter2D(
                label_images=tuple(sitk_images),
                ref_image_datasets=image_datasets,
                roi_names=tuple(label_names),
                colors=self.colors,
                meta_data=self.meta_data,
                config=self.config,
            ).convert()

        elif isinstance(self.config, RTSSConverter3DConfiguration):
            rtss = SegmentToRTSSConverter3D(
                label_images=tuple(sitk_images),
                ref_image_datasets=image_datasets,
                roi_names=tuple(label_names),
                colors=self.colors,
                meta_data=self.meta_data,
                config=self.config,
            ).convert()

        else:
            raise ValueError(f"Invalid configuration type: {type(self.config)}!")

        return rtss


# def show_polydata(polydata: vtk_dm.vtkPolyData) -> None:
#     import vtkmodules.vtkInteractionStyle
#     import vtkmodules.vtkRenderingOpenGL2
#     from vtkmodules.vtkCommonColor import vtkNamedColors
#     from vtkmodules.vtkRenderingCore import (
#         vtkActor,
#         vtkPolyDataMapper,
#         vtkRenderWindow,
#         vtkRenderWindowInteractor,
#         vtkRenderer
#     )
#
#     # Visualize
#     colors = vtkNamedColors()
#
#     mapper = vtkPolyDataMapper()
#     mapper.SetInputData(polydata)
#     actor = vtkActor()
#     actor.SetMapper(mapper)
#     actor.GetProperty().SetLineWidth(4)
#     actor.GetProperty().SetColor(colors.GetColor3d("Peacock"))
#
#     renderer = vtkRenderer()
#     renderWindow = vtkRenderWindow()
#     renderWindow.SetWindowName("Line")
#     renderWindow.AddRenderer(renderer)
#     renderWindowInteractor = vtkRenderWindowInteractor()
#     renderWindowInteractor.SetRenderWindow(renderWindow)
#
#     renderer.SetBackground(*colors.GetColor3d("Silver"))
#     renderer.AddActor(actor)
#
#     renderWindow.Render()
#     renderWindowInteractor.Start()
