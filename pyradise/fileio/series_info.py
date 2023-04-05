import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk
from pydicom import Dataset
from pydicom.tag import Tag

from pyradise.data import (Annotator, Modality, Organ, str_to_annotator,
                           str_to_modality, str_to_organ)
from pyradise.utils import (is_dir_and_exists, is_file_and_exists,
                            load_dataset, load_dataset_tag)

__all__ = [
    "SeriesInfo",
    "FileSeriesInfo",
    "IntensityFileSeriesInfo",
    "SegmentationFileSeriesInfo",
    "DicomSeriesInfo",
    "DicomSeriesImageInfo",
    "DicomSeriesRegistrationInfo",
    "DicomSeriesRTSSInfo",
    "ReferenceInfo",
    "RegistrationInfo",
    "RegistrationSequenceInfo",
]


class SeriesInfo(ABC):
    """An abstract base class for all :class:`SeriesInfo` classes. A :class:`SeriesInfo` class is used to retrieve and
    store content-specific information which is required by the :class:`~pyradise.fileio.loading.SubjectLoader` in
    order to correctly load the data and to construct a :class:`~pyradise.data.subject.Subject`.

    Depending on the type of data, the :class:`SeriesInfo` subclasses retrieve different information which is essential
    for loading and :class:`~pyradise.data.subject.Subject` construction. For example, the
    :class:`DicomSeriesImageInfo` retrieves and holds data about the :class:`~pyradise.data.modality.Modality`
    whereas the :class:`DicomSeriesRTSSInfo` manages information about the :class:`~pyradise.data.annotator.Annotator`
    and information about the referenced DICOM image series.

    Args:
        path (Union[str, Tuple[str, ...]]): The path or paths to the files to load.
    """

    def __init__(self, path: Union[str, Tuple[str, ...]]) -> None:
        super().__init__()

        if isinstance(path, str):
            self.path = (path,)
        else:
            self.path = path

        self._check_paths(self.path)

        self.patient_name = ""
        self.patient_id = ""

        self._is_updated = False

    @staticmethod
    def _check_paths(paths: Union[str, Tuple[str, ...]], should_be_dir: bool = False) -> None:
        """Check if the provided paths are files/directories.

        Args:
            paths (Union[str, Tuple[str, ...]]): The paths for checking.
            should_be_dir (bool): If True the paths will be checked if they are directories.

        Returns:
            None
        """
        if isinstance(paths, str):
            internal_path = (paths,)
        else:
            internal_path = paths

        for path in internal_path:
            if should_be_dir:
                is_dir_and_exists(path)

            else:
                is_file_and_exists(path)

    def get_path(self) -> Tuple[str]:
        """Get the file paths assigned to the :class:`SeriesInfo` instance.

        Returns:
            Tuple[str]: The file paths assigned to the :class:`SeriesInfo` instance.
        """
        return self.path

    def get_patient_name(self) -> str:
        """Get the patient name.

        Returns:
            str: The patient name.
        """
        return self.patient_name

    def get_patient_id(self) -> str:
        """Get the patient ID.

        Returns:
            str: The patient ID.
        """
        return self.patient_id

    def is_updated(self) -> bool:
        """Check if is updated.

        Returns:
            bool: True if is updated, False otherwise.
        """
        return self._is_updated

    @abstractmethod
    def update(self) -> None:
        """Update the :class:`SeriesInfo` instance.

        Returns:
            None
        """
        raise NotImplementedError()


class FileSeriesInfo(SeriesInfo):
    """An abstract base :class:`SeriesInfo` class for all discrete image files (e.g. NIFTI, NRRD, MHA, etc.).

    Important:
        For DICOM files use a  :class:`DicomSeriesInfo` subclass instead.

    Args:
        path (str): The path to the discrete image file to load.
        subject_name (str): The name of the subject.
    """

    def __init__(self, path: str, subject_name: str) -> None:
        super().__init__(path)

        # remove illegal characters in the subject name
        subject_name_ = subject_name.replace(" ", "_")
        pattern = r"""[^\da-zA-Z_-]+"""
        regex = re.compile(pattern)

        # to stay consistent the subject name is called patient name as for DICOM info
        self.patient_name = regex.sub(r"", subject_name_)
        self.patient_id = self.patient_name

    @abstractmethod
    def update(self) -> None:
        """Update the :class:`FileSeriesInfo` instance.

        Returns:
            None
        """
        raise NotImplementedError()


class IntensityFileSeriesInfo(FileSeriesInfo):
    """A :class:`FileSeriesInfo` class for intensity images. In addition to the information provided by the
    :class:`FileSeriesInfo` class, this class contains also a :class:`~pyradise.data.modality.Modality` instance.

    Args:
        path (str): The path to the discrete intensity image file to load.
        subject_name (str): The name of the subject.
        modality (Union[str, Modality]): The modality of the intensity image.
    """

    def __init__(self, path: str, subject_name: str, modality: Union[Modality, str]) -> None:
        if not isinstance(path, str):
            raise TypeError(f"Expected a string for the path but got {type(path)} instead.")
        super().__init__(path, subject_name)

        self.modality: Modality = str_to_modality(modality)

        self.update()

    def get_modality(self) -> Modality:
        """Get the :class:`~pyradise.data.modality.Modality`.

        Returns:
            Modality: The :class:`~pyradise.data.modality.Modality`.
        """
        return self.modality

    def set_modality(self, modality: Modality) -> None:
        """Set the :class:`~pyradise.data.modality.Modality`.

        Args:
            modality (Modality): The :class:`~pyradise.data.modality.Modality` to be set.

        Returns:
            None
        """
        self.modality: Modality = modality

    def update(self) -> None:
        """Update the :class:`IntensityFileSeriesInfo`.

        Returns:
            None
        """
        self._is_updated = True


class SegmentationFileSeriesInfo(FileSeriesInfo):
    """A :class:`FileSeriesInfo` class for segmentation images. In addition to the information provided by the
    :class:`FileSeriesInfo` class, this class contains also an :class:`~pyradise.data.organ.Organ` instance and a
    :class:`~pyradise.data.annotator.Annotator` instance.

    Note:
        We assume that the segmentation image is a binary image with the foreground having the value 1 and the
        background being 0. If your images are different we recommend to separate the segmentation masks into separate
        files because in RT practice segmentations may overlap.

    Args:
        path (str): The path to the discrete segmentation image file to load.
        subject_name (str): The name of the subject.
        organ (Union[Organ, str]): The organ the segmentation is representing.
        annotator (Union[Annotator, str]): The annotator who created the segmentation.
    """

    def __init__(
        self, path: str, subject_name: str, organ: Union[Organ, str], annotator: Union[Annotator, str]
    ) -> None:
        if not isinstance(path, str):
            raise TypeError(f"Expected a single path as a string but got {type(path)}.")
        super().__init__(path, subject_name)

        self.organ: Organ = str_to_organ(organ)
        self.annotator: Annotator = str_to_annotator(annotator)

        self.update()

    def get_organ(self) -> Organ:
        """Get the :class:`~pyradise.data.organ.Organ`.

        Returns:
            Organ: The :class:`~pyradise.data.organ.Organ`.
        """
        return self.organ

    def set_organ(self, organ: Organ) -> None:
        """Set the :class:`~pyradise.data.organ.Organ`.

        Args:
            organ (Organ): The :class:`~pyradise.data.organ.Organ` to be set.

        Returns:
            None
        """
        self.organ: Organ = organ

    def get_annotator(self) -> Annotator:
        """Get the :class:`~pyradise.data.annotator.Annotator`.

        Returns:
            Annotator: The :class:`~pyradise.data.annotator.Annotator`.
        """
        return self.annotator

    def set_annotator(self, annotator: Annotator) -> None:
        """Set the :class:`~pyradise.data.annotator.Annotator`.

        Args:
            annotator (Annotator): The :class:`~pyradise.data.annotator.Annotator` to be set.

        Returns:
            None
        """
        self.annotator: Annotator = annotator

    def update(self) -> None:
        """Update the :class:`SegmentationFileSeriesInfo` instance.

        Returns:
            None
        """
        self._is_updated = True


class DicomSeriesInfo(SeriesInfo):
    """An abstract base :class:`SeriesInfo` class for all DICOM data (i.e. DICOM image, DICOM registration, DICOM-RTSS).

    Important:
        For discrete image files use a  :class:`FileSeriesInfo` subclass instead.

    Args:
        path (Union[str, Tuple[str, ...]]): The path or paths specifying DICOM files to load.
    """

    def __init__(
        self,
        path: Union[str, Tuple[str, ...]],
    ) -> None:
        super().__init__(path)

        self.patient_id = ""
        self.patient_name = ""
        self.study_instance_uid = ""
        self.study_description = ""
        self.series_instance_uid = ""
        self.series_description = ""
        self.series_number = -1
        self.sop_class_uid = ""
        self.dicom_modality = ""
        self.frame_of_reference_uid = ""

        self._get_dicom_base_info()

        self._is_updated = False

    # noinspection DuplicatedCode
    def _get_dicom_base_info(self, additional_tags: Optional[Sequence[Tag]] = None) -> Dataset:
        """Get the basic information from the initial DICOM file path.

        Args:
            additional_tags (Optional[Sequence[Tag]]): Additional tags to retrieve from the DICOM file.

        Returns:
            Dataset: The dataset loaded.
        """
        tags = [
            Tag(0x0010, 0x0020),  # PatientID
            Tag(0x0010, 0x0010),  # PatientName
            Tag(0x0020, 0x000D),  # StudyInstanceUID
            Tag(0x0008, 0x1030),  # StudyDescription
            Tag(0x0020, 0x000E),  # SeriesInstanceUID
            Tag(0x0008, 0x103E),  # SeriesDescription
            Tag(0x0020, 0x0011),  # SeriesNumber
            Tag(0x0008, 0x0016),  # SOPClassUID
            Tag(0x0008, 0x0060),  # Modality
            Tag(0x0020, 0x0052),  # FrameOfReferenceUID
            Tag(0x3006, 0x0002),
        ]  # StructureSetLabel

        if additional_tags:
            tags.extend(additional_tags)

        dataset = load_dataset_tag(self.path[0], tags)

        self.patient_id = str(dataset.get("PatientID", ""))
        self.patient_name = str(dataset.get("PatientName", ""))
        self.study_instance_uid = str(dataset.get("StudyInstanceUID", ""))
        self.study_description = str(dataset.get("StudyDescription", ""))
        self.series_instance_uid = str(dataset.get("SeriesInstanceUID", ""))
        self.series_description = str(dataset.get("SeriesDescription", "Unnamed_Series"))
        self.series_number = str(dataset.get("SeriesNumber", 0) if dataset.get("SeriesNumber", 0) is not None else "")
        self.sop_class_uid = str(dataset.get("SOPClassUID", ""))
        self.dicom_modality = str(dataset.get("Modality", ""))
        self.frame_of_reference_uid = str(dataset.get("FrameOfReferenceUID", ""))
        self.structure_set_label = str(dataset.get("StructureSetLabel", ""))

        minimum_criterion = (
            self.patient_id != "",
            self.patient_name != "",
            self.study_instance_uid != "",
            self.series_instance_uid != "",
            self.sop_class_uid != "",
            self.dicom_modality != "",
        )

        if not all(minimum_criterion):
            raise ValueError(f"At least one necessary DICOM information is not provided for subject {self.patient_id}!")

        return dataset

    @abstractmethod
    def update(self) -> None:
        """Update the :class:`DicomSeriesInfo` instance.

        Returns:
            None
        """
        raise NotImplementedError()


class DicomSeriesImageInfo(DicomSeriesInfo):
    """A :class:`DicomSeriesInfo` class for DICOM images. In addition to the information provided by the
    :class:`DicomSeriesInfo` class, this class contains also a :class:`~pyradise.data.modality.Modality` instance.

    Args:
        paths (Tuple[str, ...]): The paths to the DICOM image files to load.
    """

    def __init__(self, paths: Tuple[str, ...]) -> None:
        super().__init__(paths)

        self.modality = Modality.get_default()

    def get_modality(self) -> Modality:
        """Get the :class:`~pyradise.data.modality.Modality` property.

        Returns:
            Modality: The :class:`~pyradise.data.modality.Modality` property.
        """
        return self.modality

    def set_modality(self, modality: Modality) -> None:
        """Set the :class:`~pyradise.data.modality.Modality`.

        Args:
            modality (Modality): The :class:`~pyradise.data.modality.Modality` to be assigned.

        Returns:
            None
        """
        self.modality = modality

    def update(self) -> None:
        """Update the :class:`DicomSeriesImageInfo` instance.

        Returns:
            None
        """
        self._is_updated = True


# noinspection PyUnresolvedReferences
@dataclass
class ReferenceInfo:
    """A class storing one of multiple reference infos from a DICOM registration file.

    Warning:
        This class is intended for internal use only.

    Args:
        series_instance_uid (str): The SeriesInstanceUID.
        study_instance_uid (str): The StudyInstanceUID.
        is_same_study (bool): Indicates if the series is from the same study as the reference.
    """

    series_instance_uid: str
    study_instance_uid: str
    is_same_study: bool


# noinspection PyUnresolvedReferences
@dataclass
class RegistrationSequenceInfo:
    """A class storing one of multiple registration sequence infos from a DICOM registration file.

    Warning:
        This class is intended for internal use only.

    Args:
        frame_of_reference_uid (str): The FrameOfReferenceUID.
        transforms (Tuple[sitk.AffineTransform, ...]): The transforms.
        transform_parameters (Tuple[List, ...]): The transformation parameters.
    """

    frame_of_reference_uid: str
    transforms: Tuple[sitk.AffineTransform, ...]
    transform_parameters: Tuple[List, ...]


# noinspection PyUnresolvedReferences
@dataclass
class RegistrationInfo:
    """A class storing all necessary infos for applying a registration transformation to a DICOM image.

    Warning:
        This class is intended for internal use only.

    Args:
        registration_info (RegistrationSequenceInfo): The registration sequence info.
        reference_info (ReferenceInfo): The reference info.
        is_reference_image (bool): Indicates if the image is the reference image.
    """

    registration_info: RegistrationSequenceInfo
    reference_info: ReferenceInfo
    is_reference_image: bool


class DicomSeriesRegistrationInfo(DicomSeriesInfo):
    """A :class:`DicomSeriesInfo` class for DICOM registrations. In addition to the information provided by the
    :class:`DicomSeriesInfo` class, this class contains transformation parameters and references to the pair of DICOM
    images associated with the registration.

    Args:
        path (str): The path to the DICOM registration file to load.
        image_infos (Tuple[DicomSeriesImageInfo, ...]): The :class:`DicomSeriesImageInfo` used.
        persistent_image_infos (bool): If True the class holds to the image_infos after updating, otherwise not
         (default: False).
    """

    def __init__(
        self, path: str, image_infos: Tuple[DicomSeriesImageInfo, ...], persistent_image_infos: bool = False
    ) -> None:
        self.dataset = None

        super().__init__(path)

        self.image_infos = image_infos
        self.persistent_image_infos = persistent_image_infos

        self.transform: Optional[sitk.Transform] = None
        self.transform_parameters = tuple()
        self.referenced_series_instance_uid_transform = ""
        self.referenced_series_instance_uid_identity = ""

        # since the update is lightweight let's update this class upon instantiation
        self.update()

    def _get_dicom_base_info(self, additional_tags: Optional[Sequence[Tag]] = None) -> Dataset:
        """Get the basic information from the initial DICOM file path.

        Args:
            additional_tags (Optional[Sequence[Tag]]): Additional tags to retrieve from the DICOM file.

        Returns:
            Dataset: The dataset loaded.
        """
        additional_tags_ = [
            Tag(0x0008, 0x1115),  # ReferencedSeriesSequence
            Tag(0x0008, 0x1200),
        ]  # StudiesContainingOtherReferencedInstancesSequence

        if additional_tags:
            additional_tags_.extend(additional_tags)

        super()._get_dicom_base_info(additional_tags)

        self.dataset = load_dataset(self.path[0])
        return self.dataset

    @staticmethod
    def get_referenced_series_info(registration_dataset: Dataset) -> Tuple[ReferenceInfo, ...]:
        """Get the :class:`ReferenceInfo` entries from a dataset.

        Args:
            registration_dataset (Dataset): The registration dataset to extract the infos from.

        Returns:
            Tuple[ReferenceInfo, ...]: The :class:`ReferenceInfo` retrieved from the Dataset.
        """
        referenced_series_instance_uids = []

        referenced_series_sq = registration_dataset.get("ReferencedSeriesSequence", [])
        for item in referenced_series_sq:
            referenced_series_instance_uids.append(
                ReferenceInfo(
                    str(item.get("SeriesInstanceUID", "")), str(registration_dataset.get("StudyInstanceUID", "")), True
                )
            )

        other_referenced_series_sq = registration_dataset.get("StudiesContainingOtherReferencedInstancesSequence", [])
        for item in other_referenced_series_sq:
            referenced_series_sq = item.get("ReferencedSeriesSequence", [])
            for referenced_series_item in referenced_series_sq:
                referenced_series_instance_uids.append(
                    ReferenceInfo(
                        str(referenced_series_item.get("SeriesInstanceUID", "")),
                        str(item.get("StudyInstanceUID", "")),
                        False,
                    )
                )

        return tuple(referenced_series_instance_uids)

    @staticmethod
    def _get_registration_sequence_info(registration_dataset: Dataset) -> Tuple[RegistrationSequenceInfo, ...]:
        """Get the :class:`RegistrationSequenceInfo` entries from a dataset.

        Args:
            registration_dataset (Dataset): The registration dataset to extract the information from.

        Returns:
            Tuple[RegistrationSequenceInfo, ...]: The :class:`RegistrationSequenceInfo` entries retrieved.
        """
        registration_info = []

        for item in registration_dataset.get("RegistrationSequence", []):
            frame_of_reference_uid = str(item.get("FrameOfReferenceUID", ""))
            transforms = []
            transform_parameters = []

            for matrix_reg_item in item.get("MatrixRegistrationSequence", []):
                for matrix_item in matrix_reg_item.get("MatrixSequence", []):
                    transform_type = str(matrix_item.get("FrameOfReferenceTransformationMatrixType"))
                    transform_matrix = matrix_item.get("FrameOfReferenceTransformationMatrix")

                    if transform_type == "RIGID":
                        transform_params = [float(param) for param in transform_matrix]
                        transform_matrix = np.array(transform_params).reshape(4, 4)

                        transform = sitk.AffineTransform(3)
                        transform.SetMatrix(transform_matrix[:3, :3].flatten().tolist())
                        transform.SetTranslation(transform_matrix[:-1, 3:].flatten().tolist())

                        transforms.append(transform.GetInverse())
                        transform_parameters.append(transform_params)

                    else:
                        print(
                            "Invalid transform type registered in subject "
                            f'{registration_dataset.get("PatientID", "n/a")} ({transform_type})!'
                        )

            registration_info.append(
                RegistrationSequenceInfo(frame_of_reference_uid, tuple(transforms), tuple(transform_parameters))
            )

        return tuple(registration_info)

    @staticmethod
    def _get_unique_series_instance_uid_entries(
        infos: Union[Tuple[DicomSeriesImageInfo, ...], Tuple[Dataset, ...]]
    ) -> Union[Tuple[DicomSeriesImageInfo, ...], Tuple[Dataset, ...]]:
        """Get the unique SeriesInstanceUID entries from a list of :class:`DicomSeriesImageInfo` or datasets.

        Args:
            infos (Union[Tuple[DicomSeriesImageInfo, ...], Tuple[Dataset, ...]]): The infos to extract the unique
             entries from.

        Returns:
            Union[Tuple[DicomSeriesImageInfo, ...], Tuple[Dataset, ...]]: The unique entries.
        """
        unique_infos = []

        if isinstance(infos[0], DicomSeriesImageInfo):
            unique_instance_uids = list({info.series_instance_uid for info in infos})
        else:
            unique_instance_uids = list({str(info.get("SeriesInstanceUID")) for info in infos})

        for info in infos:
            if isinstance(info, DicomSeriesImageInfo):
                if info.series_instance_uid in unique_instance_uids:
                    unique_infos.append(info)

                    index = unique_instance_uids.index(info.series_instance_uid)
                    unique_instance_uids.pop(index)
            else:
                if str(info.get("SeriesInstanceUID")) in unique_instance_uids:
                    unique_infos.append(info)

                    index = unique_instance_uids.index(str(info.get("SeriesInstanceUID")))
                    unique_instance_uids.pop(index)

        return tuple(unique_infos)

    @staticmethod
    def get_registration_infos(
        registration_dataset: Dataset, image_infos: Union[Tuple[DicomSeriesImageInfo, ...], Tuple[Dataset, ...]]
    ) -> Tuple[RegistrationInfo, ...]:
        """Extract the :class:`RegistrationInfo` entries with the corresponding :class:`ReferenceInfo`.

        Args:
            registration_dataset (Dataset): The registration dataset to extract the registration infos from.
            image_infos (Union[Tuple[DicomSeriesImageInfo, ...], Tuple[Dataset, ...]]): The image infos or datasets
             to use for the combination.

        Returns:
            Tuple[RegistrationInfo, ...]: The combined :class:`RegistrationInfo`.
        """

        registration_infos = DicomSeriesRegistrationInfo._get_registration_sequence_info(registration_dataset)

        reference_sequence_infos = DicomSeriesRegistrationInfo.get_referenced_series_info(registration_dataset)

        internal_image_infos = DicomSeriesRegistrationInfo._get_unique_series_instance_uid_entries(image_infos)

        identity_transform = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        def is_ident(transform: Tuple[list]):
            for trans in transform:
                yield tuple(trans) == identity_transform

        combined_info = []

        for image_info in internal_image_infos:
            if isinstance(image_info, Dataset):
                if not any(
                    uid in str(image_info.get("SOPClassUID", "")) + "."
                    for uid in ("1.2.840.10008.5.1.4.1.1.2.", "1.2.840.10008.5.1.4.1.1.4.")
                ):
                    raise ValueError(
                        f'The provided image info SOPClassUID ({str(image_info.get("SOPClassUID", ""))})' " is invalid!"
                    )

                image_series_instance_uid = image_info.get("SeriesInstanceUID")
                image_study_instance_uid = image_info.get("StudyInstanceUID")
                image_frame_of_reference_uid = image_info.get("FrameOfReferenceUID")

            else:
                image_series_instance_uid = image_info.series_instance_uid
                image_study_instance_uid = image_info.study_instance_uid
                image_frame_of_reference_uid = image_info.frame_of_reference_uid

            for reference_sequence_info in reference_sequence_infos:
                if not all(
                    (
                        reference_sequence_info.series_instance_uid == image_series_instance_uid,
                        reference_sequence_info.study_instance_uid == image_study_instance_uid,
                    )
                ):
                    continue

                for registration_info in registration_infos:
                    if registration_info.frame_of_reference_uid != image_frame_of_reference_uid:
                        continue

                    combined_info.append(
                        RegistrationInfo(
                            registration_info,
                            reference_sequence_info,
                            all(is_ident(registration_info.transform_parameters)),
                        )
                    )

        if len(combined_info) > 2:
            warnings.warn("There are more than two DICOM images assigned to a registration!")

        return tuple(combined_info)

    def set_image_infos(self, image_infos: Tuple[DicomSeriesImageInfo, ...]) -> None:
        """Set the :class:`DicomSeriesImageInfo` entries.

        Args:
            image_infos (Tuple[DicomSeriesImageInfo, ...]): The :class:`DicomSeriesImageInfo` entries to set.

        Returns:
            None
        """
        self.image_infos = image_infos

        self._is_updated = False

    def get_image_infos(self) -> Tuple[DicomSeriesImageInfo, ...]:
        """Get the :class:`DicomSeriesImageInfo` entries.

        Returns:
            Tuple[DicomSeriesImageInfo, ...]: The :class:`DicomSeriesImageInfo` entries.
        """
        return self.image_infos

    def update(self) -> None:
        """Update the :class:`DicomSeriesRegistrationInfo`.

        Returns:
            None
        """
        if len(self.path) != 1:
            raise ValueError("Only one registration file path is allowed, but multiple or none are provided!")

        if not self.image_infos:
            raise ValueError("No image infos are provided and thus no registration is possible!")

        if not self.dataset:
            self.dataset = load_dataset(self.path[0])

        combined_info = self.get_registration_infos(self.dataset, self.image_infos)

        for info in combined_info:
            if info.is_reference_image:
                self.referenced_series_instance_uid_identity = info.reference_info.series_instance_uid

            else:
                self.referenced_series_instance_uid_transform = info.reference_info.series_instance_uid

                if len(info.registration_info.transforms) != 1:
                    raise ValueError(
                        f"The current implementation only supports registration files with "
                        f"one transformation, but there are {len(info.registration_info.transforms)}!"
                    )

                self.transform = info.registration_info.transforms[0]
                self.transform_parameters = info.registration_info.transform_parameters[0]

        if not self.persistent_image_infos:
            self.image_infos = tuple()

        self._is_updated = True


class DicomSeriesRTSSInfo(DicomSeriesInfo):
    """A :class:`DicomSeriesInfo` class for DICOM-RTSS. In addition to the information provided by the
    :class:`DicomSeriesInfo` class, this class contains a :class:`~pyradise.data.annotator.Annotator` instance and a
    reference to the DICOM image associated with the DICOM-RTSS.

    Args:
        path (str): The path to the DICOM-RTSS file to load.
    """

    def __init__(self, path: str) -> None:
        self.dataset: Optional[Dataset] = None
        self.annotator: Annotator = Annotator.get_default()
        self.referenced_instance_uid = ""
        self.roi_names = []

        super().__init__(path)

        self.update()

    def _get_dicom_base_info(self, additional_tags: Optional[Sequence[Tag]] = None) -> Dataset:
        """Get the basic information from the initial DICOM file path.

        Args:
            additional_tags (Optional[Sequence[Tag]]): Additional tags to retrieve from the DICOM file.

        Returns:
            Dataset: The Dataset loaded.
        """
        additional_tags_ = [
            Tag(0x3006, 0x0002),  # StructureSetLabel
            Tag(0x0008, 0x1070),  # OperatorName
            Tag(0x3006, 0x0010),  # ReferencedFrameOfReferenceSequence
            Tag(0x3006, 0x0080),  # RTROIObservationsSequence
            Tag(0x3006, 0x0020),
        ]  # StructureSetROISequence

        if additional_tags:
            additional_tags_.extend(additional_tags)

        self.dataset = super()._get_dicom_base_info(additional_tags_)

        self.annotator = self._get_annotator_from_dicom(self.dataset)
        self.referenced_instance_uid = self._get_referenced_series_instance_uid(self.dataset)
        self.roi_names = self._get_roi_names()

        self._is_updated = True

        return self.dataset

    @staticmethod
    def _get_annotator_from_dicom(dataset: Dataset) -> Annotator:
        """Get the :class:`~pyradise.data.annotator.Annotator` from the provided dataset.

        Args:
            dataset (Dataset): The dataset to retrieve the :class:`~pyradise.data.annotator.Annotator` from.

        Returns:
            Annotator: The annotator.
        """
        operator_name = str(dataset.get("OperatorsName", ""))

        if operator_name:
            operator_name = operator_name.replace(" ", "_")
            pattern = r"""[^\da-zA-Z_-]+"""
            regex = re.compile(pattern)
            operator_name = regex.sub(r"", operator_name)

            search_pattern = r"""[A-Z]+"""
            abbreviation = "".join(re.findall(search_pattern, operator_name))
            abbreviation = abbreviation if abbreviation else None

            return Annotator(operator_name, abbreviation)

        return Annotator.get_default()

    def get_annotator(self) -> Annotator:
        """Get the :class:`~pyradise.data.annotator.Annotator`.

        Returns:
            Annotator: The annotator.
        """
        return self.annotator

    @staticmethod
    def _get_referenced_series_instance_uid(dataset: Dataset) -> str:
        """Get the referenced SeriesInstanceUID from the dataset.

        Args:
            dataset (Dataset): The dataset to retrieve the referenced SeriesInstanceUID from.

        Returns:
            str: The referenced SeriesInstanceUID
        """
        referenced_series_instance_uids = []

        ref_frame_of_ref_sq = dataset.get("ReferencedFrameOfReferenceSequence", [])
        for ref_frame_ref_item in ref_frame_of_ref_sq:
            rt_ref_study_sq = ref_frame_ref_item.get("RTReferencedStudySequence", [])

            for rt_ref_study_item in rt_ref_study_sq:
                rt_ref_series_sq = rt_ref_study_item.get("RTReferencedSeriesSequence", [])

                for rt_ref_series_item in rt_ref_series_sq:
                    referenced_series_instance_uids.append(str(rt_ref_series_item.get("SeriesInstanceUID")))

        if len(referenced_series_instance_uids) != 1:
            raise ValueError(
                f"There are multiple or no ({len(referenced_series_instance_uids)}) referenced "
                f"SeriesInstanceUIDs, but only one is allowed!"
            )

        return referenced_series_instance_uids[0]

    def _get_roi_names(self) -> List[str]:
        """Get the ROI names from the dataset.

        Returns:
            List[str]: The ROI names.
        """
        roi_names = []

        roi_sq = self.dataset.get("StructureSetROISequence", [])
        for roi_item in roi_sq:
            roi_names.append(str(roi_item.get("ROIName")))

        return roi_names

    # pylint: disable=unnecessary-pass
    def update(self) -> None:
        """Update the :class:`DicomSeriesRTSSInfo`.

        Returns:
            None
        """
        self._is_updated = True
