from abc import (
    ABC,
    abstractmethod)
from typing import (
    Union,
    Tuple,
    List,
    Sequence,
    Optional)
from dataclasses import dataclass
import re

import numpy as np
import SimpleITK as sitk
from pydicom import Dataset
from pydicom.tag import Tag

from pyradise.data.modality import Modality
from pyradise.data import Rater
from .utils import (
    check_is_file_and_existing,
    check_is_dir_and_existing,
    load_dataset_tag,
    load_dataset)

__all__ = ['SeriesInfo', 'DicomSeriesInfo', 'DicomSeriesImageInfo', 'DicomSeriesRegistrationInfo',
           'DicomSeriesRTStructureSetInfo', 'DicomSeriesInfoFilter', 'DicomSeriesImageInfoFilter',
           'NoDicomSeriesRegistrationInfoFilter', 'RegistrationInfo', 'RegistrationSequenceInfo']


class SeriesInfo(ABC):
    """An abstract series information class.

    Args:
        path (Union[str, Tuple[str, ...]]): The path or paths specifying files.
    """

    def __init__(self,
                 path: Union[str, Tuple[str, ...]]
                 ) -> None:
        super().__init__()

        if isinstance(path, str):
            self.path = (path,)
        else:
            self.path = path

        self._check_paths(self.path)

    @staticmethod
    def _check_paths(paths: Union[str, Tuple[str, ...]],
                     should_be_dir: bool = False
                     ) -> None:
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
                check_is_dir_and_existing(path)

            else:
                check_is_file_and_existing(path)

    def get_path(self) -> Tuple[str]:
        """Get the paths assigned to the info object.

        Returns:
            Tuple[str]: The paths assigned to the info object.
        """
        return self.path

    @abstractmethod
    def update(self) -> None:
        """Update the :class:`SeriesInfo`.

        Returns:
            None
        """
        raise NotImplementedError()


class DicomSeriesInfo(SeriesInfo):
    """An abstract DICOM series info class.

    Args:
        path (Union[str, Tuple[str, ...]]): The path or paths specifying DICOM files.
    """

    def __init__(self,
                 path: Union[str, Tuple[str, ...]],
                 ) -> None:
        super().__init__(path)

        self.patient_id = ''
        self.patient_name = ''
        self.study_instance_uid = ''
        self.study_description = ''
        self.series_instance_uid = ''
        self.series_description = ''
        self.series_number = -1
        self.sop_class_uid = ''
        self.dicom_modality = ''
        self.frame_of_reference_uid = ''

        self._get_dicom_base_info()

        self.is_updated = False

    # noinspection DuplicatedCode
    def _get_dicom_base_info(self,
                             additional_tags: Optional[Sequence[Tag]] = None
                             ) -> Dataset:
        """Get the basis information from the initial DICOM file path.

        Args:
            additional_tags (Optional[Sequence[Tag]]): Additional tags to retrieve from the DICOM file.

        Returns:
            Dataset: The dataset loaded.
        """
        tags = [Tag(0x0010, 0x0020),        # PatientID
                Tag(0x0010, 0x0010),        # PatientName
                Tag(0x0020, 0x000d),        # StudyInstanceUID
                Tag(0x0008, 0x1030),        # StudyDescription
                Tag(0x0020, 0x000e),        # SeriesInstanceUID
                Tag(0x0008, 0x103e),        # SeriesDescription
                Tag(0x0020, 0x0011),        # SeriesNumber
                Tag(0x0008, 0x0016),        # SOPClassUID
                Tag(0x0008, 0x0060),        # Modality
                Tag(0x0020, 0x0052)]        # FrameOfReferenceUID

        if additional_tags:
            tags.extend(additional_tags)

        dataset = load_dataset_tag(self.path[0], tags)

        self.patient_id = str(dataset.get('PatientID', ''))
        self.patient_name = str(dataset.get('PatientName', ''))
        self.study_instance_uid = str(dataset.get('StudyInstanceUID', ''))
        self.study_description = str(dataset.get('StudyDescription', ''))
        self.series_instance_uid = str(dataset.get('SeriesInstanceUID', ''))
        self.series_description = str(dataset.get('SeriesDescription', 'Unnamed_Series'))
        self.series_number = int(dataset.get('SeriesNumber', 111))
        self.sop_class_uid = str(dataset.get('SOPClassUID', ''))
        self.dicom_modality = str(dataset.get('Modality', ''))
        self.frame_of_reference_uid = str(dataset.get('FrameOfReferenceUID', ''))

        minimum_criterion = (self.patient_id != '',
                             self.patient_name != '',
                             self.study_instance_uid != '',
                             self.series_instance_uid != '',
                             self.sop_class_uid != '',
                             self.dicom_modality != '')

        if not all(minimum_criterion):
            raise ValueError(f'At least one necessary DICOM information is not provided for subject {self.patient_id}!')

        return dataset

    @abstractmethod
    def update(self) -> None:
        """Update the :class:`DicomSeriesInfo`.

        Returns:
            None
        """
        raise NotImplementedError()


class DicomSeriesImageInfo(DicomSeriesInfo):
    """A DICOM series image info class.

    Args:
        paths (Tuple[str, ...]): The paths specifying DICOM files.
    """

    def __init__(self,
                 paths: Tuple[str, ...]
                 ) -> None:
        super().__init__(paths)

        self.image = None
        self.modality = Modality.UNKNOWN

    def set_modality(self, modality: Modality) -> None:
        """Set the :class:`Modality`.

        Args:
            modality (Modality): The :class:`Modality`.

        Returns:
            None
        """
        self.modality = modality

    def update(self) -> None:
        """Update the :class:`DicomSeriesImageInfo`.

        Returns:
            None
        """
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(self.path)
        self.image = reader.Execute()

        self.is_updated = True


@dataclass
class ReferenceInfo:
    """A class storing one of multiple reference infos from a DICOM registration file."""
    series_instance_uid: str
    study_instance_uid: str
    is_same_study: bool


@dataclass
class RegistrationSequenceInfo:
    """A class storing one of multiple registration sequence infos from a DICOM registration file.

    Notes:
        For internal use only.
    """
    frame_of_reference_uid: str
    transforms: Tuple[sitk.AffineTransform, ...]
    transform_parameters: Tuple[List, ...]


@dataclass
class RegistrationInfo:
    """A class storing all necessary infos for applying a registration transformation to a DICOM image.

    Notes:
        For internal use only.
    """
    registration_info: RegistrationSequenceInfo
    reference_info: ReferenceInfo
    is_reference_image: bool


class DicomSeriesRegistrationInfo(DicomSeriesInfo):
    """A DICOM series registration info class.

    Args:
        path (Union[str, Tuple[str, ...]]): The path or paths to the DICOM registration files.
        image_infos (Tuple[DicomSeriesImageInfo, ...]): The :class:`DicomSeriesImageInfo` used.
        persistent_image_infos (bool): If True the class holds to the image_infos after updating, otherwise not.
    """

    def __init__(self,
                 path: Union[str, Tuple[str, ...]],
                 image_infos: Tuple[DicomSeriesImageInfo, ...],
                 persistent_image_infos: bool = False
                 ) -> None:
        self.image_infos = image_infos
        self.persistent_image_infos = persistent_image_infos
        self.transform: Optional[sitk.Transform] = None
        self.transform_parameters = tuple()
        self.referenced_series_instance_uid_transform = ''
        self.referenced_series_instance_uid_identity = ''
        self.dataset = None

        super().__init__(path)

        # since the update is lightweight let's update this class upon instantiation
        self.update()

    def _get_dicom_base_info(self,
                             additional_tags: Optional[Sequence[Tag]] = None
                             ) -> Dataset:
        """Get the basis information from the initial DICOM file path.

        Args:
            additional_tags (Optional[Sequence[Tag]]): Additional tags to retrieve from the DICOM file.

        Returns:
            Dataset: The dataset loaded.
        """
        additional_tags_ = [Tag(0x0008, 0x1115),    # ReferencedSeriesSequence
                            Tag(0x0008, 0x1200)]    # StudiesContainingOtherReferencedInstancesSequence

        if additional_tags:
            additional_tags_.extend(additional_tags)

        super()._get_dicom_base_info(additional_tags)

        self.dataset = load_dataset(self.path[0])
        return self.dataset

    @staticmethod
    def get_referenced_series_info(registration_dataset: Dataset) -> Tuple[ReferenceInfo, ...]:
        """Get the :class:`ReferenceInfos` from a dataset.

        Args:
            registration_dataset (Dataset): The registration dataset to extract the infos from.

        Returns:
            Tuple[ReferenceInfo, ...]: The :class:`ReferenceInfo` retrieved from the Dataset.
        """
        referenced_series_instance_uids = []

        referenced_series_sq = registration_dataset.get('ReferencedSeriesSequence', [])
        for item in referenced_series_sq:
            referenced_series_instance_uids.append(ReferenceInfo(str(item.get('SeriesInstanceUID', '')),
                                                                 str(registration_dataset.get('StudyInstanceUID', '')),
                                                                 True))

        other_referenced_series_sq = registration_dataset.get('StudiesContainingOtherReferencedInstancesSequence', [])
        for item in other_referenced_series_sq:
            referenced_series_sq = item.get('ReferencedSeriesSequence', [])
            for referenced_series_item in referenced_series_sq:
                referenced_series_instance_uids.append(ReferenceInfo(
                    str(referenced_series_item.get('SeriesInstanceUID', '')),
                    str(item.get('StudyInstanceUID', '')),
                    False))

        return tuple(referenced_series_instance_uids)

    @staticmethod
    def _get_registration_sequence_info(registration_dataset: Dataset) -> Tuple[RegistrationSequenceInfo, ...]:
        """Get the :class:`RegistrationSequenceInfo` from a dataset.

        Args:
            registration_dataset (Dataset): The registration dataset to extract the information from.

        Returns:
            Tuple[RegistrationSequenceInfo, ...]: The :class:`RegistrationSequenceInfo` retrieved from the dataset.
        """
        registration_info = []

        for item in registration_dataset.get('RegistrationSequence', []):
            frame_of_reference_uid = str(item.get('FrameOfReferenceUID', ''))
            transforms = []
            transform_parameters = []

            for matrix_reg_item in item.get('MatrixRegistrationSequence', []):

                for matrix_item in matrix_reg_item.get('MatrixSequence', []):
                    transform_type = str(matrix_item.get('FrameOfReferenceTransformationMatrixType'))
                    transform_matrix = matrix_item.get('FrameOfReferenceTransformationMatrix')

                    if transform_type == 'RIGID':
                        transform_params = [float(param) for param in transform_matrix]
                        transform_matrix = np.array(transform_params).reshape(4, 4)

                        transform = sitk.AffineTransform(3)
                        transform.SetMatrix(transform_matrix[:3, :3].flatten().tolist())
                        transform.SetTranslation(transform_matrix[:-1, 3:].flatten().tolist())

                        transforms.append(transform.GetInverse())
                        transform_parameters.append(transform_params)

                    else:
                        print('Invalid transform type registered in subject '
                              f'{registration_dataset.get("PatientID", "n/a")} ({transform_type})!')

            registration_info.append(RegistrationSequenceInfo(frame_of_reference_uid,
                                                              tuple(transforms),
                                                              tuple(transform_parameters)))

        return tuple(registration_info)

    @staticmethod
    def _get_unique_series_instance_uid_entries(infos: Union[Tuple[DicomSeriesImageInfo, ...], Tuple[Dataset, ...]]
                                                ) -> Union[Tuple[DicomSeriesImageInfo, ...], Tuple[Dataset, ...]]:
        unique_infos = []

        if isinstance(infos[0], DicomSeriesImageInfo):
            unique_instance_uids = list({info.series_instance_uid for info in infos})
        else:
            unique_instance_uids = list({str(info.get('SeriesInstanceUID')) for info in infos})

        for info in infos:
            if isinstance(info, DicomSeriesImageInfo):

                if info.series_instance_uid in unique_instance_uids:
                    unique_infos.append(info)

                    index = unique_instance_uids.index(info.series_instance_uid)
                    unique_instance_uids.pop(index)
            else:
                if str(info.get('SeriesInstanceUID')) in unique_instance_uids:
                    unique_infos.append(info)

                    index = unique_instance_uids.index(str(info.get('SeriesInstanceUID')))
                    unique_instance_uids.pop(index)

        return tuple(unique_infos)

    @staticmethod
    def get_registration_infos(registration_dataset: Dataset,
                               image_infos: Union[Tuple[DicomSeriesImageInfo, ...], Tuple[Dataset, ...]]
                               ) -> Tuple[RegistrationInfo, ...]:
        """Extract the :class:`RegistrationSequenceInfo` with the corresponding :class:`ReferenceInfo`.

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

        identity_transform = (1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.)

        def is_ident(transform: Tuple[list]):
            for trans in transform:
                yield tuple(trans) == identity_transform

        combined_info = []

        for image_info in internal_image_infos:

            if isinstance(image_info, Dataset):
                if not any(uid in str(image_info.get('SOPClassUID', '')) + '.' for uid in
                           ('1.2.840.10008.5.1.4.1.1.2.', '1.2.840.10008.5.1.4.1.1.4.')):
                    raise ValueError(f'The provided image info SOPClassUID ({str(image_info.get("SOPClassUID", ""))})'
                                     ' is invalid!')

                image_series_instance_uid = image_info.get('SeriesInstanceUID')
                image_study_instance_uid = image_info.get('StudyInstanceUID')
                image_frame_of_reference_uid = image_info.get('FrameOfReferenceUID')

            else:
                image_series_instance_uid = image_info.series_instance_uid
                image_study_instance_uid = image_info.study_instance_uid
                image_frame_of_reference_uid = image_info.frame_of_reference_uid

            for reference_sequence_info in reference_sequence_infos:
                if not all((reference_sequence_info.series_instance_uid == image_series_instance_uid,
                            reference_sequence_info.study_instance_uid == image_study_instance_uid)):
                    continue

                for registration_info in registration_infos:

                    if registration_info.frame_of_reference_uid != image_frame_of_reference_uid:
                        continue

                    combined_info.append(RegistrationInfo(registration_info,
                                                          reference_sequence_info,
                                                          all(is_ident(registration_info.transform_parameters))))

        if len(combined_info) > 2:
            raise ValueError('There are more than two DICOM images assigned to this registration!')

        return tuple(combined_info)

    def set_image_infos(self,
                        image_infos: Tuple[DicomSeriesImageInfo, ...]
                        ) -> None:
        """Set the :class:`DicomSeriesImageInfo`.

        Args:
            image_infos (Tuple[DicomSeriesImageInfo, ...]): The :class:`DicomSeriesImageInfos` to set.

        Returns:
            None
        """
        self.image_infos = image_infos

        self.is_updated = False

    def get_image_infos(self) -> Tuple[DicomSeriesImageInfo, ...]:
        """Get the DicomSeriesImageInfos.

        Returns:
            Tuple[DicomSeriesImageInfo, ...]: A tuple of :class:`DicomSeriesImageInfo`.
        """
        return self.image_infos

    def update(self) -> None:
        """Update the :class:`DicomSeriesRegistrationInfo`.

        Returns:
            None
        """
        if len(self.path) != 1:
            raise ValueError(f'The number of registration files is different from 1 ({len(self.path)}), but must be 1!')

        if not self.image_infos:
            raise ValueError('No image infos are provided and thus no registration is possible!')

        combined_info = self.get_registration_infos(self.dataset, self.image_infos)

        for info in combined_info:

            if info.is_reference_image:
                self.referenced_series_instance_uid_identity = info.reference_info.series_instance_uid

            else:
                self.referenced_series_instance_uid_transform = info.reference_info.series_instance_uid

                if len(info.registration_info.transforms) != 1:
                    raise ValueError(f'The current implementation only supports registration files with '
                                     f'one transformation, but there are {len(info.registration_info.transforms)}!')

                self.transform = info.registration_info.transforms[0]
                self.transform_parameters = info.registration_info.transform_parameters[0]

        if not self.persistent_image_infos:
            self.image_infos = tuple()

        self.is_updated = True


class DicomSeriesRTStructureSetInfo(DicomSeriesInfo):
    """A DICOM series RT Structure Set info class.

    Args:
        path (Union[str, Tuple[str, ...]]): The path or paths specifying DICOM files.
    """

    def __init__(self,
                 path: Union[str, Tuple[str, ...]]
                 ) -> None:
        self.dataset: Optional[Dataset] = None
        self.rater: Optional[Rater] = None
        self.referenced_instance_uid = ''
        super().__init__(path)
        self.is_updated = True

    def _get_dicom_base_info(self,
                             additional_tags: Optional[Sequence[Tag]] = None
                             ) -> Dataset:
        """Get the basis information from the initial DICOM file path.

        Args:
            additional_tags (Optional[Sequence[Tag]]): Additional tags to retrieve from the DICOM file.

        Returns:
            Dataset: The Dataset loaded.
        """
        additional_tags_ = [Tag(0x3006, 0x0002),    # StructureSetLabel
                            Tag(0x0008, 0x1070),    # OperatorName
                            Tag(0x3006, 0x0010),    # ReferencedFrameOfReferenceSequence
                            Tag(0x3006, 0x0039),    # ROIContourSequence
                            Tag(0x3006, 0x0020),    # StructureSetROISequence
                            Tag(0x3006, 0x0080)]    # RTROIObservationsSequence

        if additional_tags:
            additional_tags_.extend(additional_tags)

        self.dataset = super()._get_dicom_base_info(additional_tags_)

        self.rater = self._get_rater()
        self.referenced_instance_uid = self._get_referenced_series_instance_uid()

        self.is_updated = True

        return self.dataset

    def _get_rater(self) -> Rater:
        """Get the :class:`Rater` from the dataset.

        Returns:
            Rater: The Rater.
        """
        operator_name = str(self.dataset.get('OperatorsName', ''))

        if operator_name:
            operator_name = operator_name.replace(' ', '_')
            regex = re.compile('[^\da-zA-Z_-]+')
            operator_name = regex.sub(r'', operator_name)

            abbreviation = ''.join(re.findall(r'[A-Z]+', operator_name))
            abbreviation = abbreviation if abbreviation else None

            return Rater(operator_name, abbreviation)

        return Rater('UNKNOWN', 'NA')

    def _get_referenced_series_instance_uid(self) -> str:
        """Get the referenced SeriesInstanceUID from the dataset.

        Returns:
            str: The referenced SeriesInstanceUID
        """
        referenced_series_instance_uids = []

        ref_frame_of_ref_sq = self.dataset.get('ReferencedFrameOfReferenceSequence', [])
        for ref_frame_ref_item in ref_frame_of_ref_sq:
            rt_ref_study_sq = ref_frame_ref_item.get('RTReferencedStudySequence', [])

            for rt_ref_study_item in rt_ref_study_sq:
                rt_ref_series_sq = rt_ref_study_item.get('RTReferencedSeriesSequence', [])

                for rt_ref_series_item in rt_ref_series_sq:
                    referenced_series_instance_uids.append(str(rt_ref_series_item.get('SeriesInstanceUID')))

        if len(referenced_series_instance_uids) != 1:
            raise ValueError(f'There are multiple or no ({len(referenced_series_instance_uids)}) referenced '
                             f'SeriesInstanceUIDs, but only one is allowed!')

        return referenced_series_instance_uids[0]

    # pylint: disable=unnecessary-pass
    def update(self) -> None:
        """Update the :class:`DicomSeriesRTStructureSetInfo`.

        Returns:
            None
        """
        self.is_updated = True


class DicomSeriesInfoFilter(ABC):
    """An abstract class for filtering :class:`DicomSerieInfo`."""

    @abstractmethod
    def filter(self,
               infos: Tuple[DicomSeriesInfo]
               ) -> Tuple[DicomSeriesInfo, ...]:
        """Execute the filter.

        Args:
            infos (Tuple[DicomSeriesInfo]): The :class:`DicomSerieInfo` to filter.

        Returns:
            Tuple[DicomSeriesInfo]: The filtered :class:`DicomSerieInfos`.
        """
        raise NotImplementedError()


class DicomSeriesImageInfoFilter(DicomSeriesInfoFilter):
    """A class for filtering :class:`DicomSeriesInfo` entries based on the modalities.
    All other :class:`DicomSeriesInfo` will be kept.

    Args:
        keep (Tuple[Modality, ...]): The modalities to keep.
    """

    def __init__(self,
                 keep: Tuple[Modality, ...]
                 ) -> None:
        super().__init__()

        self.keep = keep

    @staticmethod
    def _remove_additional_registrations(series_infos: Tuple[DicomSeriesInfo]) -> List[DicomSeriesInfo]:
        registrations = []
        images = []
        remaining_infos = []

        for series_info in series_infos:
            if isinstance(series_info, DicomSeriesImageInfo):
                images.append(series_info)
            elif isinstance(series_info, DicomSeriesRegistrationInfo):
                registrations.append(series_info)
            else:
                remaining_infos.append(series_info)

        remove_indices = []
        for i, registration in enumerate(registrations):
            criteria_images = [image.series_instance_uid == registration.referenced_series_instance_uid_transform
                               for image in images]

            if not any(criteria_images):
                remove_indices.append(i)

        for idx in reversed(remove_indices):
            registrations.pop(idx)

        filtered_series_infos = []
        filtered_series_infos.extend(images)
        filtered_series_infos.extend(registrations)
        filtered_series_infos.extend(remaining_infos)

        return filtered_series_infos

    def filter(self,
               infos: Tuple[DicomSeriesInfo]
               ) -> Tuple[DicomSeriesInfo, ...]:
        """Execute the filter.

        Args:
            infos (Tuple[DicomSeriesInfo, ...]): The series infos which should be filtered.

        Returns:
            Tuple[DicomSeriesInfo, ...]: The filtered :class:`DicomSeriesInfo`.
        """
        keep: List[DicomSeriesInfo] = []

        for info in infos:
            if isinstance(info, DicomSeriesImageInfo):
                if info.modality in self.keep:
                    keep.append(info)

            else:
                keep.append(info)

        keep = self._remove_additional_registrations(tuple(keep))

        return tuple(keep)


class NoDicomSeriesRegistrationInfoFilter(DicomSeriesInfoFilter):
    """A class for filtering :class:`DicomSeriesInfo` entries by removing all :class:`DicomSeriesRegistrationInfo`.
    """

    def __init__(self) -> None:
        super().__init__()

    def filter(self,
               infos: Tuple[DicomSeriesInfo]
               ) -> Tuple[DicomSeriesInfo, ...]:
        """Execute the filter.

        Args:
            infos (Tuple[DicomSeriesInfo, ...): The series infos which should be filtered.

        Returns:
            Tuple[DicomSeriesInfo, ...]: The filtered :class:`DicomSeriesInfo`.
        """
        keep = []

        for info in infos:
            if not isinstance(info, DicomSeriesRegistrationInfo):
                keep.append(info)

        return tuple(keep)
