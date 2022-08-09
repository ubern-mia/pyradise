from typing import (
    Tuple,
    Optional,
    Union,
    List)

import numpy as np
import SimpleITK as sitk
from pydicom import Dataset

from pyradise.curation.data import (
    Subject,
    IntensityImage,
    SegmentationImage,
    Organ,
    Modality)
from .series_information import (
    DicomSeriesInfo,
    DicomSeriesImageInfo,
    DicomSeriesRegistrationInfo,
    DicomSeriesRTStructureSetInfo)
from .base_conversion import Converter
from .rtss_conversion import (
    RTSSToImageConverter,
    ImageToRTSSConverter)
from .utils import load_datasets


class DicomSeriesImageConverter(Converter):
    """A DICOM image converter which converts DicomSeriesImageInfo into one or multiple IntensityImages."""

    def __init__(self,
                 image_info: Tuple[DicomSeriesImageInfo, ...],
                 registration_info: Tuple[DicomSeriesRegistrationInfo, ...]
                 ) -> None:
        """
        Args:
            image_info (Tuple[DicomSeriesImageInfo, ...]): A tuple of DicomSeriesImageInfo which contain the necessary
             DICOM information for the generation of an IntensityImage.
            registration_info (Tuple[DicomSeriesRegistrationInfo, ...]): A tuple of DicomSeriesRegistrationInfo which
             contain information about the registration of the DICOM series.
        """
        super().__init__()
        self.image_info = image_info
        self.registration_info = registration_info

    def _get_image_info_by_series_instance_uid(self,
                                               series_instance_uid: str
                                               ) -> Optional[DicomSeriesImageInfo]:
        """Gets the DicomSeriesImageInfo entries which match with the specified SeriesInstanceUID.

        Args:
            series_instance_uid (str): The SeriesInstanceUID which must be contained in the returned
             DicomSeriesImageInfo entries.

        Returns:
            Optional[DicomSeriesImageInfo]: The DicomSeriesImageInfo which contains the specified SeriesInstanceUID or
             None
        """
        if not self.image_info:
            return None

        selected = []

        for info in self.image_info:
            if info.series_instance_uid == series_instance_uid:
                selected.append(info)

        if len(selected) > 1:
            raise ValueError(f'Multiple image infos detected with the same SeriesInstanceUID ({series_instance_uid})!')

        if len(selected) == 0:
            return None

        return selected[0]

    # noinspection DuplicatedCode
    def _get_registration_info(self,
                               image_info: DicomSeriesImageInfo,
                               ) -> Optional[DicomSeriesRegistrationInfo]:
        """Gets the DicomSeriesRegistrationInformation which belongs to the specified DicomSeriesImageInfo.

        Args:
            image_info (DicomSeriesImageInfo): The DicomSeriesImageInfo for which the DicomSeriesRegistrationInfo
             is requested.

        Returns:
            Optional[DicomSeriesRegistrationInfo]: The DicomSeriesRegistrationInfo which belongs to the specified
             DicomSeriesImageInfo or None.
        """
        if not self.registration_info:
            return None

        selected = []

        for reg_info in self.registration_info:

            if not reg_info.is_updated:
                reg_info.update()

            if reg_info.referenced_series_instance_uid_transform == image_info.series_instance_uid:
                selected.append(reg_info)

        if len(selected) > 1:
            raise ValueError(f'Multiple registration infos detected with the same referenced '
                             f'SeriesInstanceUID ({image_info.series_instance_uid})!')

        if len(selected) == 0:
            return None

        return selected[0]

    @staticmethod
    def transform_image(image: sitk.Image,
                        transform: sitk.Transform,
                        is_intensity: bool
                        ) -> sitk.Image:
        """Transforms an image.

        Args:
            image (sitk.Image): The image to transform.
            transform (sitk.Transform): The transform to be applied to the image.
            is_intensity (bool): If True the image will be resampled using a B-Spline interpolation function,
             otherwise a nearest neighbour interpolation function will be used.

        Returns:
            sitk.Image: The transformed image.
        """
        if is_intensity:
            interpolator = sitk.sitkBSpline
        else:
            interpolator = sitk.sitkNearestNeighbor

        image_np = sitk.GetArrayFromImage(image)
        default_pixel_value = np.min(image_np).astype(np.float)

        new_origin = transform.GetInverse().TransformPoint(image.GetOrigin())

        new_direction_0 = transform.TransformVector(image.GetDirection()[:3], image.GetOrigin())
        new_direction_1 = transform.TransformVector(image.GetDirection()[3:6], image.GetOrigin())
        new_direction_2 = transform.TransformVector(image.GetDirection()[6:], image.GetOrigin())
        new_direction = new_direction_0 + new_direction_1 + new_direction_2

        new_direction_matrix = np.array(new_direction).reshape(3, 3)
        original_direction_matrix = np.array(image.GetDirection()).reshape(3, 3)
        new_direction_corr = np.dot(np.dot(new_direction_matrix, original_direction_matrix).transpose(),
                                    original_direction_matrix).transpose()

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
        """Converts the specified DicomSeriesImageInfos into IntensityImages.

        Returns:
            Tuple[IntensityImage, ...]: A tuple of IntensityImage
        """
        images = []

        for info in self.image_info:
            reg_info = self._get_registration_info(info)

            if not info.is_updated:
                info.update()

            if reg_info is None:
                images.append(IntensityImage(info.image, info.modality))
                continue

            reference_series_instance_uid = reg_info.referenced_series_instance_uid_identity
            reference_image_info = self._get_image_info_by_series_instance_uid(reference_series_instance_uid)

            if reference_image_info is None:
                raise ValueError(f'The reference image with SeriesInstanceUID {reference_series_instance_uid} '
                                 f'is missing for the registration!')

            info.image = self.transform_image(info.image, reg_info.transform, is_intensity=True)

            images.append(IntensityImage(info.image, info.modality))

        return tuple(images)


class DicomSeriesRTStructureSetConverter(Converter):
    """A DICOM RT Structure Set converter which converts DicomSeriesRTStructureSetInfo into one or multiple
    SegmentationImages.
    """

    def __init__(self,
                 rtss_infos: Union[DicomSeriesRTStructureSetInfo, Tuple[DicomSeriesRTStructureSetInfo, ...]],
                 image_infos: Tuple[DicomSeriesImageInfo, ...],
                 registration_infos: Optional[Tuple[DicomSeriesRegistrationInfo, ...]]
                 ) -> None:
        """
        Args:
            rtss_infos (Union[DicomSeriesRTStructureSetInfo, Tuple[DicomSeriesRTStructureSetInfo, ...]]): The
             DicomSeriesRTStructureSetInfo to be converted to SegmentationImages.
            image_infos (Tuple[DicomSeriesImageInfo, ...]): DicomSeriesImageInfos which will be used as a
             reference image.
            registration_infos(Optional[Tuple[DicomSeriesRegistrationInfo, ...]]: Possible registrations to be used for
             the RT Structure Set.
        """
        super().__init__()

        if isinstance(rtss_infos, DicomSeriesRTStructureSetInfo):
            self.rtss_infos = (rtss_infos,)
        else:
            self.rtss_infos = rtss_infos

        self.image_infos = image_infos
        self.registration_infos = registration_infos

    # noinspection DuplicatedCode
    def _get_referenced_image_info(self,
                                   rtss_info: DicomSeriesRTStructureSetInfo
                                   ) -> Optional[DicomSeriesImageInfo]:
        """Gets the DicomSeriesImageInfo which is referenced in the DicomSeriesRTStructureSetInfo.

        Args:
            rtss_info (DicomSeriesRTStructureSetInfo): The DicomSeriesRTStructureSetInfo for which the reference image
             should be retrieved.

        Returns:
            Optional[DicomSeriesImageInfo]: The DicomSeriesImageInfo representing the reference image for the DICOM
             RT Structure Set or None.
        """
        if not self.image_infos:
            return None

        selected = []

        for image_info in self.image_infos:
            if image_info.series_instance_uid == rtss_info.referenced_instance_uid:
                selected.append(image_info)

        if len(selected) > 1:
            raise ValueError(f'Multiple image infos detected with the same referenced '
                             f'SeriesInstanceUID ({rtss_info.referenced_instance_uid})!')

        if len(selected) == 0:
            raise ValueError(f'The reference image with the SeriesInstanceUID '
                             f'{rtss_info.referenced_instance_uid} for the RTSS conversion is missing!')

        return selected[0]

    def _get_referenced_registration_info(self,
                                          rtss_info: DicomSeriesRTStructureSetInfo,
                                          ) -> Optional[DicomSeriesRegistrationInfo]:
        """Gets the DicomSeriesRegistrationInfo which is referenced in the DicomSeriesRTStructureSetInfo.

        Args:
            rtss_info (DicomSeriesRTStructureSetInfo): The DicomSeriesRTStructureSetInfo for which the referenced
             registration should be retrieved.

        Returns:
            Optional[DicomSeriesRegistrationInfo]: The DicomSeriesRegistrationInfo representing the registration for the
             DICOM RT Structure Set or None.
        """

        if not self.registration_infos:
            return None

        selected = []

        for registration_info in self.registration_infos:
            if registration_info.referenced_series_instance_uid_transform == rtss_info.referenced_instance_uid:
                selected.append(registration_info)

        if not selected:
            return None

        if len(selected) > 1:
            raise NotImplementedError('The number of referenced registrations is larger than one! '
                                      'The sequential application of registrations is not supported yet!')

        return selected[0]

    def convert(self) -> Tuple[SegmentationImage, ...]:
        """Converts the specified DicomSeriesRTStructureSetInfos into SegmentationImages.

        Returns:
            Tuple[SegmentationImage, ...]: A tuple of SegmentationImages.
        """
        images = []

        for rtss_info in self.rtss_infos:

            referenced_image_info = self._get_referenced_image_info(rtss_info)
            referenced_registration_info = self._get_referenced_registration_info(rtss_info)

            if referenced_registration_info:
                registration_dataset = referenced_registration_info.dataset
            else:
                registration_dataset = None

            converted_structures = RTSSToImageConverter(rtss_info.dataset, referenced_image_info.path,
                                                        registration_dataset).convert()

            for roi_name, segmentation_image in converted_structures.items():
                images.append(SegmentationImage(segmentation_image, Organ(roi_name), rtss_info.rater))

        return tuple(images)


class DicomSubjectConverter(Converter):
    """A DICOM converter which converts multiple DicomSeriesInfos into a Subject if the provided information matches."""

    def __init__(self,
                 infos: Tuple[DicomSeriesInfo, ...],
                 disregard_rtss: bool = False,
                 validate_info: bool = True,
                 ) -> None:
        """
        Args:
            infos (Tuple[DicomSeriesInfo, ...]): The DicomSeriesInfo which specify the subject.
            disregard_rtss (bool): If true no RTSS data will be considered.
            validate_info (bool): If true, validates the DicomSeriesInfo entries if they belong to the same patient
             (Default: True).
        """
        super().__init__()
        self.infos = infos
        self.disregard_rtss = disregard_rtss
        self.validate_info = validate_info
        self.image_infos, self.registration_infos, self.rtss_infos = self._separate_infos(infos)

        if disregard_rtss:
            self.rtss_infos = tuple()

    @staticmethod
    def _separate_infos(infos: Tuple[DicomSeriesInfo]
                        ) -> Tuple[Tuple[DicomSeriesImageInfo],
                                   Tuple[DicomSeriesRegistrationInfo],
                                   Tuple[DicomSeriesRTStructureSetInfo]]:
        """Separates the DicomSeriesInfos into the respective groups.

        Args:
            infos (Tuple[DicomSeriesInfo]):  The DicomSeriesInfos to separate into groups.

        Returns:
            Tuple[Tuple[DicomSeriesImageInfo],
            Tuple[DicomSeriesRegistrationInfo],
            Tuple[DicomSeriesRTStructureSetInfo]]: The separated DicomSeriesInfos.
        """
        image_infos = []
        registration_infos = []
        rtss_infos = []

        for info in infos:
            if isinstance(info, DicomSeriesImageInfo):
                image_infos.append(info)

            elif isinstance(info, DicomSeriesRegistrationInfo):
                registration_infos.append(info)

            elif isinstance(info, DicomSeriesRTStructureSetInfo):
                rtss_infos.append(info)

            else:
                raise Exception('Unknown DicomSeriesInfo type recognized!')

        return tuple(image_infos), tuple(registration_infos), tuple(rtss_infos)

    def _validate_patient_identification(self) -> bool:
        """Validates the patient identification using all available DicomSeriesInfo.

        Returns:
            bool: True if all DicomSeriesInfo entries belong to the same patient. Otherwise False.
        """
        if not self.infos:
            return False

        results = []

        patient_name = self.infos[0].patient_name
        patient_id = self.infos[0].patient_id

        for info in self.infos:
            criteria = (patient_name == info.patient_name,
                        patient_id == info.patient_id)

            if all(criteria):
                results.append(True)
            else:
                results.append(False)

        if all(results):
            return True

        return False

    def _validate_registrations(self) -> bool:
        """Validates if for all DicomSeriesRegistrationInfos a matching DicomSeriesImageInfo exists.

        Returns:
            bool: True if for all DicomSeriesRegistrationInfos a matching DicomSeriesImageInfo exists, otherwise False.
        """

        def is_series_instance_uid_in_image_infos(series_instance_uids: List[str],
                                                  image_infos: Tuple[DicomSeriesImageInfo]
                                                  ) -> List[bool]:
            result = []
            for series_instance_uid in series_instance_uids:
                contained = False

                for image_info in image_infos:
                    if series_instance_uid == image_info.series_instance_uid:
                        contained = True
                        result.append(True)
                        break

                if not contained:
                    result.append(False)
            return result

        if not self.infos:
            return False

        if not self.registration_infos:
            return True

        series_uids_ident = []
        series_uids_trans = []

        for registration_info in self.registration_infos:

            if not registration_info.is_updated:
                registration_info.update()

            series_uids_ident.append(registration_info.referenced_series_instance_uid_identity)
            series_uids_trans.append(registration_info.referenced_series_instance_uid_transform)

        results_ident = is_series_instance_uid_in_image_infos(series_uids_ident, self.image_infos)
        results_trans = is_series_instance_uid_in_image_infos(series_uids_trans, self.image_infos)

        if all(results_ident) and all(results_trans):
            return True

        return False

    def convert(self) -> Subject:
        """Converts the specified DicomSeriesInfo into a Subject.

        Returns:
            Subject: The Subject as defined by the provided DicomSeriesInfo.
        """
        if self.validate_info:
            criteria = (self._validate_patient_identification(),
                        self._validate_registrations())

            if not all(criteria):
                raise ValueError('An error occurred during the subject level validation!')

        image_converter = DicomSeriesImageConverter(self.image_infos, self.registration_infos)
        intensity_images = image_converter.convert()

        subject = Subject(self.infos[0].patient_name)
        subject.add_images(intensity_images, force=True)

        if self.rtss_infos:
            rtss_converter = DicomSeriesRTStructureSetConverter(self.rtss_infos, self.image_infos,
                                                                self.registration_infos)
            segmentation_images = rtss_converter.convert()
            subject.add_images(segmentation_images, force=True)

        return subject


class SubjectRTStructureSetConverter(Converter):
    """A class converting a subjects segmentation images into an RT Structure Set."""

    def __init__(self,
                 subject: Subject,
                 infos: Tuple[DicomSeriesInfo],
                 reference_modality: Modality = Modality.T1c
                 ) -> None:
        super().__init__()

        assert subject.segmentation_images, 'The subject must contain segmentation images!'

        self.subject = subject

        assert infos, 'There must be infos provided for the conversion!'

        image_infos = []
        for info in infos:
            if isinstance(info, DicomSeriesImageInfo):
                if info.modality == reference_modality:
                    image_infos.append(info)

        assert image_infos, 'There must be image infos in the provided infos!'

        assert len(image_infos) == 1, 'There are multiple image infos fitting the reference modality!'

        self.image_info = image_infos[0]

        self.reference_modality = reference_modality

    def convert(self) -> Dataset:
        images = []
        label_names = []

        for segmentation in self.subject.segmentation_images:
            images.append(segmentation.get_image(as_sitk=True))
            label_names.append(segmentation.get_organ(as_str=True))

        image_datasets = load_datasets(self.image_info.path)

        converter = ImageToRTSSConverter(tuple(images), image_datasets, tuple(label_names), None)
        rtss = converter.convert()

        return rtss
