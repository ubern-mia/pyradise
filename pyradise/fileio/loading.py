from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

import SimpleITK as sitk

from pyradise.data import IntensityImage, SegmentationImage, Subject

from .dicom_conversion import (DicomImageSeriesConverter,
                               DicomRTSSSeriesConverter)
from .series_info import (DicomSeriesImageInfo, DicomSeriesRegistrationInfo,
                          DicomSeriesRTSSInfo, IntensityFileSeriesInfo,
                          SegmentationFileSeriesInfo, SeriesInfo)

__all__ = ["Loader", "ExplicitLoader", "SubjectLoader", "IterableSubjectLoader"]


class Loader(ABC):
    """An abstract base class for all :class:`Loader` classes. A :class:`Loader` class typically takes a sequence of
    :class:`~pyradise.fileio.series_info.SeriesInfo` entries and loads the data based on the information
    provided by the :class:`~pyradise.fileio.series_info.SeriesInfo` entries. The data is then returned as a
    :class:`~pyradise.data.subject.Subject` such that it can be used directly for further processing with for example
    the :mod:`~pyradise.process` package or the :mod:`~pyradise.fileio.writing` module.
    """

    @staticmethod
    def _extract_info_by_type(info: Sequence[SeriesInfo], type_: type) -> Tuple:
        """Extract all :class:`~pyradise.fileio.series_info.SeriesInfo` entries of the specified type from the
        provided sequence of :class:`~pyradise.fileio.series_info.SeriesInfo`.

        Args:
            info (Sequence[SeriesInfo]): The sequence of :class:`~pyradise.fileio.series_info.SeriesInfo` entries.
            type_ (type): The specific sub-type of the :class:`~pyradise.fileio.series_info.SeriesInfo` class to be
             extracted.

        Returns:
            Tuple[SeriesInfo]: The extracted :class:`~pyradise.fileio.series_info.SeriesInfo` entries.
        """
        return tuple(filter(lambda x: isinstance(x, type_), info))


class ExplicitLoader(Loader, ABC):
    """An abstract :class:`Loader` class that implements a :meth:`load` method such that multiple sets of
    :class:`~pyradise.fileio.series_info.SeriesInfo` entries can be loaded with the same :class:`Loader` instance."""

    @abstractmethod
    def load(self, info: Tuple[SeriesInfo, ...]) -> Subject:
        """Load the :class:`~pyradise.data.subject.Subject`.

        Args:
            info (Tuple[SeriesInfo, ...]): The :class:`~pyradise.fileio.series_info.SeriesInfo` entries to be loaded.

        Returns:
            Subject: The loaded :class:`~pyradise.data.subject.Subject`.
        """
        raise NotImplementedError()


class SubjectLoader(ExplicitLoader):
    """An :class:`ExplicitLoader` for loading a :class:`~pyradise.data.subject.Subject` based on its
    :class:`~pyradise.fileio.series_info.SeriesInfo` entries. This loader can load both DICOM data (i.e.
    :class:`~pyradise.fileio.series_info.DicomSeriesInfo`) and discrete image data (i.e.
    :class:`~pyradise.fileio.series_info.FileSeriesInfo`). The loader validates the provided
    :class:`~pyradise.fileio.series_info.SeriesInfo` entries before loading and raises appropriate errors if the
    information is not valid.

    Examples:

        Load and normalize NIFTI files and save the subject as NRRD files:

        >>> from argparse import ArgumentParser
        >>> from pyradise.fileio import (SubjectFileCrawler, SubjectLoader,
        >>>                              SubjectWriter, ImageFileFormat)
        >>> from pyradise.process import (ZScoreNormFilter,
        >>>                               ZScoreNormFilterParams)
        >>>
        >>>
        >>> def main(input_path: str, output_path: str, subject_name: str) -> None:
        >>>   # Crawl the input directory for compressed NIFTI files
        >>>   info = SubjectFileCrawler(input_path, subject_name, 'nii.gz').execute()
        >>>
        >>>   # Load the subject
        >>>   subject = SubjectLoader().load(info)
        >>>
        >>>   # Perform the normalization
        >>>   normalization_params = ZScoreNormFilterParams(loop_axis=1)
        >>>   normalization_filter = ZScoreNormFilter(normalization_params)
        >>>   subject = normalization_filter.execute(subject)
        >>>
        >>>   # Write the subject to the output directory
        >>>   writer = SubjectWriter(ImageFileFormat.NRRD)
        >>>   writer.write(output_path, subject, write_transforms=False)
        >>>
        >>>
        >>> if __name__ == '__main__':
        >>>   parser = ArgumentParser()
        >>>   parser.add_argument('input_path', type=str, help='The input directory.')
        >>>   parser.add_argument('output_path', type=str, help='The output directory.')
        >>>   parser.add_argument('subject_name', type=str, help='The name of the subject.')
        >>>   args = parser.parse_args()
        >>>
        >>>   main(args.input_path, args.output_path, args.subject_name)


        Load DICOM data and save the converted data as NIFTI files:

        >>> from argparse import ArgumentParser
        >>> from pyradise.fileio import SubjectDicomCrawler, SubjectLoader, SubjectWriter
        >>>
        >>>
        >>> def main(input_path: str, output_path: str) -> None:
        >>>   # Crawl the input directory for DICOM data
        >>>   # Note: We assume that the modality configuration file (modality_config.json)
        >>>   # is existing.
        >>>   info = SubjectDicomCrawler(input_path).execute()
        >>>
        >>>   # Load the subject
        >>>   subject = SubjectLoader().load(info)
        >>>
        >>>   # Write the subject to the output directory
        >>>   writer = SubjectWriter()
        >>>   writer.write(output_path, subject, write_transforms=False)
        >>>
        >>>
        >>> if __name__ == '__main__':
        >>>   parser = ArgumentParser()
        >>>   parser.add_argument('input_path', type=str, help='The input directory.')
        >>>   parser.add_argument('output_path', type=str, help='The output directory.')
        >>>   args = parser.parse_args()
        >>>
        >>>   main(args.input_path, args.output_path)

    Args:
        intensity_pixel_value_type (int): The pixel value type of the intensity imagesm when loading discrete image
         files (default: sitk.sitkFloat32).
        segmentation_pixel_value_type (int): The pixel value type of the segmentation images when loading discrete
         files (default: sitk.sitkUInt8).
        fill_hole_search_distance (int): The search distance for the hole filling algorithm. If the search distance is
         set to zero the hole filling algorithm is omitted. The search distance must be an odd number larger than 1
         (default: 0).

    """

    def __init__(
        self,
        intensity_pixel_value_type: int = sitk.sitkFloat32,
        segmentation_pixel_value_type: int = sitk.sitkUInt8,
        fill_hole_search_distance: int = 0,
    ) -> None:
        super().__init__()

        self.intensity_pixel_type = intensity_pixel_value_type
        self.segmentation_pixel_type = segmentation_pixel_value_type

        # store the fill hole search distance
        if fill_hole_search_distance == 0:
            self.fill_hole_distance = 0
        elif fill_hole_search_distance % 2 == 0:
            raise ValueError("The fill hole search distance must be an odd number.")
        elif fill_hole_search_distance == 1:
            raise ValueError("The fill hole search distance must be larger than 1.")
        else:
            self.fill_hole_distance = fill_hole_search_distance

    @staticmethod
    def _load_intensity_images(
        info: Tuple[IntensityFileSeriesInfo], pixel_value_type: sitk.sitkFloat32
    ) -> Tuple[IntensityImage]:
        """Load the :class:`~pyradise.data.image.IntensityImage` s.

        Args:
            info (Tuple[IntensityFileSeriesInfo]): The :class:`~pyradise.data.image.IntensityFileSeriesInfo` entries
             containing the file paths to the images.
            pixel_value_type (int): The pixel value type for the intensity images.

        Returns:
            Tuple[IntensityImage]: The loaded intensity :class:`~pyradise.data.image.IntensityImage` instances.
        """
        images = []
        for info_entry in info:
            image = sitk.ReadImage(info_entry.get_path()[0], pixel_value_type)
            images.append(IntensityImage(image, info_entry.get_modality()))

        return tuple(images)

    @staticmethod
    def _load_segmentation_images(
        info: Tuple[SegmentationFileSeriesInfo], pixel_value_type: sitk.sitkUInt8
    ) -> Tuple[SegmentationImage]:
        """Load the :class:`~pyradise.data.image.SegmentationImage` s.

        Args:
            info (Tuple[SegmentationFileSeriesInfo]): The
             :class:`~pyradise.fileio.series_info.SegmentationFileSeriesInfo` entries containing the file paths to
             the images.
            pixel_value_type (int): The pixel value type for the segmentation images.

        Returns:
            Tuple[SegmentationImage]: The loaded :class:`~pyradise.data.image.SegmentationImage` instances.
        """
        images = []
        for info_entry in info:
            image = sitk.ReadImage(info_entry.get_path()[0], pixel_value_type)
            images.append(SegmentationImage(image, info_entry.get_organ(), info_entry.get_annotator()))

        return tuple(images)

    @staticmethod
    def _validate_patient_identification(info: Tuple[SeriesInfo]) -> bool:
        """Validate the patient identification of the provided :class:`~pyradise.fileio.series_info.SeriesInfo` entries.

        Args:
            info (Tuple[SeriesInfo]): The :class:`~pyradise.fileio.series_info.SeriesInfo` entries to check.

        Returns:
            bool: True if the patient identification is valid for all info entries, otherwise False.
        """
        if not info:
            return False

        names = [entry.get_patient_name() for entry in info]
        ids = [entry.get_patient_id() for entry in info]

        return all(name == names[0] for name in names) and all(id_ == ids[0] for id_ in ids)

    @staticmethod
    def _validate_registration(
        reg_info: Tuple[DicomSeriesRegistrationInfo], image_info: Tuple[DicomSeriesImageInfo]
    ) -> bool:
        """Validate the ReferencedSeriesInstanceUIDs of the provided
        :class:`~pyradise.fileio.series_info.DicomSeriesRegistrationInfo` entries by checking if the referenced DICOM
        image data is provided.

        Args:
            reg_info (Tuple[DicomSeriesRegistrationInfo]): The
             :class:`~pyradise.fileio.series_info.DicomSeriesRegistrationInfo` entries to check.
            image_info (Tuple[DicomSeriesImageInfo]): The :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo`
             entries containing the referenced SeriesInstanceUIDs.

        Returns:
            bool: True if the image infos for all registration infos is available, otherwise False.

        """

        def is_image_info_available(instance_uids: List[str], image_info_: Tuple[DicomSeriesImageInfo]) -> bool:
            comparison = [[info.series_instance_uid == uid for info in image_info_] for uid in instance_uids]
            return all(any(comparison_) for comparison_ in comparison)

        if not reg_info:
            return True

        if not image_info:
            return False

        identity_uids = []
        transform_uids = []
        for reg_info_entry in reg_info:
            reg_info_entry.update() if not reg_info_entry.is_updated() else None

            identity = reg_info_entry.referenced_series_instance_uid_identity
            if identity != "":
                identity_uids.append(identity)

            transform = reg_info_entry.referenced_series_instance_uid_transform
            if transform != "":
                transform_uids.append(transform)

        if is_image_info_available(identity_uids, image_info) and is_image_info_available(transform_uids, image_info):
            return True

        return False

    @staticmethod
    def _validate_rtss_info(rtss_info: Tuple[DicomSeriesRTSSInfo], image_info: Tuple[DicomSeriesImageInfo]) -> bool:
        """Validate if all SeriesInstanceUIDs referenced in the DICOM-RTSSs are provided.

        Args:
            rtss_info (Tuple[DicomSeriesRTSSInfo]): The :class:`~pyradise.fileio.series_info.DicomSeriesRTSSInfo`
             entries to check.
            image_info (Tuple[DicomSeriesImageInfo]): The :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo`
             entries containing the SeriesInstanceUIDs.

        Returns:
            bool: True if the referenced image infos for all RTSS infos are available, otherwise False.
        """
        if not rtss_info:
            return True

        if not image_info:
            return False

        comparison = [
            any(info.series_instance_uid == rtss_info_entry.referenced_instance_uid for info in image_info)
            for rtss_info_entry in rtss_info
        ]

        return all(comparison)

    def load(self, info: Tuple[SeriesInfo, ...]) -> Subject:
        """Load a :class:`~pyradise.data.subject.Subject` from the provided
        :class:`~pyradise.fileio.series_info.SeriesInfo` entries.

        Args:
            info (Tuple[SeriesInfo, ...]): The :class:`~pyradise.fileio.series_info.SeriesInfo` entries containing the
             necessary information for loading the subject.

        Raises:
            ValueError: If ``info`` is an empty tuple.
            ValueError: If ``info`` is not a tuple of :class:`~pyradise.fileio.series_info.SeriesInfo` entries.
            ValueError: If the patient name and patient id of the provided
             :class:`~pyradise.fileio.series_info.SeriesInfo` entries are not equal.
            ValueError: If not all referenced :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo` entries are
             provided for registration.
            ValueError: If not all referenced :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo` entries are
             provided for RTSS loading.

        Returns:
            Subject: The loaded subject.

        """
        # check if the info entries have the correct structure
        if not info:
            raise ValueError("The provided info entries are empty.")

        if not all(isinstance(entry, SeriesInfo) for entry in info):
            raise ValueError(
                "The provided info entries are not of type SeriesInfo. "
                "Make sure to provide a tuple of SeriesInfo entries."
            )

        # separate the info entries
        dicom_image_info = self._extract_info_by_type(info, DicomSeriesImageInfo)
        dicom_reg_info = self._extract_info_by_type(info, DicomSeriesRegistrationInfo)
        dicom_rtss_info = self._extract_info_by_type(info, DicomSeriesRTSSInfo)
        intensity_image_info = self._extract_info_by_type(info, IntensityFileSeriesInfo)
        segmentation_image_info = self._extract_info_by_type(info, SegmentationFileSeriesInfo)

        # validate the info entries
        if not self._validate_patient_identification(info):
            raise ValueError("The patient identification (patient_name and patient_id) is not unique!")

        if not self._validate_registration(dicom_reg_info, dicom_image_info):
            raise ValueError("At least one referenced image in the registration is missing!")

        if not self._validate_rtss_info(dicom_rtss_info, dicom_image_info):
            raise ValueError("The referenced image in the RTSS is not available!")

        # create the subject
        if dicom_image_info:
            subject = Subject(dicom_image_info[0].get_patient_name())
        elif intensity_image_info:
            subject = Subject(intensity_image_info[0].get_patient_name())
        elif segmentation_image_info:
            subject = Subject(segmentation_image_info[0].get_patient_name())
        else:
            raise ValueError("Subject can not be constructed because a subject name is missing!")

        # load the images and add them to the subject
        if dicom_image_info:
            dicom_images = DicomImageSeriesConverter(dicom_image_info, dicom_reg_info).convert()
            subject.add_images(dicom_images)

        if dicom_rtss_info:
            dicom_segmentations = DicomRTSSSeriesConverter(
                dicom_rtss_info, dicom_image_info, dicom_reg_info, self.fill_hole_distance
            ).convert()
            subject.add_images(dicom_segmentations, force=True)

        intensity_images = self._load_intensity_images(intensity_image_info, self.intensity_pixel_type)
        segmentation_images = self._load_segmentation_images(segmentation_image_info, self.segmentation_pixel_type)
        subject.add_images(intensity_images + segmentation_images, force=True)

        return subject


class IterableSubjectLoader(Loader):
    """An :class:`Loader` for loading a sequence of :class:`~pyradise.data.subject.Subject` s based on their
    :class:`~pyradise.fileio.series_info.SeriesInfo` entries. This loader can load both DICOM data (i.e.
    :class:`~pyradise.fileio.series_info.DicomSeriesInfo`) and discrete image data (i.e.
    :class:`~pyradise.fileio.series_info.FileSeriesInfo`). The loader validates the provided
    :class:`~pyradise.fileio.series_info.SeriesInfo` entries before loading and raises appropriate errors if the
    information is not valid.

    Notes:
        For loading large DICOM dataset we recommend to use the :class:`SubjectLoader` instead because the antecedent
        crawling process can require a lot of computation time and memory.

    Raises:
        ValueError: If ``info`` is an empty tuple.
        ValueError: If ``info`` is not a tuple of tuples of :class:`~pyradise.fileio.series_info.SeriesInfo` entries.

    Args:
        info (Tuple[Tuple[SeriesInfo, ...], ...]): The :class:`~pyradise.fileio.series_info.SeriesInfo` entries for all
         subjects to load.
        intensity_pixel_value_type (int): The pixel value type of the intensity imagesm when loading discrete image
         files (default: sitk.sitkFloat32).
        segmentation_pixel_value_type (int): The pixel value type of the segmentation images when loading discrete
         files (default: sitk.sitkUInt8).
        fill_hole_search_distance (int): The search distance for the hole filling algorithm. If the search distance is
         set to zero the hole filling algorithm is omitted. The search distance must be an odd number larger than 1
         (default: 0).

    Examples:

        Load, normalize and save a NIFTI dataset with multiple subjects:

        >>> from argparse import ArgumentParser
        >>> from pyradise.fileio import (DatasetFileCrawler, IterableSubjectLoader,
        >>>                              SubjectWriter)
        >>> from pyradise.process import (ZScoreNormFilter,
        >>>                               ZScoreNormFilterParams)
        >>>
        >>>
        >>> def main(input_path: str, output_path: str) -> None:
        >>>     # Crawl the dataset info
        >>>     info = DatasetFileCrawler(input_path, '.nii.gz').execute()
        >>>
        >>>     # Construct the loader
        >>>     loader = IterableSubjectLoader(info)
        >>>
        >>>     # Construct the normalization filter
        >>>     normalization_params = ZScoreNormFilterParams(loop_axis=1)
        >>>     normalization_filter = ZScoreNormFilter(normalization_params)
        >>>
        >>>     # Construct the writer
        >>>     writer = SubjectWriter()
        >>>
        >>>     # Iteratively load the subjects
        >>>     for subject in loader:
        >>>         # Normalize the images
        >>>         subject = normalization_filter.execute(subject)
        >>>
        >>>         # Save the subject
        >>>         writer.write_to_subject_folder(output_path, subject, write_transforms=False)
        >>>
        >>>
        >>> if __name__ == '__main__':
        >>>     parser = ArgumentParser()
        >>>     parser.add_argument('--input_path', type=str,
        >>>                         help='The dataset input directory.')
        >>>     parser.add_argument('--output_path', type=str,
        >>>                         help='The dataset output directory.')
        >>>     args = parser.parse_args()
        >>>
        >>>     main(args.input_path, args.output_path)
    """

    def __init__(
        self,
        info: Tuple[Tuple[SeriesInfo, ...], ...],
        intensity_pixel_value_type: int = sitk.sitkFloat32,
        segmentation_pixel_value_type: int = sitk.sitkUInt8,
        fill_hole_search_distance: int = 0,
    ):
        super().__init__()

        if not info:
            raise ValueError("The provided infos are empty.")
        if not all(isinstance(entry, tuple) for entry in info):
            raise ValueError(
                "The provided first level info entries are not of type tuple. "
                "Make sure that the info is a tuple of tuples."
            )
        self.info = info

        self.intensity_pixel_type = intensity_pixel_value_type
        self.segmentation_pixel_type = segmentation_pixel_value_type

        # store the fill hole search distance
        if fill_hole_search_distance == 0:
            self.fill_hole_distance = 0
        elif fill_hole_search_distance % 2 == 0:
            raise ValueError("The fill hole search distance must be an odd number.")
        elif fill_hole_search_distance == 1:
            raise ValueError("The fill hole search distance must be larger than 1.")
        else:
            self.fill_hole_distance = fill_hole_search_distance

        self.current_idx = 0
        self.num_subjects = len(self.info)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> Subject:
        if self.current_idx < self.num_subjects:
            loader = SubjectLoader(self.intensity_pixel_type, self.segmentation_pixel_type, self.fill_hole_distance)
            subject = loader.load(self.info[self.current_idx])
            self.current_idx += 1
            return subject

        raise StopIteration()

    def __len__(self):
        return self.num_subjects
