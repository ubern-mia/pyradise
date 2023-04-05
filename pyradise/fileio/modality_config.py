import json
import os
from typing import List, NamedTuple, Optional, Tuple, Union

from pyradise.data import Modality
from pyradise.utils import is_file_and_exists

from .series_info import DicomSeriesImageInfo, DicomSeriesInfo

__all__ = ["ModalityConfiguration"]


ModalityConfigurationEntry = NamedTuple(
    "ModalityConfigurationEntry",
    SOPClassUID=str,
    StudyInstanceUID=str,
    SeriesInstanceUID=str,
    SeriesDescription=str,
    SeriesNumber=str,
    DICOM_Modality=str,
    Modality=Modality,
)


class ModalityConfiguration:
    """A class representation the mapping between the :class:`~pyradise.data.modality.Modality` and multiple DICOM
    images from one subject.

    The modality configuration us used for the identification of the detailed modalities belonging to multiple DICOM
    images from one subject. Typically, the modality configuration is stored in the subjects directory as a JSON file
    which can be modified manually. The :class:`ModalityConfiguration` class provides methods to load and write the
    modality configuration file. In addition, it provides functionality to retrieve the modality information from
    a :class:`~pyradise.fileio.series_info.DicomSeriesInfo` entries and to add the modality information to
    :class:`~pyradise.fileio.series_info.DicomSeriesInfo` entries.

    Typically, the :class:`ModalityConfiguration` class is used as part of a :class:`~pyradise.fileio.crawling.Crawler`
    which generates the modality configuration skeleton from a series of DICOM series. The modality configuration
    skeleton can then be stored on disk and manually modified. After modification, the modality configuration can be
    loaded and used to retrieve the modality information from a series of DICOM series.

    Examples:

        Generate the modality configuration skeleton from a series of DICOM series:

        >>> from pyradise.fileio import DatasetDicomCrawler
        >>>
        >>> def generate_skeleton(dataset_path: str) -> None:
        >>>     # Generate the modality configuration file skeleton by setting
        >>>     # write_modality_config = True
        >>>     crawler = DatasetDicomCrawler(dataset_path, write_modality_config=True)
        >>>     crawler.execute()
        >>>
        >>>
        >>> if __name__ == '__main__':
        >>>     generate_skeleton('path/to/dataset')

        Example of modality configuration file skeleton (named: modality_config.json):

        >>> [
        >>>    {
        >>>        "SOPClassUID": "1.2.840.10008.5.1.4.1.1.4",
        >>>        "StudyInstanceUID": "1.3.6.1.4.1.5962.99.1.1556635153761.6.0",
        >>>        "SeriesInstanceUID": "1.3.6.1.4.1.5962.99.1.1556635153761.239.0",
        >>>        "SeriesDescription": "t1_mpr_sag_we_p2_iso",
        >>>        "SeriesNumber": "7",
        >>>        "DICOM_Modality": "MR",
        >>>        "Modality": "MR"
        >>>   },
        >>>   {
        >>>        "SOPClassUID": "1.2.840.10008.5.1.4.1.1.2",
        >>>        "StudyInstanceUID": "1.3.6.1.4.1.5962.99.1.1557406273346.1015.0",
        >>>        "SeriesInstanceUID": "1.3.6.1.4.1.5962.99.1.1557406273346.1016.0",
        >>>        "SeriesDescription": "t2_fl_sag_p2_iso",
        >>>        "SeriesNumber": "2",
        >>>        "DICOM_Modality": "MR",
        >>>        "Modality": "MR"
        >>>    }
        >>> ]

        Example of modality configuration file (named: modality_config.json) content with filled "Modality" field:

        >>> [
        >>>    {
        >>>        "SOPClassUID": "1.2.840.10008.5.1.4.1.1.4",
        >>>        "StudyInstanceUID": "1.3.6.1.4.1.5962.99.1.1556635153761.6.0",
        >>>        "SeriesInstanceUID": "1.3.6.1.4.1.5962.99.1.1556635153761.239.0",
        >>>        "SeriesDescription": "t1_mpr_sag_we_p2_iso",
        >>>        "SeriesNumber": "7",
        >>>        "DICOM_Modality": "MR",
        >>>        "Modality": "T1c"
        >>>   },
        >>>   {
        >>>        "SOPClassUID": "1.2.840.10008.5.1.4.1.1.2",
        >>>        "StudyInstanceUID": "1.3.6.1.4.1.5962.99.1.1557406273346.1015.0",
        >>>        "SeriesInstanceUID": "1.3.6.1.4.1.5962.99.1.1557406273346.1016.0",
        >>>        "SeriesDescription": "t2_fl_sag_p2_iso",
        >>>        "SeriesNumber": "2",
        >>>        "DICOM_Modality": "MR",
        >>>        "Modality": "FLAIR"
        >>>    }
        >>> ]

        Load the data with modalities assigned according to the generated and modified modality configuration file:

        >>> from pyradise.fileio import DatasetDicomCrawler, SubjectLoader
        >>>
        >>> def load_data(dataset_path: str) -> None:
        >>>     # Create a crawler with write_modality_config = False to avoid
        >>>     # overwriting the existing file
        >>>     crawler = DatasetDicomCrawler(dataset_path,
        >>>                                   write_modality_config=False)
        >>>
        >>>     # Load the data with modalities assigned according to the modality
        >>>     # configuration file
        >>>     for subject_info in crawler:
        >>>         subject = SubjectLoader().load(subject_info)
        >>>         # Do something with the subject
        >>>         print(subject.get_name())
        >>>
        >>>
        >>> if __name__ == '__main__':
        >>>     load_data('path/to/dataset')

    """

    def __init__(self) -> None:
        super().__init__()
        self.configuration = []  # type: List[ModalityConfigurationEntry, ...]

    @classmethod
    def from_file(cls, path: str) -> "ModalityConfiguration":
        """Class method to load a modality configuration from file.

        Args:
            path (str): The path to the modality configuration file.

        Returns:
            ModalityConfiguration: The loaded modality configuration.
        """
        config = ModalityConfiguration()
        config._load_modality_file(path, False)
        return config

    def _load_modality_file(self, path: str, append: bool = False) -> None:
        """Load a modality file.

        Args:
            path (str): The file path to the modality file.
            append (bool): Indicates if the loaded modality information is appended or overwritten by the
             loading process.

        Returns:
            None
        """
        is_file_and_exists(path)

        if not append:
            self.configuration = []

        with open(path, "r") as file:
            data = json.load(file)

        for entry in data:
            series_instance_uid = entry.get("SeriesInstanceUID", "")
            series_description = entry.get("SeriesDescription", "")
            modality = Modality(entry.get("Modality", ""))

            if modality.is_default():
                print(
                    f"The modality for SeriesInstanceUID {series_instance_uid} and SeriesDescription "
                    f"{series_description} could not be detected from the ModalityConfiguration!"
                )

            # pylint: disable=not-callable
            config_entry = ModalityConfigurationEntry(
                SOPClassUID=entry.get("SOPClassUID", ""),
                StudyInstanceUID=entry.get("StudyInstanceUID", ""),
                SeriesInstanceUID=series_instance_uid,
                SeriesDescription=series_description,
                SeriesNumber=entry.get("SeriesNumber", ""),
                DICOM_Modality=entry.get("DICOM_Modality", ""),
                Modality=modality,
            )

            self.configuration.append(config_entry)

    @classmethod
    def from_dicom_series_info(cls, dicom_infos: Tuple[DicomSeriesInfo, ...]) -> "ModalityConfiguration":
        """Class method to generate a :class:`ModalityConfiguration` from a list of
        :class:`~pyradise.fileio.series_info.DicomSeriesInfo` entries.

        Args:
            dicom_infos (Tuple[DicomSeriesInfo, ...]): The info entries from which the modality information will be
             retrieved.

        Returns:
            ModalityConfiguration: The generated modality configuration.
        """
        config = ModalityConfiguration()
        config._generate_from_dicom_info(dicom_infos, False, False)
        return config

    def _generate_from_dicom_info(
        self, dicom_infos: Tuple[DicomSeriesInfo], update_infos: bool = False, append: bool = False
    ) -> None:
        """Generates the modality information entries for multiple DICOM image series.

        Args:
            dicom_infos (Tuple[DicomSeriesInfo]): The DicomSeriesInfo from which the modality information
             will be retrieved.
            update_infos (bool): Indicates if the provided DicomSeriesInfo entries needs to be updated before modality
             information retrieval.
            append (bool): Indicates if the retrieved modality information is appended or overwritten.

        Returns:
            None
        """
        if not append:
            self.configuration = []

        for info in dicom_infos:
            if update_infos:
                info.update()

            modality = info.get_modality() if isinstance(info, DicomSeriesImageInfo) else Modality.get_default()

            # pylint: disable=not-callable
            config_entry = ModalityConfigurationEntry(
                SOPClassUID=info.sop_class_uid,
                StudyInstanceUID=info.study_instance_uid,
                SeriesInstanceUID=info.series_instance_uid,
                SeriesDescription=info.series_description,
                SeriesNumber=info.series_number,
                DICOM_Modality=info.dicom_modality,
                Modality=modality,
            )

            self.configuration.append(config_entry)

    def to_file(self, path: str, override: bool = False) -> None:
        """Write the current modality configuration to a modality configuration file.

        Args:
            path (str): The file path for the modality file.
            override (bool): Indicates if an existing modality file should be overwritten.

        Returns:
            None
        """
        if not self.configuration:
            return

        if os.path.exists(path):
            if override:
                os.remove(path)
            else:
                raise FileExistsError(f"A modality configuration file already exists ({path}).")

        if not path.endswith(".json"):
            path += ".json"

        data = []
        for entry in self.configuration:
            dict_data = entry._asdict()
            dict_data["Modality"] = dict_data.get("Modality").name
            data.append(dict_data)

        with open(path, "w") as file:
            json.dump(data, file, indent=4)

    def add_modality_to_info(self, info: DicomSeriesImageInfo) -> None:
        """Add the modality information from the modality configuration to a
        :class:`~pyradise.fileio.series_info.DicomSeriesImageInfo` entry, if the information is available.

        Args:
            info (DicomSeriesImageInfo): The DicomSeriesImageInfo to which the modality should be added.

        Returns:
            None
        """
        modality, result = self._get_modality_for_series_instance_uid(info.series_instance_uid)

        if result:
            info.set_modality(modality)

    def add_modalities_to_info(self, infos: Tuple[DicomSeriesImageInfo]) -> None:
        """Add the modality information from the modality configuration to the provided DicomSeriesImageInfos if
        available.

        Args:
            infos (Tuple[DicomSeriesImageInfo]): The DicomSeriesImageInfos to which the modalities should be added.

        Returns:
            None
        """
        _ = tuple(map(self.add_modality_to_info, infos))

    def add_modality_entry(
        self,
        sop_class_uid: str,
        study_instance_uid: str,
        series_instance_uid: str,
        series_description: str,
        series_number: str,
        dicom_modality: str,
        modality: Union[Modality, str],
    ) -> None:
        """Add a modality configuration entry to the current modality configuration.

        Args:
            sop_class_uid (str): The SOP Class UID of the DICOM image series.
            study_instance_uid (str): The Study Instance UID of the DICOM image series.
            series_instance_uid (str): The Series Instance UID of the DICOM image series.
            series_description (str): The Series Description of the DICOM image series.
            series_number (str): The Series Number of the DICOM image series.
            dicom_modality (str): The DICOM Modality of the DICOM image series that is retrieved from the DICOM file.
            modality (Union[Modality, str]): The user-defined modality of the DICOM image series.

        Returns:
            None
        """

        if isinstance(modality, str):
            modality = Modality(modality)

        config_entry = ModalityConfigurationEntry(
            SOPClassUID=sop_class_uid,
            StudyInstanceUID=study_instance_uid,
            SeriesInstanceUID=series_instance_uid,
            SeriesDescription=series_description,
            SeriesNumber=series_number,
            DICOM_Modality=dicom_modality,
            Modality=modality,
        )
        self.configuration.append(config_entry)

    def _get_modality_for_series_instance_uid(
        self, series_instance_uid: str, force: bool = False
    ) -> Tuple[Optional[Modality], bool]:
        """Get the modality information for the specified SeriesInstanceUID if it is available in the configuration.

        Args:
            series_instance_uid (str): The SeriesInstanceUID for which the modality information should be returned.
            force (bool): If True, there will always be a returned modality information, otherwise a :class:`None`
             value may be returned.

        Returns:
            Tuple[Optional[Modality], bool]: The requested modality information or None and an indicator if the
             modality information could be retrieved from the configuration.
        """
        valid_entries = [entry for entry in self.configuration if entry.SeriesInstanceUID == series_instance_uid]

        if len(valid_entries) == 1:
            return valid_entries[0].Modality, True

        if force:
            return Modality.get_default(), False

        return None, False

    def has_default_modalities(self) -> bool:
        """Indicate if the modality configuration contains default modalities.

        Returns:
            bool: True if the modality configuration contains default modalities, otherwise False.
        """
        if not self.configuration:
            return True
        return any(entry.Modality.is_default() for entry in self.configuration)

    def has_duplicate_modalities(self) -> bool:
        """Indicate if the modality configuration contains duplicate :class:`~pyradise.data.modality.Modality` entries.

        Returns:
            bool: True if the modality configuration contains duplicate modality entries, otherwise False.
        """
        if not self.configuration:
            return False

        modalities = [entry.Modality for entry in self.configuration]
        return len(modalities) != len(set(modalities))
