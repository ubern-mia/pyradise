from typing import (
    Tuple,
    List,
    NamedTuple,
    Optional)
import os
import json

from pyradise.data import Modality
from pyradise.utils import is_file_and_exists
from .series_info import (
    DicomSeriesInfo,
    DicomSeriesImageInfo)

__all__ = ['ModalityConfiguration']


ModalityConfigurationEntry = NamedTuple('ModalityConfigurationEntry',
                                        SOPClassUID=str,
                                        StudyInstanceUID=str,
                                        SeriesInstanceUID=str,
                                        SeriesDescription=str,
                                        SeriesNumber=int,
                                        DICOM_Modality=str,
                                        Modality=Modality)


class ModalityConfiguration:
    """Represents a configuration managing the :class:`Modality` handling."""

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

    def _load_modality_file(self,
                            path: str,
                            append: bool = False
                            ) -> None:
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

        with open(path, 'r') as file:
            data = json.load(file)

        for entry in data:
            series_instance_uid = entry.get('SeriesInstanceUID', '')
            series_description = entry.get('SeriesDescription', '')
            modality = Modality(entry.get('Modality', ''))

            if modality.is_default():
                print(f'The modality for SeriesInstanceUID {series_instance_uid} and SeriesDescription '
                      f'{series_description} could not be detected from the ModalityConfiguration!')

            # pylint: disable=not-callable
            config_entry = ModalityConfigurationEntry(SOPClassUID=entry.get('SOPClassUID', ''),
                                                      StudyInstanceUID=entry.get('StudyInstanceUID', ''),
                                                      SeriesInstanceUID=series_instance_uid,
                                                      SeriesDescription=series_description,
                                                      SeriesNumber=entry.get('SeriesNumber', ''),
                                                      DICOM_Modality=entry.get('DICOM_Modality', ''),
                                                      Modality=modality)

            self.configuration.append(config_entry)

    @classmethod
    def from_dicom_series_info(cls, dicom_infos: Tuple[DicomSeriesInfo, ...]) -> "ModalityConfiguration":
        """Generate a modality configuration from DICOM series infos.

        Args:
            dicom_infos (Tuple[DicomSeriesInfo, ...]): The DicomSeriesInfo from which the modality information will be
             retrieved.

        Returns:
            ModalityConfiguration: The generated modality configuration.
        """
        config = ModalityConfiguration()
        config._generate_from_dicom_info(dicom_infos, False, False)
        return config

    def _generate_from_dicom_info(self,
                                  dicom_infos: Tuple[DicomSeriesInfo],
                                  update_infos: bool = False,
                                  append: bool = False
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

            modality = info.get_modality() if isinstance(info, DicomSeriesImageInfo) else \
                Modality.get_default()

            # pylint: disable=not-callable
            config_entry = ModalityConfigurationEntry(SOPClassUID=info.sop_class_uid,
                                                      StudyInstanceUID=info.study_instance_uid,
                                                      SeriesInstanceUID=info.series_instance_uid,
                                                      SeriesDescription=info.series_description,
                                                      SeriesNumber=info.series_number,
                                                      DICOM_Modality=info.dicom_modality,
                                                      Modality=modality)

            self.configuration.append(config_entry)

    def to_file(self,
                path: str,
                override: bool = False
                ) -> None:
        """Write the current modality information to a modality configuration file.

        Args:
            path (str): The file path of the modality file.
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
                raise FileExistsError(f'The path {path} exists and overriding is forbidden!')

        if not path.endswith('.json'):
            path += '.json'

        data = []
        for entry in self.configuration:
            dict_data = entry._asdict()
            dict_data['Modality'] = dict_data.get('Modality').name
            data.append(dict_data)

        with open(path, 'w') as file:
            json.dump(data, file, indent=4)

    def add_modality_to_info(self, info: DicomSeriesImageInfo) -> None:
        """Add the modality information from the modality configuration to a DicomSeriesImageInfo if available.

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

    def _get_modality_for_series_instance_uid(self,
                                              series_instance_uid: str,
                                              force: bool = False
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
        """Indicate if the modality configuration contains duplicate :class:`Modality` entries.

        Returns:
            bool: True if the modality configuration contains :class:`Modality` entries, otherwise False.
        """
        if not self.configuration:
            return False

        modalities = [entry.Modality for entry in self.configuration]
        return len(modalities) != len(set(modalities))
