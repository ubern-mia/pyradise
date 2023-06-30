import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

from pydicom.tag import Tag

from pyradise.data import Annotator, Modality, Organ
from pyradise.utils import is_dicom_file, load_dataset_tag

__all__ = [
    "Extractor",
    "ModalityExtractor",
    "SimpleModalityExtractor",
    "OrganExtractor",
    "SimpleOrganExtractor",
    "AnnotatorExtractor",
    "SimpleAnnotatorExtractor",
]


class Extractor(ABC):
    """An abstract base class for all extractors. An extractor extracts information about a file from its file path,
    the files content or from any other source of data in order to provide identification information
    (e.g. the imaging modality of a certain NIFTI file). Extractors can be used in combination with a
    :class:`~pyradise.fileio.crawling.Crawler` to extract the :class:`~pyradise.data.modality.Modality`,
    :class:`~pyradise.data.organ.Organ` or :class:`~pyradise.data.annotator.Annotator` instances for
    :class:`~pyradise.data.subject.Subject` construction.

    Typically, the user needs to implement the concrete extractor classes specific for the current task. This renders
    flexibility and allows for a wide range of use cases. However, the user can also use the provided implementations
    and examples to get started quickly.

    """

    @abstractmethod
    def extract(self, path: str) -> Any:
        """Extract information about the file at the specified path.

        Args:
            path (str): The path to the file for which information needs to be extracted.

        Returns:
            Any: The extracted information.
        """
        pass


class ModalityExtractor(Extractor):
    """A prototype class to extract the :class:`~pyradise.data.modality.Modality` from DICOM files and discrete image
    file paths. It must be implemented by the user and is intended to be used with the
    :class:`~pyradise.fileio.crawling.Crawler` types for DICOM and discrete image files. Thus, both abstract methods
    (i.e. :meth:`extract_from_dicom` and :meth:`extract_from_path`) need to be implemented. In case of working
    exclusively on DICOM or discrete image files, one extraction method may contain just a ``return None``.

    Important:
        If the file path does not specify an intensity image the extractor must return :data:`None`.

    Warnings:
        If :data:`return_default` is set to :data:`True` the :class:`ModalityExtractor` will return an enumerated
        default :class:`~pyradise.data.modality.Modality` for each file for which no modality could be extracted.
        This will have the effect that no error will be raised during loading. However, this functionality is intended
        to be used exlusively for experimenting and debugging purposes such that the user can load data without
        implementing a complete extractor. It's not recommended to use this feature for production purposes.
        Subsequent errors may arise.

    Notes:
        If using the :class:`ModalityExtractor` in combination with a :class:`~pyradise.fileio.crawling.Crawler` all
        paths to the discrete image files are provided sequentially to extract the
        :class:`~pyradise.data.modality.Modality`. In case of working with DICOM data the
        :class:`~pyradise.fileio.crawling.Crawler` will provide just one arbitrary file path to the
        :class:`ModalityExtractor`.

    Example:

        Example of a :class:`ModalityExtractor` implementation to identify detailed modalities:

        >>> from typing import (Any, Dict, Optional)
        >>>
        >>> from pyradise.fileio import (ModalityExtractor, Tag)
        >>> from pyradise.data import Modality
        >>>
        >>>
        >>> class ExampleModalityExtractor(ModalityExtractor):
        >>>
        >>>     @staticmethod
        >>>     def _get_mr_modality(ds_dict: Dict[str, Any]) -> Optional[Modality]:
        >>>         # check for different variants of attributes to get the sequence
        >>>         # identification
        >>>         scanning_sq = ds_dict.get('Scanning Sequence', {}).get('value', [])
        >>>         scanning_sq = [scanning_sq] if isinstance(scanning_sq, str) else scanning_sq
        >>>         contrast = ds_dict.get('Contrast/Bolus Agent', {}).get('value', '')
        >>>
        >>>         if all(val in scanning_sq for val in ('SE', 'IR')):
        >>>             return Modality('FLAIR')
        >>>         elif all(val in scanning_sq for val in ('GR', 'IR')) and len(contrast) > 0:
        >>>             return Modality('T1c')
        >>>         elif all(val in scanning_sq for val in ('GR', 'IR')) and len(contrast) == 0:
        >>>             return Modality('T1w')
        >>>         elif all(val == 'SE' for val in scanning_sq):
        >>>             return Modality('T2w')
        >>>         else:
        >>>             return None
        >>>
        >>>     def extract_from_dicom(self, path: str) -> Optional[Modality]:
        >>>         # extract the necessary attributes from the file
        >>>         tags = (Tag(0x0008, 0x0060),  # Modality
        >>>                 Tag(0x0018, 0x0010),  # ContrastBolusAgent
        >>>                 Tag(0x0018, 0x0020))  # ScanningSequence
        >>>         dataset_dict = self._load_dicom_attributes(tags, path)
        >>>
        >>>         # identify the modality
        >>>         extracted_modality = dataset_dict.get('Modality', {}).get('value', None)
        >>>         if extracted_modality == 'CT':
        >>>             return Modality('CT')
        >>>         elif extracted_modality == 'MR':
        >>>             return self._get_mr_modality(dataset_dict)
        >>>         else:
        >>>             return None
        >>>
        >>>     def extract_from_path(self, path: str) -> Optional[Modality]:
        >>>         # extract the necessary attributes from the file name
        >>>         file_name = os.path.basename(path)
        >>>         if 'T1c' in file_name:
        >>>             return Modality('T1c')
        >>>         elif 'T1w' in file_name:
        >>>             return Modality('T1w')
        >>>         elif 'T2w' in file_name:
        >>>             return Modality('T2w')
        >>>         elif 'FLAIR' in file_name:
        >>>             return Modality('FLAIR')
        >>>         elif 'CT' in file_name:
        >>>             return Modality('CT')
        >>>         else:
        >>>             return None

    Args:
        return_default (bool): Indicates if an enumerated default :class:`~pyradise.data.modality.Modality` should be
         returned if the extraction was not successful. Use this option exclusively for experimentation and debugging
         because it can cause severe damage (default: False).
    """

    modality_default_idx = 0
    default_modality_name = "UnknownModality"

    def __init__(self, return_default: bool = False) -> None:
        super().__init__()

        self.return_default = return_default

    @staticmethod
    def _load_dicom_attributes(tags: Union[Tuple[Tuple[int, int], ...], Tuple[Tag, ...]], path: str) -> Dict[str, Any]:
        """Load the DICOM attributes for the specified tags.

        Args:
            tags (Union[Tuple[Tuple[int, int], ...], Tuple[Tag, ...]]): The DICOM tags to extract the attributes for.
            path (str): The path to the DICOM file to extract the attributes from.

        Returns:
            Dict[str, Any]: The loaded DICOM attributes.
        """
        tags_ = [Tag(tag) for tag in tags]
        dataset = load_dataset_tag(path, tags_)

        data = {}
        for tag in tags_:
            item = dataset.get(tag, None)
            if item is not None:
                data[item.name] = {"name": item.name, "value": item.value, "vr": item.VR}

        return data

    def _get_next_default_modality_name(self) -> str:
        """Get the next enumerated modality name for unrecognized modalities.

        Returns:
            str: The next enumerated modality name.
        """
        name = self.default_modality_name + str(self.modality_default_idx)
        self.modality_default_idx += 1
        return name

    def is_enumerated_default_modality(self, modality: Optional[Union[Modality, str]]) -> bool:
        """Check if the specified modality is an enumerated default modality.

        Args:
            modality (Optional[Union[Modality, str]]): The modality to check.

        Returns:
            bool: True if the modality is an enumerated default modality, False otherwise.
        """
        if modality is None:
            return False

        if isinstance(modality, Modality):
            modality = modality.name
        return self.default_modality_name in modality

    @abstractmethod
    def extract_from_dicom(self, path: str) -> Optional[Modality]:
        """Extract the :class:`~pyradise.data.modality.Modality` from the DICOM file at the specified path.
        If the modality can not be detected :data:`None` must be returned.

        Notes:
            For your implementation you can load the DICOM file or specific DICOM attributes using the
            :meth:`load_dataset` or :meth:`load_dataset_tag` functions from the :mod:`pyradise.utils` module.
            For a detailed description of the DICOM attributes we refer to the `DICOM Standard
            <https://www.dicomstandard.org/>`_ and the `DICOM Standard Browser <https://dicom.innolitics.com/>`_.

        Args:
            path (str): The path to the DICOM file to extract the :class:`~pyradise.data.modality.Modality` from.

        Returns:
            Optional[Modality]: The extracted :class:`~pyradise.data.modality.Modality` or :data:`None`.
        """
        raise NotImplementedError()

    @abstractmethod
    def extract_from_path(self, path: str) -> Optional[Modality]:
        """Extract the :class:`~pyradise.data.modality.Modality` from the file path to a discrete image file or
        from another other data source. If the modality can not be detected :data:`None` must be returned.

        Args:
            path (str): The path to the file to extract the :class:`~pyradise.data.modality.Modality` for.

        Returns:
            Optional[Modality]: The extracted :class:`~pyradise.data.modality.Modality` or :data:`None`.
        """
        raise NotImplementedError()

    def extract(self, path: str) -> Optional[Modality]:
        """Extract the :class:`~pyradise.data.modality.Modality` for either a DICOM or a discrete medical image file.

        Args:
            path (str): The path to the file to extract the :class:`~pyradise.data.modality.Modality` for.

        Returns:
            Optional[Modality]: The extracted :class:`~pyradise.data.modality.Modality` or :data:`None`.
        """
        if is_dicom_file(path):
            modality = self.extract_from_dicom(path)
        else:
            modality = self.extract_from_path(path)

        if self.return_default and modality is None:
            return Modality(self._get_next_default_modality_name())

        return modality


class SimpleModalityExtractor(ModalityExtractor):
    """A simple :class:`ModalityExtractor` implementation that uses the 'Modality' attribute in the provided DICOM
    image or searches for a provided set of modality names (``modalities``) in the file name in case of a
    discrete image file to generate a :class:`~pyradise.data.modality.Modality` with the same name. If no match is
    found :data:`None` is returned.

    Args:
        modalities (Tuple[str, ...]): The possible modality names for the intensity files which will also
         be used to name the :class:`~pyradise.data.modality.Modality`.
        return_default (bool): Indicates if an enumerated default :class:`~pyradise.data.modality.Modality` should be
         returned if the extraction was not successfully. Use this option exclusively for experimentation and debugging
         because it can cause severe damage (default: False).

    """

    def __init__(
        self,
        modalities: Tuple[str, ...],
        return_default: bool = False,
    ) -> None:
        super().__init__(return_default)

        self.modalities = modalities

    def extract_from_path(self, path: str) -> Optional[Modality]:
        """Extract the :class:`~pyradise.data.modality.Modality` from the file name using the provided
        ``modalities``. If there is no match :data:`None` is returned.

        Args:
            path (str): The path to the file to extract the :class:`~pyradise.data.modality.Modality` for.

        Returns:
            Optional[Modality]: The extracted :class:`~pyradise.data.modality.Modality` or :data:`None`.
        """
        file_name = os.path.basename(path)

        for modality in self.modalities:
            if modality in file_name:
                return Modality(modality)

        return None

    def extract_from_dicom(self, path: str) -> Optional[Modality]:
        """Extract the DICOM attribute 'Modality' from the provided DICOM file. If no or an invalid 'Modality'
        attribute is found, :data:`None` is returned.

        Notes:
            This method exclusively extracts the following top-level modalities: CT, MR, PT, and US.
            For all other values of the DICOM 'Modality' attribute :data:`None` is returned.

        Args:
            path (str): The path to the DICOM file to extract the :class:`~pyradise.data.modality.Modality` from.

        Returns:
            Optional[Modality]: The extracted :class:`~pyradise.data.modality.Modality` or :data:`None`.
        """
        # extract the Modality attribute
        tags = (Tag(0x0008, 0x0060),)  # Modality
        dataset_dict = self._load_dicom_attributes(tags, path)

        # get the general modality
        extracted_modality = dataset_dict.get("Modality", {}).get("value", None)
        if extracted_modality in ("CT", "MR", "PT", "US"):
            return Modality(extracted_modality)
        else:
            return None


class OrganExtractor(Extractor):
    """A prototype class to extract an :class:`~pyradise.data.organ.Organ` from a discrete image file path. This class
    must be implemented by the user and is intended to be used with a :class:`~pyradise.fileio.crawling.Crawler` for
    discrete image formats.

    Important:
        If the file path does not specify a segmentation image the extractor must return :data:`None`.

    Example:

        Example of an :class:`OrganExtractor` implementation which takes search strings and associated organ names to
        extract an :class:`Organ` from a file path:

        >>> from typing import (Any, Dict, Optional)
        >>>
        >>> from pyradise.fileio import OrganExtractor
        >>> from pyradise.data import Organ
        >>>
        >>>
        >>> class ExampleOrganExtractor(OrganExtractor):
        >>>
        >>>     def __init__(self,
        >>>                  search_strings: Dict[str, str],
        >>>                  names: Tuple[str, ...]
        >>>                  ) -> None:
        >>>         super().__init__()
        >>>
        >>>         assert len(search_strings) == len(names), /
        >>>         f'Number of search strings ({len(search_strings)}) must match the ' \
        >>>         f'number of organ names ({len(names)})!'
        >>>
        >>>         self.search_strings = search_strings
        >>>         self.names = names
        >>>
        >>>     def extract(self, path: str) -> Optional[Organ]:
        >>>         file_name = os.path.basename(path)
        >>>
        >>>         for search_string, name in zip(self.search_strings, self.names):
        >>>             if search_string in file_name:
        >>>                 return Organ(name)
        >>>
        >>>         return None
    """

    def extract(self, path: str) -> Optional[Organ]:
        """Extract the :class:`~pyradise.data.organ.Organ` from the file path.

        Args:
            path (str): The path to the file to extract the :class:`~pyradise.data.organ.Organ` for.

        Returns:
            Optional[Organ]: The extracted :class:`~pyradise.data.organ.Organ` or :data:`None`.
        """
        raise NotImplementedError("The extract method needs to be adopted for the intended use case!")


class SimpleOrganExtractor(OrganExtractor):
    """A simple :class:`OrganExtractor` implementation that searches for a provided set of organ names
    (``organs``) in the file name and generates an :class:`~pyradise.data.organ.Organ` with the same name. If no
    match is found :data:`None` is returned.

    Args:
        organs (Tuple[str, ...]): The possible organ names which will also be used to name the output
         :class:`~pyradise.data.organ.Organ`.

    """

    def __init__(self, organs: Tuple[str, ...]) -> None:
        super().__init__()
        self.organs = organs

    def extract(self, path: str) -> Optional[Organ]:
        """Extract the :class:`~pyradise.data.organ.Organ` from the file name using the provided ``organs``. If
        no :class:`~pyradise.data.organ.Organ` can be extracted or the file does not contain a segmentation image
        :data:`None` is returned.

        Args:
            path (str): The path to the file to extract the :class:`~pyradise.data.organ.Organ` for.

        Returns:
            Optional[Organ]: The extracted :class:`~pyradise.data.organ.Organ` or :data:`None`.
        """
        file_name = os.path.basename(path)

        for organ in reversed(self.organs):
            if organ in file_name:
                return Organ(organ)

        return None


class AnnotatorExtractor(Extractor):
    """A prototype class to extract an :class:`~pyradise.data.annotator.Annotator` from a discrete image file path.
    This class must be implemented by the user and is intended to be used with a
    :class:`~pyradise.fileio.crawling.Crawler` for discrete image formats.

    Important:
        If the file path does not specify a segmentation image the extractor must return :data:`None`.

    Example:

        Example of an :class:`AnnotatorExtractor` implementation which takes search strings and associated annotator
        names to extract a :class:`~pyradise.data.annotator.Annotator` from a file path:

        >>> from typing import (Any, Dict, Optional)
        >>>
        >>> from pyradise.fileio import AnnotatorExtractor
        >>> from pyradise.data import Annotator
        >>>
        >>>
        >>> class ExampleAnnotatorExtractor(AnnotatorExtractor):
        >>>
        >>>     def __init__(self,
        >>>                  search_strings: Dict[str, str],
        >>>                  names: Tuple[str, ...]
        >>>                  ) -> None:
        >>>         super().__init__()
        >>>
        >>>         assert len(search_strings) == len(names), /
        >>>         f'Number of search strings ({len(search_strings)}) must match the' \
        >>>         f'number of annotator names ({len(names)})!'
        >>>
        >>>         self.search_strings = search_strings
        >>>         self.names = names
        >>>
        >>>     def extract(self, path: str) -> Optional[Annotator]:
        >>>         file_name = os.path.basename(path)
        >>>
        >>>         for search_string, name in zip(self.search_strings, self.names):
        >>>             if search_string in file_name:
        >>>                 return Annotator(name)
        >>>
        >>>         return None
    """

    def extract(self, path: str) -> Optional[Annotator]:
        """Extract the :class:`~pyradise.data.annotator.Annotator` from the file path.

        Args:
            path (str): The path to the file to extract the :class:`~pyradise.data.annotator.Annotator` for.

        Returns:
            Optional[Annotator]: The extracted :class:`~pyradise.data.annotator.Annotator` or :data:`None`.
        """
        raise NotImplementedError("The extract method needs to be adopted for the intended use case!")


class SimpleAnnotatorExtractor(AnnotatorExtractor):
    """A simple :class:`AnnotatorExtractor` implementation that searches for a provided set of annotator names
    (``annotators``) in the file name and generates a :class:`~pyradise.data.annotator.Annotator` with the same name.
    If no match is found :data:`None` is returned.

    Args:
        annotators (Tuple[str, ...]): The possible annotator names which will also be used to name the output
         :class:`Annotator`.

    """

    def __init__(self, annotators: Tuple[str, ...]) -> None:
        super().__init__()

        self.annotators = annotators

    def extract(self, path: str) -> Optional[Annotator]:
        """Extract the :class:`~pyradise.data.annotator.Annotator` from the file name using the provided ``annotators``.
        If no :class:`~pyradise.data.annotator.Annotator` can be extracted or the file does not contain a segmentation
        image :data:`None` is returned.

        Args:
            path (str): The path to the file to extract the :class:`~pyradise.data.annotator.Annotator` for.

        Returns:
            Optional[Annotator]: The extracted :class:`~pyradise.data.annotator.Annotator` or :data:`None`.
        """
        file_name = os.path.basename(path)

        for annotator in self.annotators:
            if annotator in file_name:
                return Annotator(annotator)

        return None
