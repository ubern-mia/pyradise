from abc import (
    ABC,
    abstractmethod)
from typing import (
    Any,
    Dict,
    List)

import SimpleITK as sitk

from pyradise.data import (
    Subject,
    IntensityImage,
    SegmentationImage)
from .definitions import (
    PATH,
    MODALITY,
    ORGAN,
    RATER)


class SubjectLoader(ABC):
    """Abstract base class for a subject loader."""

    def __init__(self,
                 data_origin: Any
                 ) -> None:
        """Constructs a subject loader.

        Args:
            data_origin (Any): The origin of the data.
        """
        super().__init__()
        self.data_origin = data_origin

    @abstractmethod
    def load_by_subject_name(self,
                             subject_name: str,
                             *args,
                             **kwargs
                             ) -> Subject:
        """Loads a subject by its name.

        Args:
            subject_name (str): The subjects name.
            *args: n/a
            **kwargs: n/a

        Returns:
            Subject: The loaded subject.
        """
        raise NotImplementedError()

    @abstractmethod
    def load_by_subject_index(self,
                              subject_index: int,
                              *args,
                              **kwargs
                              ) -> Subject:
        """Loads a subject by its index.

        Args:
            subject_index (int): The subjects index.
            *args: n/a
            **kwargs: n/a

        Returns:
            Subject: The loaded subject.
        """
        raise NotImplementedError()


class IterableSubjectLoader(SubjectLoader):
    """Abstract base class for an iterable subject loader."""

    @abstractmethod
    def load_by_subject_name(self,
                             subject_name: str,
                             *args,
                             **kwargs
                             ) -> Subject:
        raise NotImplementedError()

    @abstractmethod
    def load_by_subject_index(self,
                              subject_index: int,
                              *args,
                              **kwargs
                              ) -> Subject:
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def __next__(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()


class IterableNiftiSubjectLoader(IterableSubjectLoader):
    """An iterable NIFTI subject loader."""

    def __init__(self,
                 data_origin: Dict[str, List]
                 ) -> None:
        """Constructs the iterable subject loader.

        Args:
            data_origin (Dict[str, List]): The data origin information from a crawler.
        """
        super().__init__(data_origin)
        self.num_subjects = len(data_origin.keys())
        self.subject_names = tuple(data_origin.keys())

        self.subject_index = 0

    def load_by_subject_name(self,
                             subject_name: str,
                             *args,
                             **kwargs
                             ) -> Subject:
        """Loads a subject by name.

        Args:
            subject_name (str): The name of the subject.
            *args: n/a
            **kwargs: n/a

        Returns:
            Subject: The loaded subject.
        """
        subject = Subject(subject_name)

        subject_info = self.data_origin.get(subject_name)

        for file_info in subject_info:
            if not file_info.get(PATH):
                continue

            raw_image = sitk.ReadImage(file_info.get(PATH))

            if file_info.get(MODALITY):
                image = IntensityImage(raw_image, file_info.get(MODALITY))
                subject.add_image(image)

            if file_info.get(RATER):
                rater = file_info.get(RATER)
            else:
                rater = None

            if file_info.get(ORGAN):
                image = SegmentationImage(raw_image, file_info.get(ORGAN), rater)
                subject.add_image(image)

        return subject

    def load_by_subject_index(self,
                              subject_index: int,
                              *args,
                              **kwargs
                              ) -> Subject:
        """Loads a subject by its index.

        Args:
            subject_index (int): The index of the subject to load.
            *args: n/a
            **kwargs: n/a

        Returns:
            Subject: The loaded subject.
        """
        subject_name = self.subject_names[subject_index]
        return self.load_by_subject_name(subject_name)

    def __iter__(self):
        self.subject_index = 0
        return self

    def __next__(self) -> Subject:
        if self.subject_index < self.num_subjects:
            subject = self.load_by_subject_index(self.subject_index)

            self.subject_index += 1

            return subject

        raise StopIteration

    def __len__(self) -> int:
        return self.num_subjects
