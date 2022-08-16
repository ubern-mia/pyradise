from abc import (
    ABC,
    abstractmethod)
from typing import (
    Any,
    Dict,
    List)
import os

from .filtering import FileFilter


class Crawler(ABC):
    """Abstract base class for crawling the file system."""

    def __init__(self,
                 path: str
                 ) -> None:
        """Constructs a crawler.

        Args:
            path (str): The path to a directory.
        """
        super().__init__()

        if not os.path.exists(path):
            raise NotADirectoryError(f'The path {path} is not existing!')

        if not os.path.isdir(path):
            raise NotADirectoryError(f'The path {path} is not a directory!')

        self.path = path

    @abstractmethod
    def execute(self) -> Any:
        """Executes the crawler.

        Returns:
            Any: The crawled data.
        """
        raise NotImplementedError()


class SubjectDirectoryCrawler(Crawler):
    """A class crawling a subject directory."""

    def __init__(self,
                 path: str,
                 filter_: FileFilter
                 ) -> None:
        """Constructs a SubjectDirectoryCrawler.

        Args:
            path (str): The path to the dataset base directory.
            filter_ (FileFilter): The file filter to apply.
        """
        super().__init__(path)
        self.filter_ = filter_

    def execute(self) -> Dict[str, List]:
        """Executes the crawling process.

        Returns:
            Dict[str, List]: The crawled directory data.
        """
        subject_folders = [entry for entry in os.scandir(self.path) if entry.is_dir()]

        info = {}

        for subject_folder in subject_folders:
            subject_info = []
            files = [file for file in os.scandir(subject_folder.path) if file.is_file()]

            for file in files:
                file_info = self.filter_.filter(file.path)

                if file_info is None:
                    continue

                subject_info.append(file_info)

            info.update({subject_folder.name: subject_info})

        return info
