import os
from typing import Optional

from pyradise.curation.data import Subject


class DirectoryBuilder:
    """A class for building subject directories."""

    def __init__(self,
                 base_path: str,
                 dir_name: Optional[str] = None
                 ) -> None:
        """Constructs the DirectoryBuilder.

        Args:
            base_path (str): The path to the location where the dataset should be placed.
            dir_name (str): The directory name of the dataset.
        """
        super().__init__()

        if not os.path.exists(base_path):
            raise NotADirectoryError(f'The path {base_path} is not existing!')

        if not os.path.isdir(base_path):
            raise NotADirectoryError(f'The path {base_path} is not a directory!')

        self.base_path = base_path
        self.dir_name = dir_name

        if self.dir_name is not None:
            self.dataset_base_path = os.path.join(self.base_path, self.dir_name)
            os.mkdir(self.dataset_base_path)
        else:
            self.dataset_base_path = self.base_path

    def build_subject_directory(self,
                                subject: Subject
                                ) -> str:
        """Builds a new subject directory from a subject.

        Args:
            subject (Subject): The subject providing the information for the subject directory.

        Returns:
            str: The path to the subject directory.
        """
        subject_path = os.path.join(self.dataset_base_path, subject.name)
        os.mkdir(subject_path)
        return subject_path
