import os
from typing import (Optional, Tuple)

import pyradise.fileio as ps_fio
from pyradise.data import Modality


# pylint: disable=duplicate-code
class ModalityExtractorNifti(ps_fio.ModalityExtractor):
    """A modality extractor that always returns the same modality."""

    def __init__(self,
                 modalities: Tuple[str, ...],
                 identifier: str = 'img_',
                 return_default: bool = False) -> None:
        super().__init__(return_default)

        self.modalities = modalities
        self.identifier = identifier

    def extract_from_dicom(self, path: str) -> Optional[Modality]:
        return None

    def extract_from_path(self, path: str) -> Optional[Modality]:
        file_name = os.path.basename(path)

        if not file_name.startswith(self.identifier):
            return None

        for modality in self.modalities:
            if modality in file_name:
                return Modality(modality)
        return None
