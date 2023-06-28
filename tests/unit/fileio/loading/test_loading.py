import os

import SimpleITK as sitk

from pyradise.fileio.loading import IterableSubjectLoader, SubjectLoader


def test_subject_loader_load_intensity_images(img_series_dcm):
    pass
    # loader = SubjectLoader()
    # images = loader._load_intensity_images(info=os.listdir(img_series_dcm), pixel_value_type=sitk.sitkFloat32)
    #
    # assert subject.shape == (3, 3, 3)
