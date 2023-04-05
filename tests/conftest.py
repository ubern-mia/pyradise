import os

import pytest

BASE_EXAMPLE_DATA = "D:/example_data/"


@pytest.fixture
def dicom_test_dataset_path() -> str:
    return os.path.join(BASE_EXAMPLE_DATA, "dicom_data/")


@pytest.fixture
def dicom_test_subj_path() -> str:
    return os.path.join(BASE_EXAMPLE_DATA, "dicom_data/VS-SEG-001")


@pytest.fixture
def nii_test_dataset_path() -> str:
    return os.path.join(BASE_EXAMPLE_DATA, "nifti_data/")


@pytest.fixture
def nii_test_subj_path() -> str:
    return os.path.join(BASE_EXAMPLE_DATA, "nifti_data/VS-SEG-001")


@pytest.fixture
def model_path() -> str:
    return os.path.join(BASE_EXAMPLE_DATA, "model/model.pth")
