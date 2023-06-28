import pytest

from pyradise.data import Modality
from pyradise.process.intensity import (
    RescaleIntensityFilter,
    RescaleIntensityFilterParams,
)


def test__init__1():
    params = RescaleIntensityFilterParams(0, 1, ("modality",))
    assert params.modalities == (Modality("modality"),)
    assert params.min_out == 0
    assert params.max_out == 1


def test__init__2():
    params = RescaleIntensityFilterParams(0, 1, None)
    assert params.modalities is None
    assert params.min_out == 0
    assert params.max_out == 1


def test__init__3():
    params = RescaleIntensityFilterParams(1, 0, ("modality",))
    assert params.modalities == (Modality("modality"),)
    assert params.min_out == 0
    assert params.max_out == 1


def test__init__4():
    with pytest.raises(ValueError):
        RescaleIntensityFilterParams(1, 1, None)
