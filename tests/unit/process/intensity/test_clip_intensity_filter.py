import pytest

from pyradise.data import Modality
from pyradise.process.intensity import ClipIntensityFilter, ClipIntensityFilterParams


def test__init__1():
    params = ClipIntensityFilterParams(0, 1, ("modality",))
    assert params.modalities == (Modality("modality"),)
    assert params.min_value == 0
    assert params.max_value == 1


def test__init__2():
    params = ClipIntensityFilterParams(0, 1, None)
    assert params.modalities is None
    assert params.min_value == 0
    assert params.max_value == 1


def test__init__3():
    params = ClipIntensityFilterParams(1, 0, ("modality",))
    assert params.modalities == (Modality("modality"),)
    assert params.min_value == 0
    assert params.max_value == 1


def test__init__4():
    with pytest.raises(ValueError):
        ClipIntensityFilterParams(1, 1, None)
