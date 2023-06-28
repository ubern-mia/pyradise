import pytest

from pyradise.data import Modality
from pyradise.process.intensity import GaussianFilter, GaussianFilterParams


def test__init__1():
    params = GaussianFilterParams(1, 2, ("modality",))
    assert params.modalities == (Modality("modality"),)
    assert params.variance == 1
    assert params.kernel_size == 2


def test__init__2():
    params = GaussianFilterParams(1, 2, None)
    assert params.modalities is None
    assert params.variance == 1
    assert params.kernel_size == 2


def test__init__3():
    with pytest.raises(ValueError):
        GaussianFilterParams(0, 1, None)


def test__init__4():
    with pytest.raises(ValueError):
        GaussianFilterParams(1, 0, None)
