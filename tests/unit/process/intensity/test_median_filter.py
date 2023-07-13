from pyradise.data import Modality
from pyradise.process.intensity import MedianFilter, MedianFilterParams


def test__init__1():
    params = MedianFilterParams(1, ("modality",))
    assert params.modalities == (Modality("modality"),)
    assert params.radius == 1


def test__init__2():
    params = MedianFilterParams(1, None)
    assert params.modalities is None
    assert params.radius == 1
