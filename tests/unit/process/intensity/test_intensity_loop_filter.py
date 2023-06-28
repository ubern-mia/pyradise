from pyradise.data import Modality
from pyradise.process.intensity import IntensityLoopFilter, IntensityLoopFilterParams


def test__init__1():
    params = IntensityLoopFilterParams(0, ("modality",))
    assert params.modalities == (Modality("modality"),)


def test__init__2():
    params = IntensityLoopFilterParams(None)
    assert params.modalities is None


def test__init__3():
    pass
