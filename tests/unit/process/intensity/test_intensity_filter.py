from pyradise.data import Modality
from pyradise.process.intensity import IntensityFilter, IntensityFilterParams


def test__init__1():
    params = IntensityFilterParams(("modality",))
    assert params.modalities == (Modality("modality"),)


def test__init__2():
    params = IntensityFilterParams(None)
    assert params.modalities is None
