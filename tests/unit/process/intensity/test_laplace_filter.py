from pyradise.data import Modality
from pyradise.process.intensity import LaplacianFilter, LaplacianFilterParams


def test__init__1():
    params = LaplacianFilterParams(("modality",))
    assert params.modalities == (Modality("modality"),)


def test__init__2():
    params = LaplacianFilterParams(None)
    assert params.modalities is None
