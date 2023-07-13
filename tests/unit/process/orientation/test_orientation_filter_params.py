import pytest

from pyradise.process.orientation import OrientationFilterParams, SpatialOrientation


def test__init__1():
    params = OrientationFilterParams("LPS")
    assert params.output_orientation == SpatialOrientation.LPS


def test__init__2():
    with pytest.raises(ValueError):
        OrientationFilterParams("LLL")


def test__init__3():
    params = OrientationFilterParams(SpatialOrientation.LAS)
    assert (
        params.output_orientation == OrientationFilterParams("LAS").output_orientation
    )
