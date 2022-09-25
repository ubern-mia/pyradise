Orientation Module
==================
Module: :mod:`pyradise.process.orientation`

.. module:: pyradise.process.orientation
    :noindex:

General
-------

The orientation module provides functionality to reorient images such that the anatomical orientation of the data is
standardized.

Class Overview
--------------

The following classes are provided by the :mod:`~pyradise.process.orientation` module:

+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| Class                                   | Description                                                                                                                |
+=========================================+============================================================================================================================+
| :class:`SpatialOrientation`             | Enum class for spatial image orientations (e.g. RAS, LAS).                                                                 |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`OrientationFilterParams`        | Parameterization class for the :class:`OrientationFilter`.                                                                 |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`OrientationFilter`              | Filter for reorienting :class:`~pyradise.data.image.Image` instances.                                                      |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+

Details
-------

.. automodule:: pyradise.process.orientation
    :members:
    :show-inheritance:


