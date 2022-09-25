Intensity Module
================
Module: :mod:`pyradise.process.intensity`

.. module:: pyradise.process.intensity
    :noindex:

General
-------

The intensity module provides intensity-modifying filters, such as a
:class:`~pyradise.process.intensity.ZScoreNormFilter`. Furthermore, the intensity module
provides separate intensity filter base classes (i.e., :class:`~pyradise.process.intensity.IntensityFilter` and
:class:`~pyradise.process.intensity.IntensityLoopFilter`) to simplify implementation of new intensity-modifying filters
and to facilitate the user-driven implementation.

Class Overview
--------------

The following classes are provided by the :mod:`~pyradise.process.intensity` module:

+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| Class                                   | Description                                                                                                                |
+=========================================+============================================================================================================================+
| :class:`IntensityFilterParams`          | Parameterization base class for the :class:`IntensityFilter` base class.                                                   |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`IntensityFilter`                | Simplified base class for intensity filters.                                                                               |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`IntensityLoopFilterParams`      | Parameterization base class for the :class:`IntensityLoopFilter` base class.                                               |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`IntensityLoopFilter`            | Simplified base class for intensity filters that process the images via looping over a predefined axis.                    |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`ZScoreNormFilterParams`         | Parameterization class for the :class:`ZScoreNormFilter`.                                                                  |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`ZScoreNormFilter`               | Filter for z-score normalization of :class:`~pyradise.data.image.IntensityImage` instances.                                |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`ZeroOneNormFilterParams`        | Parameterization class for the :class:`ZeroOneNormFilter`.                                                                 |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`ZeroOneNormFilter`              | Filter for zero-one (1-0) normalization of :class:`~pyradise.data.image.IntensityImage` instances.                         |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`RescaleIntensityFilterParams`   | Parameterization class for the :class:`RescaleIntensityFilter`.                                                            |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`RescaleIntensityFilter`         | Filter to rescale intensity values of :class:`~pyradise.data.image.IntensityImage` instances.                              |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`ClipIntensityFilterParams`      | Parameterization class for the :class:`ClipIntensityFilter`.                                                               |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`ClipIntensityFilter`            | Filter to limit intensity values of :class:`~pyradise.data.image.IntensityImage` instances to defined limits.              |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`GaussianFilterParams`           | Parameterization class for the :class:`GaussianFilter`.                                                                    |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`GaussianFilter`                 | Filter for Gaussian blurring of :class:`~pyradise.data.image.IntensityImage` instances.                                    |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`MedianFilterParams`             | Parameterization class for the :class:`MedianFilter`.                                                                      |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`MedianFilter`                   | Filter for median blurring of :class:`~pyradise.data.image.IntensityImage` instances.                                      |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`LaplacianFilterParams`          | Parameterization class for the :class:`LaplacianFilter`.                                                                   |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`LaplacianFilter`                | Filter for Laplacian sharpening of :class:`~pyradise.data.image.IntensityImage` instances.                                 |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+

Details
-------

.. automodule:: pyradise.process.intensity
    :members:
    :show-inheritance:

