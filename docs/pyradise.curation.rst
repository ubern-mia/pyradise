.. role:: hidden
    :class: hidden-section

.. module:: pyradise.curation

Curation Package
================

The :mod:`pyradise.curation` package provides functionality for pre-processing, deep learning model inference, and post-processing.

.. contents::
    :depth: 2
    :local:
    :backlinks: top

The :mod:`pyradise.curation` modules main objects are the :class:`Filter` class and the :class:`FilterPipeline` class which modify a :class:`Subject` and its ingredients (i.e. :class:`IntensityImage` and :class:`SegmentationImage`) and represent a pre-processing, model inference, or post-processing.

The main concept of :mod:`pyradise.curation` is illustrated in the figure below:

MAIN CONCEPT ILLUSTRATION


Filter Base Module
-------------------
Module: :mod:`pyradise.curation.base`

.. automodule:: pyradise.curation.base
    :show-inheritance:
    :members:


Normalization Module
--------------------
Module: :mod:`pyradise.curation.normalization`

.. automodule:: pyradise.curation.normalization
    :show-inheritance:
    :members:


Orientation Module
------------------
Module: :mod:`pyradise.curation.orientation`

.. automodule:: pyradise.curation.orientation
    :show-inheritance:
    :members:


Registration Module
-------------------
Module: :mod:`pyradise.curation.registration`

.. automodule:: pyradise.curation.registration
    :show-inheritance:
    :members:


Resampling Module
-----------------
Module: :mod:`pyradise.curation.resampling`

.. automodule:: pyradise.curation.resampling
    :show-inheritance:
    :members:


Segmentation Combination Module
-------------------------------
Module: :mod:`pyradise.curation.segmentation_combination`

.. automodule:: pyradise.curation.segmentation_combination
    :show-inheritance:
    :members:


Segmentation Post-Processing Module
-----------------------------------
Module: :mod:`pyradise.curation.segmentation_postprocessing`

.. automodule:: pyradise.curation.segmentation_postprocessing
    :show-inheritance:
    :members:


Transformation Module
---------------------
Module: :mod:`pyradise.curation.transformation`

.. automodule:: pyradise.curation.transformation
    :show-inheritance:
    :members:


Validation Module
-----------------
Module: :mod:`pyradise.curation.validation`

.. automodule:: pyradise.curation.validation
    :show-inheritance:
    :members:

