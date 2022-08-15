.. role:: hidden
    :class: hidden-section

.. module:: pyradise.data

Data Package
============

The :mod:`pyradise.data` package contains the data structures for PyRaDiSe.

.. contents::
    :depth: 2
    :local:
    :backlinks: top

The main data representation object in PyRaDiSe is the :class:`Subject` which contain sequences of :class:`IntensityImage` and :class:`SegmentationImage`.
Both image types contain the image information (i.e. :class:`itk.Image`) and a :class:`TransformTape` which records all transformations and render a playback of them feasible. This ensures that after processing images that the original orientation can be restored.


The main concept of the :mod:`pyradise.data` package is illustrated in the figures below.

.. image:: _static/data_image_0.png
    :width: 600
    :align: center
    :alt: main concept conversion module


Subject Module
--------------------
Module: :mod:`pyradise.data.subject`

The :mod:`subject` module provides the :class:`Subject` which is the main data object in PyRaDiSe.

.. automodule:: pyradise.data.subject
    :show-inheritance:
    :members:


Image Module
--------------------
Module: :mod:`pyradise.data.image`

The :mod:`image` module provide the functionality for the :class:`IntensityImage` and :class:`SegmentationImage` used in the :class:`Subject`.

.. automodule:: pyradise.data.image
    :show-inheritance:
    :members:

Taping Module
--------------------
Module: :mod:`pyradise.data.taping`

The :mod:`taping` module provides the functionality for the recording and playback of the transformations applied to images.

.. automodule:: pyradise.data.taping
    :show-inheritance:
    :members:


Modality Module
--------------------
Module: :mod:`pyradise.data.modality`

The :mod:`modality` module provides the functionality to manage information about the :class:`Modality` of a certain :class:`IntensityImage`.

.. automodule:: pyradise.data.modality
    :show-inheritance:
    :members:


Rater Module
--------------------
Module: :mod:`pyradise.data.rater`

The :mod:`rater` module provides the functionality to manage information about the :class:`Rater` of a certain :class:`SegmentationImage`.

.. automodule:: pyradise.data.rater
    :show-inheritance:
    :members:

Organ Module
--------------------
Module: :mod:`pyradise.data.organ`

The :mod:`organ` module provides the functionality to manage information about the :class:`Organ` of a certain :class:`SegmentationImage`.

.. automodule:: pyradise.data.organ
    :show-inheritance:
    :members:
