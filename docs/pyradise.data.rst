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

The main data representation object in PyRaDiSe is the :class:`Subject` which contain sequences of
:class:`IntensityImage` and :class:`SegmentationImage`. Both image types contain the image information
(i.e. :class:`itk.Image`) and a :class:`TransformTape` which records all transformations and render a playback of them
feasible. This ensures that the original physical properties of each image can be restored after processing.

**Intensity Image**

The intensity image contains in addition to the image information and the transform tape information about the
:class:`Modality`. In PyRaDiSe a :class:`Modality` provides information about the type of imaging modality or MR
sequence. In detail, the taxonomy is not perfect for different MR-sequences but provides a simple and easy handling.


**Segmentation Image**

The segmentation image contains in addition information about the :class:`Organ` segmented on the image and the
:class:`Rater` which generated the segmentation / contours. Thus, each :class:`SegmentationImage` typically contains a
binary segmentation of one :class:`Organ`. However, this paradigm is not enforced and will be broken upon the
combination of multiple segmentation masks into one :class:`SegmentationImage` when serializing combined segmentation
masks into one file.

The main concept of the :mod:`pyradise.data` package is illustrated in the figures below.

.. image:: _static/data_image_0.png
    :width: 600
    :align: center
    :alt: main concept conversion module

|

Subject Module
--------------------
Module: :mod:`pyradise.data.subject`

The :mod:`subject` module provides the :class:`Subject` which is the main data object in PyRaDiSe.

|

.. automodule:: pyradise.data.subject
    :show-inheritance:
    :members:


Image Module
--------------------
Module: :mod:`pyradise.data.image`

The :mod:`image` module provide the functionality for the :class:`IntensityImage` and :class:`SegmentationImage` used
in the :class:`Subject`. An :class:`IntensityImage` is an image with intensity values (e.g. images from an MR scan) and
a :class:`SegmentationImage` is a segmentation mask (e.g. a segmentation of an organ).

.. image:: _static/data_image_1.png
    :width: 600
    :align: center
    :alt: difference between intensity and segmentation image

*Figure: Examples of an intensity image and a segmentation image.*

|

.. automodule:: pyradise.data.image
    :show-inheritance:
    :members:

Taping Module
--------------------
Module: :mod:`pyradise.data.taping`

The :mod:`taping` module provides the functionality for the recording and playback of the transformations applied to images.

|

.. automodule:: pyradise.data.taping
    :show-inheritance:
    :members:


Modality Module
--------------------
Module: :mod:`pyradise.data.modality`

The :mod:`modality` module provides the functionality to manage information about the :class:`Modality` of a certain
:class:`IntensityImage`. In PyRaDiSe the taxonomy for identifying the imaging modality or the MR sequence is identical.
The name :class:`Modality` was a design choice for which we believe that it is easily understandable and is
taxonomically sufficient precise.

.. image:: _static/data_image_2.png
    :width: 800
    :align: center
    :alt: examples of different modalities

*Figure: Examples of different modalities.*

|

.. automodule:: pyradise.data.modality
    :show-inheritance:
    :members:


Rater Module
--------------------
Module: :mod:`pyradise.data.rater`

The :mod:`rater` module provides the functionality to manage information about the expert rater (called :class:`Rater`)
which generated the segmentation on a certain :class:`SegmentationImage`. The :class:`Rater` can be a human being which
generated the contours / segmentations manually or an auto-segmentation algorithm (e.g. a deep learning model).

|

.. automodule:: pyradise.data.rater
    :show-inheritance:
    :members:

Organ Module
--------------------
Module: :mod:`pyradise.data.organ`

The :mod:`organ` module provides the functionality to manage information about the :class:`Organ` of a certain :class:`SegmentationImage`.

|

.. automodule:: pyradise.data.organ
    :show-inheritance:
    :members:
