.. role:: hidden
    :class: hidden-section

.. module:: pyradise.data

Data Package
============

The :mod:`pyradise.data` package contains the data model for PyRaDiSe which will be used during curation
(see :mod:`pyradise.curation`) and serialization (see :mod:`pyradise.serialization`). Furthermore, if PyRaDiSe is used
to load data from a discrete image file format (see :mod:`pyradise.loading`) PyRaDiSe will represent the data with
the subsequently described data model.

The main data representation object in PyRaDiSe is the :class:`Subject` which contain sequences of
:class:`IntensityImage` and :class:`SegmentationImage`. Both image types contain the image data
(i.e. :class:`itk.Image`) and a :class:`TransformTape` which records all transformations and renders a playback of them
feasible. This ensures that the original physical properties of each :class:`Image` can be restored after processing.

**Intensity Image**

In addition to the image data and the transform tape, an intensity image contains information about the
:class:`Modality`. In PyRaDiSe the modality provides information about the type of imaging modality or MR sequence.
However, this taxonomy is not perfect to discriminate between different MR sequences but in our opinion is a sufficient
approximation.


**Segmentation Image**

Additionally to the image data and the transform tape, a :class:`SegmentationImage` contains information about the
:class:`Organ` and the :class:`Rater` who generated the segmentations / contours. Due to the fact that a single
:class:`Organ` is associated with each :class:`SegmentationImage` the image data typically is a binary mask.
However, this paradigm is not enforced and the will be broken after the combination of multiple segmentation masks into
one :class:`SegmentationImage` as it is required to serialize multi-label segmentations into one discrete image file.

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
    :inherited-members:


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
    :inherited-members:

Taping Module
--------------------
Module: :mod:`pyradise.data.taping`

The :mod:`taping` module provides the functionality for the recording and playback of the transformations applied to
images.

|

.. automodule:: pyradise.data.taping
    :show-inheritance:
    :members:
    :inherited-members:


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
    :inherited-members:


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
    :inherited-members:

Organ Module
--------------------
Module: :mod:`pyradise.data.organ`

The :mod:`organ` module provides the functionality to manage information about the :class:`Organ` of a certain
:class:`SegmentationImage`.

|

.. automodule:: pyradise.data.organ
    :show-inheritance:
    :members:
    :inherited-members:
