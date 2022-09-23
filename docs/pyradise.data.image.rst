Image Module
============
Module: :mod:`pyradise.data.image`

.. module:: pyradise.data.image
    :noindex:

The image module provides the abstract :class:`Image` class and the implementations for the :class:`IntensityImage` and
:class:`SegmentationImage` classes. If a new image type is required for a certain task, the new image type should be
derived from the :class:`Image` class.

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
