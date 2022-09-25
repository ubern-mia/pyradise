.. currentmodule:: pyradise.data.image

Image Module
============
Module: :mod:`pyradise.data.image`

.. module:: pyradise.data.image
    :noindex:

General
-------

The image module provides the abstract :class:`Image` base class and the implementations for the :class:`IntensityImage`
and :class:`SegmentationImage` classes.

.. figure:: ../_static/data_image_1.png
    :name: data_image_1
    :width: 80 %
    :align: center
    :alt: difference between intensity and segmentation image

    Examples of an intensity image and a segmentation image.

Class Overview
--------------

The following classes are provided by the :mod:`~pyradise.data.image` module:

+-----------------------------+---------------------------------------------------------------------------------------------------+
| Class                       | Description                                                                                       |
+=============================+===================================================================================================+
| :class:`Image`              | Base class for all :class:`Image` subclasses.                                                     |
+-----------------------------+---------------------------------------------------------------------------------------------------+
| :class:`IntensityImage`     | :class:`Image` class for intensity images (e.g. MR, CT, US images).                               |
+-----------------------------+---------------------------------------------------------------------------------------------------+
| :class:`SegmentationImage`  | :class:`Image` class for segmentation images (i.e. manual or automatically generated label maps). |
+-----------------------------+---------------------------------------------------------------------------------------------------+

Details
-------

.. automodule:: pyradise.data.image
    :show-inheritance:
    :members:
    :inherited-members:
