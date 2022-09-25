Inference Module
================
Module: :mod:`pyradise.process.inference`

.. module:: pyradise.process.inference
    :noindex:

General
-------

The inference module provides a prototype implementation of a DL-framework agnostic and filter-based inference class
(i.e. :class:`~pyradise.process.inference.InferenceFilter`) which is left for implementation to the user such that
the installation of a DL-framework is not required when installing PyRaDiSe.

Class Overview
--------------

The following classes are provided by the :mod:`~pyradise.process.inference` module:

+--------------------------------------------+--------------------------------------------------------------+
| Class                                      | Description                                                  |
+============================================+==============================================================+
| :class:`InferenceFilterParams`              | Parameterization class for the :class:`InferenceFilter`.    |
+--------------------------------------------+--------------------------------------------------------------+
| :class:`InferenceFilter`                    | Filter prototype for deep learning model inference.         |
+--------------------------------------------+--------------------------------------------------------------+

Details
-------

.. automodule:: pyradise.process.inference
    :members:
    :show-inheritance:
