Invertibility Module
====================
Module: :mod:`pyradise.process.invertibility`

.. module:: pyradise.process.invertibility
    :noindex:

General
-------

The invertibility module provides functionality to playback the :class:`~pyradise.process.base.Filter` s of a
processing pipeline, if they are invertible. The given feature is helpful if the processing modified the spatial
properties of the input data that need to be restored (e.g. registration of the input data to an atlas before
DL-model inference).

Class Overview
--------------

The following classes are provided by the :mod:`~pyradise.process.invertibility` module:

+--------------------------------------------+---------------------------------------------------------------------------------------------------------+
| Class                                      | Description                                                                                             |
+============================================+=========================================================================================================+
| :class:`PlaybackTransformTapeFilterParams` | Parameterization class for the :class:`PlaybackTransformTapeFilter`.                                    |
+--------------------------------------------+---------------------------------------------------------------------------------------------------------+
| :class:`PlaybackTransformTapeFilter`       | Filter to playback the filter operations applied to a :class:`~pyradise.data.subject.Subject` instance. |
+--------------------------------------------+---------------------------------------------------------------------------------------------------------+


Details
-------

.. automodule:: pyradise.process.invertibility
    :members:
    :show-inheritance:
