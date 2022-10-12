Annotator Module
================
Module: :mod:`pyradise.data.annotator`

.. module:: pyradise.data.annotator
    :noindex:

General
-------

The :mod:`~pyradise.data.annotator` module provides the functionality to manage information about the expert
:class:`Annotator` who generated the segmentation on a certain :class:`~pyradise.data.image.SegmentationImage`.
The :class:`Annotator` can take any name such that it can be used to identify a human expert as well as an
auto-segmentation algorithm (e.g. a deep learning model).

Examples of different annotator naming:

.. code-block:: python

        from pyradise.data.annotator import Annotator

        annotator = Annotator("John Doe")
        annotator = Annotator("John Doe", "JD")
        annotator = Annotator("Robust Auto-Segmentation Algorithm")
        annotator = Annotator("Robust Auto-Segmentation Algorithm", "RASA")
        annotator = Annotator("Segmentation Algorithm v0.1")
        annotator = Annotator("Segmentation Algorithm v0.1", "SA-V01")

Class Overview
--------------

The following class is provided by the :mod:`~pyradise.data.annotator` module:

+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| Class                            | Description                                                                                                                                               |
+==================================+===========================================================================================================================================================+
| :class:`Annotator`               | Class to identify the human expert or the auto-segmentation algorithm that generated a certain :class:`~pyradise.data.image.SegmentationImage` instance.  |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+

Details
-------

.. automodule:: pyradise.data.annotator
    :show-inheritance:
    :members:
    :inherited-members:
