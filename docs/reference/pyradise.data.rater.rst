Rater Module
============
Module: :mod:`pyradise.data.rater`

.. module:: pyradise.data.rater
    :noindex:

General
-------

The :mod:`~pyradise.data.rater` module provides the functionality to manage information about the expert :class:`Rater`
who generated the segmentation on a certain :class:`~pyradise.data.image.SegmentationImage`. The :class:`Rater` can
take any name such that it can be used to identify a human expert as well as an auto-segmentation algorithm
(e.g. a deep learning model).

Examples of different rater naming:

.. code-block:: python

        from pyradise.data.rater import Rater

        rater = Rater("John Doe")
        rater = Rater("John Doe", "JD")
        rater = Rater("Robust Auto-Segmentation Algorithm")
        rater = Rater("Robust Auto-Segmentation Algorithm", "RASA")
        rater = Rater("Segmentation Algorithm v0.1")
        rater = Rater("Segmentation Algorithm v0.1", "SA-V01")

Class Overview
--------------

The following class is provided by the :mod:`~pyradise.data.rater` module:

+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| Class                            | Description                                                                                                                                               |
+==================================+===========================================================================================================================================================+
| :class:`Rater`                   | Class to identify the human expert or the auto-segmentation algorithm that generated a certain :class:`~pyradise.data.image.SegmentationImage` instance.  |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+

Details
-------

.. automodule:: pyradise.data.rater
    :show-inheritance:
    :members:
    :inherited-members:
