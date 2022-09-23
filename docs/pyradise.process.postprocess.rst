Post-Processing Module
======================
Module: :mod:`pyradise.process.postprocess`

.. module:: pyradise.process.postprocess
    :noindex:

The postprocess module includes a single component filter class (i.e.
:class:`~pyradise.process.postprocess.SingleConnectedComponentFilter`) which exclusively keeps the largest single
component on the provided :class:`~pyradise.data.image.SegmentationImage` s. Furthermore, this module provides a filter
for alphabetically sorting the subject-associated :class:`~pyradise.data.image.SegmentationImage` s via their
:class:`~pyradise.data.organ.Organ` name. This functionality is essential if the :class:`~pyradise.data.organ.Organ` s
in a subsequently generated DICOM-RTSS dataset must be sorted.

.. automodule:: pyradise.process.postprocess
    :members:
    :show-inheritance:
