.. _selection_module:

Selection Module
================
Module: :mod:`pyradise.fileio.selection`

.. module:: pyradise.fileio.selection
    :noindex:

General
-------

The :mod:`~pyradise.fileio.selection` module provides extensible functionality to select appropriate
:class:`~pyradise.fileio.series_info.SeriesInfo` instances from a list such that unused data does not need to be
loaded. This is especially useful if more data is provided to your pipeline than there is actually needed.

In this module a :class:`SeriesInfoSelector` base class is provided and several :class:`SeriesInfoSelector`
implementations which can be used to select specific :class:`~pyradise.fileio.series_info.SeriesInfo` entries according
to the :class:`~pyradise.data.modality.Modality`, :class:`~pyradise.data.organ.Organ`, or
:class:`~pyradise.data.rater.Rater`. In addition, two :class:`SeriesInfoSelector` implementations are provided to
exclude all :class:`~pyradise.fileio.series_info.DicomSeriesRegistrationInfo` entries and all
:class:`~pyradise.fileio.series_info.DicomSeriesRTSSInfo` entries such that no registration is applied to
the data during loading and that no DICOM-RTSS is loaded, respectively.

In order to extend the provided functionality, the :class:`SeriesInfoSelector` base class can be subclassed.

Class Overview
--------------

The following :class:`SeriesInfoSelector` classes are provided:

+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| Class                                  | Description                                                                                                                                               |
+========================================+===========================================================================================================================================================+
| :class:`SeriesInfoSelector`            | Base class for all :class:`SeriesInfoSelector` classes.                                                                                                   |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`ModalityInfoSelector`          | A :class:`SeriesInfoSelector` to keep :class:`~pyradise.fileio.series_info.SeriesInfo` entries with specific :class:`~pyradise.data.modality.Modality` s. |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`OrganInfoSelector`             | A :class:`SeriesInfoSelector` to keep :class:`~pyradise.fileio.series_info.SeriesInfo` entries with specific :class:`~pyradise.data.organ.Organ` s.       |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`RaterInfoSelector`             | A :class:`SeriesInfoSelector` to keep :class:`~pyradise.fileio.series_info.SeriesInfo` entries with specific :class:`~pyradise.data.rater.Rater` s.       |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`NoRegistrationInfoSelector`    | A :class:`SeriesInfoSelector` to exclude all :class:`~pyradise.fileio.series_info.DicomSeriesRegistrationInfo` entries.                                   |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`NoRTSSInfoSelector`            | A :class:`SeriesInfoSelector` to exclude all :class:`~pyradise.fileio.series_info.DicomSeriesRTSSInfo` entries.                                           |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+

Details
-------

.. automodule:: pyradise.fileio.selection
    :members:
    :show-inheritance:
    :inherited-members:

