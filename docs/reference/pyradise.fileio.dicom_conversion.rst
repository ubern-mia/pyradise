.. _dicom_conversion_module:

DICOM Conversion Module
=======================
Module: :mod:`pyradise.fileio.dicom_conversion`

.. module:: pyradise.fileio.dicom_conversion
    :noindex:

General
-------

The :mod:`~pyradise.fileio.dicom_conversion` module provides functionality to convert from and to DICOM data. This
functionality is especially useful if working in the clinical context where the input and / or output data is provided
in the DICOM format.

Class Overview
--------------

The following :class:`Converter` classes are provided:

+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| Class                                  | Description                                                                                                                                               |
+========================================+===========================================================================================================================================================+
| :class:`Converter`                     | Base class for all :class:`Converter` classes.                                                                                                            |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`DicomImageSeriesConverter`     | A :class:`Converter` to convert DICOM image files to :class:`~pyradise.data.image.IntensityImage` s (incl. registration if provided).                     |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`DicomRTSSSeriesConverter`      | A :class:`Converter` to convert DICOM-RTSS files to :class:`~pyradise.data.image.SegmentationImage` s (incl. registration if provided).                   |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`SubjectToRTSSConverter`        | A :class:`Converter` to convert a :class:`~pyradise.data.subject.Subject` to a DICOM-RTSS :class:`~pydicom.dataset.Dataset`.                              |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`RTSSToSegmentConverter`        | A low level :class:`Converter` to convert a DICOM-RTSS :class:`~pydicom.dataset.Dataset` to :class:`~pyradise.data.image.SegmentationImage` s.            |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`SegmentToRTSSConverter`        | A low level :class:`Converter` to convert :class:`~pyradise.data.image.SegmentationImage` s to a DICOM-RTSS :class:`~pydicom.dataset.Dataset`.            |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`RTSSMetaData`                  | A class defining the DICOM-RTSS metadata (used in combination with :class:`SubjectToRTSSConverter` and :class:`SegmentToRTSSConverter`).                  |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+

Details
-------

.. automodule:: pyradise.fileio.dicom_conversion
    :members:
    :show-inheritance:
    :inherited-members:

