.. _writing_module:

Writing Module
==============
Module: :mod:`pyradise.fileio.writing`

.. module:: pyradise.fileio.writing
    :noindex:

The :mod:`~pyradise.fileio.writing` module provides functionality to write :class:`~pyradise.data.subject.Subject`
instances in a structured way to disk. Furthermore, the module provides also writers for writing DICOM-RTSS
:class:`~pydicom.dataset.Dataset` s (see `pydicom Dataset <https://pydicom.github.io/pydicom/dev/reference/generated/pydicom.dataset.Dataset.html>`_).

The following writer classes are provided:

+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Class                                  | Description                                                                                                                                                                 |
+========================================+=============================================================================================================================================================================+
| :class:`SubjectWriter`                 | A writer class for the serialization of a :class:`~pyradise.data.subject.Subject` and its content.                                                                          |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`DirectorySubjectWriter`        | A writer class for the serialization of :class:`~pydicom.dataset.Dataset` s and copy functionality based on a source path.                                                  |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`DicomSeriesSubjectWriter`      | A writer class for the serialization of :class:`~pydicom.dataset.Dataset` s and copy functionality based on :class:`~pyradise.fileio.series_info.DicomSeriesInfo` instances.|
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

|

.. automodule:: pyradise.fileio.writing
    :members:
    :show-inheritance:
    :inherited-members:
