.. _series_info_module:

Series Information Module
=========================
Module: :mod:`pyradise.fileio.series_info`

.. module:: pyradise.fileio.series_info
    :noindex:

The :mod:`~pyradise.fileio.series_info` module provides functionality to hold and retrieve information about data
entities which are required during the loading procedure for :class:`~pyradise.data.subject.Subject` and
:class:`~pyradise.data.image.Image` creation. The base class of this module is the :class:`SeriesInfo` which is named
in accordance with the DICOM Standard because it manages information about a single data entity (e.g. a DICOM-RTSS) or
a series of associated data entities (e.g. a series of image slices). For each supported type of data this module
provides separate classes because the information required for loading varies between the different types of data.

The :class:`SeriesInfo` instances are typically generated automatically during crawling (see
:ref:`Crawling Module <crawling_module>`) and render the selection (see :ref:`Selection Module <selection_module>`)
of appropriate :class:`SeriesInfo` instances feasible before executing the time consuming loading procedure. This is
especially useful if the user wants to process specific data exclusively and wants to keep the computation time and
the memory footprint low. Afterwards, the :class:`SeriesInfo` instances are used to load the data (see
:ref:`Loading Module <loading_module>`) and to create the :class:`~pyradise.data.subject.Subject` instance.


The following :class:`SeriesInfo` classes are provided:

+--------------------------------------+---------------------------------------------------------------------------+
| Class                                | Description                                                               |
+======================================+===========================================================================+
| :class:`SeriesInfo`                  | Base class for all :class:`SeriesInfo` classes.                           |
+--------------------------------------+---------------------------------------------------------------------------+
| :class:`FileSeriesInfo`              | Base class for all discrete image file format :class:`SeriesInfo` classes.|
+--------------------------------------+---------------------------------------------------------------------------+
| :class:`DicomSeriesInfo`             | Base class for all DICOM format :class:`SeriesInfo` classes.              |
+--------------------------------------+---------------------------------------------------------------------------+
| :class:`IntensityFileSeriesInfo`     | :class:`SeriesInfo` class for discrete intensity image files.             |
+--------------------------------------+---------------------------------------------------------------------------+
| :class:`SegmentationFileSeriesInfo`  | :class:`SeriesInfo` class for discrete segmentation image files.          |
+--------------------------------------+---------------------------------------------------------------------------+
| :class:`DicomSeriesImageInfo`        | :class:`SeriesInfo` class for DICOM image files.                          |
+--------------------------------------+---------------------------------------------------------------------------+
| :class:`DicomSeriesRegistrationInfo` | :class:`SeriesInfo` class for DICOM registration files.                   |
+--------------------------------------+---------------------------------------------------------------------------+
| :class:`DicomSeriesRTSSInfo`         | :class:`SeriesInfo` class for DICOM-RTSS files.                           |
+--------------------------------------+---------------------------------------------------------------------------+

|

.. automodule:: pyradise.fileio.series_info
    :members:
    :show-inheritance:
    :inherited-members:

