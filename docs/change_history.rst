.. module:: pyradise
    :noindex:

Change History
==============

0.1.4 (TBD)
------------------

* Added support for converting empty segmentation masks to an RTSS.
* Optimized the writing of directory hierarchies to circumvent issues during copying of multi-level hierarchies.

0.1.3 (12.11.2022)
------------------

* Added a bugfix for DICOMDIR file parsing.
* Added a support for DICOM registration files that do not contain bi-directional references to DICOM image series.
* Optimized :class:`~pyradise.process.postprocess.SingleConnectedComponentFilter` for faster processing.
* Optimized :class:`~pyradise.process.base.Filter` and :class:`~pyradise.process.invertibility.PlaybackTransformTapeFilter` for faster inversion.
* Fixed and optimized :class:`~pyradise.fileio.dicom_conversion.SegmentToRTSSConverter3D`.
* Updated documentation.

0.1.2 (24.10.2022)
------------------

* Introduced integration tests for inference and conversion.
* Fixed an error caused by the ITK dependency on Microsoft Windows platforms in combination with Python 3.10 and 3.11.
* Optimized :class:`~pyradise.fileio.crawling.DatasetDicomCrawler` for fast single subject processing.
* Updated documentation.

0.1.1 (16.10.2022)
------------------

* Fixed an error caused by the ITK dependency on Microsoft Windows platforms.
* Removed PyPI version 0.1.0 to avoid errors on Microsoft Windows platforms.


0.1.0 (15.10.2022)
------------------

* Initial public release on PyPI.
