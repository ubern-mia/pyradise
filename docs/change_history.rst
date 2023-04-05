.. module:: pyradise
    :noindex:

Change History
==============

0.2.2 (05.04.2023)
------------------

* Minor bugfix due to deprecated `np.float` and `np.int`.
* Reformatted code to comply with PEP8.


0.2.1 (30.01.2023)
------------------

* Added meme to README
* Minor bugfix on :class:`~pyradise.data.image.Image` class


0.2.0 (07.01.2023)
------------------

* Added support for converting empty segmentation masks to an RTSS.
* Added support for color selection in :class:`~pyradise.fileio.dicom_conversion.SubjectToRTSSConverter`.
* Added a data field to each :class:`~pyradise.data.image.Image` sub-class to store additional information.
* Added an additional mechanism to all :class:`~pyradise.process.base.Filter` sub-classes to raise warnings when a non-invertible filter's operation is inverted.
* Added a hole filling algorithm to the classes for loading DICOM-RTSS files to circumvent issues with unexpected small holes in the resulting segmentation masks.
* Extended the :class:`~pyradise.fileio.modality_config.ModalityConfiguration` class to support manual addition of modality configuration entries.
* General optimization of the converter classes.
* Optimized the writing of directory hierarchies to circumvent issues during copying of multi-level hierarchies.
* Relaxed exception handling in :class:`~pyradise.fileio.series_info.DicomSeriesRegistrationInfo` to allow for more flexible handling of non-compliant DICOM registration files.
* Fixed the crawling procedure to circumvent issues with single DICOM image series loading.
* Minor updates to the documentation.


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
