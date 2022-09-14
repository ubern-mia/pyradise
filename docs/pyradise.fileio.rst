.. role:: hidden
    :class: hidden-section

.. module:: pyradise.fileio
.. currentmodule:: pyradise.fileio


FileIO Package
==============

The :mod:`pyradise.fileio` package provides functionality for converting DICOM data into :class:`pyradise.data.Subject`..


Crawling Module
---------------
Module: :mod:`pyradise.fileio.crawling`

The :mod:`crawling` module provides functionality to search for loadable data in a filesystem hierarchy and to construct
intermediate information (i.e. :class:`SeriesInfo` and subclasses) for enabling the subject construction in the
data loading process. The advantage of this intermediate loading step is that unnecessary data does not need to be
loaded and can be skipped before the loading process. This is especially useful because working with imaging data is
often memory intensive. Furthermore, the loading of DICOM data may involve conversion steps (i.e. in case or available
registration files or DICOM-RTSS) which are time consuming and can be omitted using this intermediate step.

The :mod:`crawling` module provides separate crawlers for discrete image files and for DICOM data because we assume
that the user is either working with discrete image files or with DICOM data. For both input data types crawlers for
single subject directory or multi-subject hierarchies are provided. Single subject directory crawlers require that
the data of the subject is contained in a single folder including sub-folders. For multi-subject hierarchies, also
known commonly as a dataset, each subject must have its own directory at the top-most hierarchy level. For the
respective dataset crawlers a execute-at-once and an iterative version is provided. The iterative version is
especially useful if the user wants to process the data sequentially and wants to keep the memory footprint low.

IMAGE ABOUT THE HIERARCHIES

Due to the fact that not all information necessary for :class:`Subject` creation is available in a structured way
each crawler provides interfaces for information retrieving methods. The discrete image file crawlers (recognized by
the word :data:`File` in their name) provide interfaces for three types of :class:`Extractor` s
(i.e. :class:`ModalityExtractor`, :class:`OrganExtractor`, and :class:`RaterExtractor`) which need to be implemented
by the user for its specific task. Because DICOM data is more structured and the information about the rater and the
organs can be accessed directly in the DICOM-RTSS, DICOM crawlers provide just interfaces for retrieving the
modality information of DICOM image data. This is essential when working with data containing multiple
images from the same modality because DICOM provides just minimal standardized information such as for example MR for
all types of MR-sequences. This minimal information may not be sufficient in many radiotherapy applications because
working with different MR-sequences is common. Therefore, the user needs to have a way to distinguish between the
different uni-modal images. For the DICOM crawler this is done by either using a modality configuration file
(see :mod:`modality_config` for more details) which allows to build a persistent mapping between each
SeriesInstanceUID and the manually determined modality or by implementing a custom
:class:`ModalityExtractor` which extracts the modality information directly from the DICOM data. Both approaches
provide the same functionality but have distinctive advantages. While the modality configuration file approach may be
more convenient for recurrent work on the same data because it does not rely on metadata may containing ambiguities
the :class:`ModalityExtractor` approach may be better suited for building deployable solutions for which data can be
expected to posses the required metadata. The selection of the approach is up to the user.




.. automodule:: pyradise.fileio.crawling
    :members:
    :show-inheritance:

Modality Configuration Module
-----------------------------
Module: :mod:`pyradise.fileio.modality_config`

DESCRIPTION GOES HERE

.. automodule:: pyradise.fileio.modality_config
    :members:
    :show-inheritance:

Extraction Module
-----------------
Module: :mod:`pyradise.fileio.extraction`

The :mod:`extractor` module provides class prototypes, simple implementations, and examples
of extractors which are intended to be used to retrieve information from file paths or DICOM files to construct
:class:`Modality`, :class:`Organ`, and :class:`Rater` instances. Typically, extractors are used in combination with
a :class:`Crawler` to retrieve the necessary information for :class:`Subject` construction which happens during
loading.

If working with DICOM data, extractors provide an alternative to generating and maintaining modality configuration
files. This alternative is especially useful if the data is well organized and the necessary information can be
retrieved easily from the data. If the data varies and contains ambiguous content we recommend to use modality
configuration files instead because they are more flexible.

.. automodule:: pyradise.fileio.extraction
    :members:
    :show-inheritance:


Series Information Module
-------------------------
Module: :mod:`pyradise.fileio.series_info`

DESCRIPTION GOES HERE

.. automodule:: pyradise.fileio.series_info
    :members:
    :show-inheritance:


Selection Module
----------------
Module: :mod:`pyradise.fileio.selection`

DESCRIPTION GOES HERE

.. automodule:: pyradise.fileio.selection
    :members:
    :show-inheritance:


Loading Module
--------------
Module: :mod:`pyradise.fileio.loading`

DESCRIPTION GOES HERE

.. automodule:: pyradise.fileio.loading
    :members:
    :show-inheritance:


DICOM Conversion Module
-----------------------
Module: :mod:`pyradise.fileio.dicom_conversion`

DESCRIPTION GOES HERE

.. automodule:: pyradise.fileio.dicom_conversion
    :members:
    :show-inheritance:

Writing Module
--------------
Module: :mod:`pyradise.fileio.writing`

DESCRIPTION GOES HERE

.. automodule:: pyradise.fileio.writing
    :members:
    :show-inheritance:


Dataset Construction Module
---------------------------
Module: :mod:`pyradise.fileio.dataset_construction`

DESCRIPTION GOES HERE

.. automodule:: pyradise.fileio.dataset_construction
    :members:
    :show-inheritance:
