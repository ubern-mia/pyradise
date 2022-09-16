.. role:: any

.. automodule:: pyradise.fileio


FileIO Package
==============

The :mod:`pyradise.fileio` package provides functionality for loading, converting, and writing medical images in
discrete medical image formats and in the clinical DICOM format. In contrast to other medical image libraries such as
for example `SimpleITK <https://simpleitk.org/>`_, the :mod:`pyradise.fileio` package can process DICOM-RT
Structure Sets (DICOM-RTSS) which contain contours of delineated anatomical structures. Furthermore, the
:mod:`pyradise.fileio` package is able to load, assign and apply DICOM registrations such that the associated DICOM
images and DICOM-RTSS are registered to each other. In summary, this package provides the often missing piece of
functionality to work easily with clinical DICOM data in radiotherapy.

Due to the complex relations and dependencies between DICOM images, DICOM-RTSS, and DICOM registrations, the loading
process is not as straightforward as loading a single DICOM image. However, the :mod:`pyradise.fileio` package tries
to reduce the complexity of the loading process by providing simple and intuitive interfaces and mechanisms, automation,
and neat examples. To understand the loading process, it is recommended to follow the provided examples.

If the data successfully loaded and processed, the :mod:`pyradise.fileio` package provides functionality to write the
resulting data in a structured way to disk. This includes the writing of the data in various formats such as for example
NIFTI. In addition, the resulting data can also be converted into a DICOM-RTSS before writing it to disk.

To understand the functionality of the :mod:`pyradise.fileio` package, we recommend to read the following sections for
in the following orders:

**Data Loading**
    1. :ref:`Crawling Module <crawling_module>`
    2. :ref:`Modality Configuration Module <modality_config_module>`
    3. :ref:`Extraction Module <extraction_module>`
    4. :ref:`Loading Module <loading_module>`

**Data Writing**
    1. :ref:`Writing Module <writing_module>`
    2. :ref:`DICOM Conversion Module <dicom_conversion_module>` (for writing DICOM-RTSS)


.. _crawling_module:

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
    :inherited-members:


.. _modality_config_module:

Modality Configuration Module
-----------------------------
Module: :mod:`pyradise.fileio.modality_config`

One drawback of the DICOM standard is that it does only provide the imaging modality in a minimal way (i.e. MR, CT,
etc.) and little extra information about the acquisition parameters. If working with multiple images from the
same modality, this information may not sufficient to distinguish between the different images. Therefore,
the :mod:`modality_config` module provides functionality to build a persistent mapping between the DICOM images and
their modality. This mapping is stored in a JSON file and may need to be modified manually. If there are not multiple
images of the same DICOM modality, the modality configuration file is not required and can be omitted.

This mechanism of identifying the different images is especially useful when working recurrently with the same DICOM
data or if the DICOM attributes are guaranteed to be named consistently. If the DICOM attributes are not guaranteed
to be named consistently, the user may want to use the :class:`ModalityExtractor` mechanism to extract the modality
information directly from the DICOM data based on user defined rules or by accessing a different data source which
provides the necessary modality information. The extractor approach is more flexible but requires the user to implement
a custom set of rules to extract the modality information. The modality configuration approach is more convenient but
requires the user to manually modify the configuration files if there is more than one DICOM image of the same DICOM
modality.

If using the modality configuration approach, the modality configuration file skeleton filled with the DICOM modalities
can be generated automatically using the appropriate DICOM :class:`Crawler` from the :mod:`pyradise.fileio.crawling`
module. The generated modality configuration files are stored in the same directory as the DICOM files. After modifying
the generated modality configuration files manually or via an appropriate script, the data can be loaded using the
appropriate :class:`Loader` from the :mod:`pyradise.fileio.loading` module.


.. automodule:: pyradise.fileio.modality_config
    :members:
    :show-inheritance:
    :inherited-members:


.. _extraction_module:

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
    :inherited-members:

.. _series_info_module:

Series Information Module
-------------------------
Module: :mod:`pyradise.fileio.series_info`

DESCRIPTION GOES HERE

.. automodule:: pyradise.fileio.series_info
    :members:
    :show-inheritance:
    :inherited-members:


.. _selection_module:

Selection Module
----------------
Module: :mod:`pyradise.fileio.selection`

DESCRIPTION GOES HERE

.. automodule:: pyradise.fileio.selection
    :members:
    :show-inheritance:
    :inherited-members:


.. _loading_module:

Loading Module
--------------
Module: :mod:`pyradise.fileio.loading`

DESCRIPTION GOES HERE

.. automodule:: pyradise.fileio.loading
    :members:
    :show-inheritance:
    :inherited-members:


.. _dicom_conversion_module:

DICOM Conversion Module
-----------------------
Module: :mod:`pyradise.fileio.dicom_conversion`

DESCRIPTION GOES HERE

.. automodule:: pyradise.fileio.dicom_conversion
    :members:
    :show-inheritance:
    :inherited-members:


.. _writing_module:

Writing Module
--------------
Module: :mod:`pyradise.fileio.writing`

DESCRIPTION GOES HERE

.. automodule:: pyradise.fileio.writing
    :members:
    :show-inheritance:
    :inherited-members:


.. _dataset_construction_module:

Dataset Construction Module
---------------------------
Module: :mod:`pyradise.fileio.dataset_construction`

DESCRIPTION GOES HERE

.. automodule:: pyradise.fileio.dataset_construction
    :members:
    :show-inheritance:
    :inherited-members:
