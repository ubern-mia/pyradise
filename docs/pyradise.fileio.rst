.. role:: any

.. automodule:: pyradise.fileio


FileIO Package
==============

The :mod:`~pyradise.fileio` package provides functionality for loading, converting, and writing medical images in
discrete medical image formats and in the clinical DICOM format. In contrast to other medical image libraries such as
for example `SimpleITK <https://simpleitk.org/>`_, the :mod:`~pyradise.fileio` package can process DICOM-RT
Structure Sets (DICOM-RTSS) which contain contours of delineated anatomical structures. Furthermore, the
:mod:`~pyradise.fileio` package is able to load, assign and apply DICOM registrations such that the associated DICOM
images and DICOM-RTSS are registered to each other. In summary, this package provides the often missing piece of
functionality to work easily with clinical DICOM data in radiotherapy.

Due to the complex relations and dependencies between DICOM images, DICOM-RTSS, and DICOM registrations, the loading
process is not as straightforward as loading a single DICOM image. However, the :mod:`~pyradise.fileio` package tries
to reduce the complexity of the loading process by providing simple and intuitive interfaces and mechanisms, automation,
and neat examples. To understand the loading process, it is recommended to follow the provided examples.

If the data successfully loaded and processed, the :mod:`~pyradise.fileio` package provides functionality to write the
resulting data in a structured way to disk. This includes the writing of the data in various formats such as for example
NIFTI. In addition, the resulting data can also be converted into a DICOM-RTSS before writing it to disk.

To understand the functionality of the :mod:`~pyradise.fileio` package, we recommend to read the following sections for
in the following orders:

**Data Loading**
    1. :ref:`Crawling Module <crawling_module>`
    2. :ref:`Modality Configuration Module <modality_config_module>` (only for DICOM data)
    3. :ref:`Extraction Module <extraction_module>`
    4. :ref:`Loading Module <loading_module>`

**Data Writing**
    1. :ref:`Writing Module <writing_module>`
    2. :ref:`DICOM Conversion Module <dicom_conversion_module>` (only for writing DICOM-RTSS)


.. _crawling_module:

Crawling Module
---------------
Module: :mod:`pyradise.fileio.crawling`

.. module:: pyradise.fileio.crawling
    :noindex:

The :mod:`~pyradise.fileio.crawling` module provides functionality to search for loadable data in a filesystem
hierarchy and to construct intermediate information (i.e. :class:`~pyradise.fileio.series_info.SeriesInfo` and
subclasses) for enabling the subject construction in the data loading process. The advantage of this intermediate
loading step is that unnecessary data does not need to be loaded and can be skipped before the loading process (see
:ref:`selection_module`). This is especially useful because working with imaging data is often memory intensive.
Furthermore, the loading of DICOM data may involve conversion steps (i.e. in case or available registration files
or DICOM-RTSS) which are time consuming and can be omitted using this intermediate step.

The :mod:`~pyradise.fileio.crawling` module provides separate crawlers for discrete image files and for DICOM data
because we assume that the user is either working with discrete image files or with DICOM data. For both input data
types crawlers for :ref:`single subject directory <crawling_ds_subject>` or
:ref:`multi-subject datasets <crawling_ds_dataset>` are provided. Single subject directory crawlers require that the
data of the subject is contained in a single folder including sub-folders. For multi-subject datasets each subject
must have its own directory at the top-most hierarchy level. For the dataset crawlers a execute-at-once and an
iterative approach is provided. The iterative approach is especially useful if the user wants to process the data
sequentially and wants to keep the memory footprint low. On the other hand, the execute-at-once approach may be useful
if the user wants to analyse the data in parallel.

Due to the fact that not all information necessary for :class:`~pyradise.data.subject.Subject` creation is available in
the file content the crawlers provide interfaces for information retrieval methods. The discrete image file
crawlers (recognized by the word :data:`File` in their name) provide interfaces for three types of
:class:`~pyradise.fileio.extraction.Extractor` s (i.e. :class:`~pyradise.fileio.extraction.ModalityExtractor`,
:class:`~pyradise.fileio.extraction.OrganExtractor`, and :class:`~pyradise.fileio.extraction.RaterExtractor`) which
need to be implemented by the user for its specific task. Typically, the :class:`~pyradise.fileio.extraction.Extractor`
s use parts of the file name to retrieve the necessary information (e.g. the modality from the file name or from a
lookup table). Because DICOM data is more structured than discrete file formats and the information
about the rater and the organs can be accessed directly in the DICOM-RTSS, DICOM crawlers provide just interfaces for
retrieving the modality information of DICOM image data. This is essential when working with subject data consisting of
multiple images from the same modality because DICOM provides just minimal information about the imaging modality such
as for example MR for all types of MR-sequences. This minimal information may not be sufficient in many radiotherapy
applications because working with different MR-sequences is common and a discrimination between the MR-sequences is
evident to feed the different MR-sequences in the correct order through a processing pipeline or a DL-model.
Thus, the user must have a mechanism to distinguish between the different uni-modal images. The DICOM crawlers provide
two separate mechanisms to retrieve detailed modality information. The first and prioritized approach is using a
modality configuration file (see :mod:`~pyradise.fileio.modality_config` for more details) which stores a persistent
mapping between the DICOM SeriesInstanceUID its modality. The skeleton of this file can be generated automatically with
the appropriate crawler and needs to be modified accordingly by the user. The second approach is using a user-defined
:class:`~pyradise.fileio.extraction.ModalityExtractor` which extracts the necessary modality details directly from the
DICOM file content. Both approaches provide the same functionality but have distinctive advantages. While the modality
configuration file approach may be more convenient for recurrent work on the same data the extractor approach may be
better suited for building deployable solutions for which the modality details can be retrieved rule-based. The
selection of the appropriate approach is up to the user.


The following :class:`Crawler` classes are provided by the :mod:`~pyradise.fileio.crawling` module:

+-----------------------------------------+---------------------------------------------------------------------------+
| Class                                   | Description                                                               |
+=========================================+===========================================================================+
| :class:`Crawler`                        | Base class for all :class:`Crawler` subclasses                            |
+-----------------------------------------+---------------------------------------------------------------------------+
| :class:`SubjectFileCrawler`             | Crawler class for discrete image files in a single subject directory      |
+-----------------------------------------+---------------------------------------------------------------------------+
| :class:`DatasetFileCrawler`             | Crawler class for discrete image files in a dataset directory             |
+-----------------------------------------+---------------------------------------------------------------------------+
| :class:`SubjectDicomCrawler`            | Crawler class for DICOM files in a single subject directory               |
+-----------------------------------------+---------------------------------------------------------------------------+
| :class:`DatasetDicomCrawler`            | Crawler class for DICOM files in a dataset directory                      |
+-----------------------------------------+---------------------------------------------------------------------------+

.. _crawling_ds_subject:

**Data Structure for Subject Crawlers**

.. code-block:: bash

        <subject_dir>
        ├── <file_0>
        ├── <file_1>
        └── ...

.. _crawling_ds_dataset:

**Data Structure for Dataset Crawlers**

.. code-block:: bash

        <dataset_dir>
        ├── <subject_0>
        │   ├── <file_0>
        │   ├── <file_1>
        │   └── ...
        ├── <subject_1>
        │   ├── <file_0>
        │   ├── <file_1>
        │   └── ...
        └── ...

|

.. automodule:: pyradise.fileio.crawling
    :members:
    :show-inheritance:
    :inherited-members:


.. _modality_config_module:

Modality Configuration Module
-----------------------------
Module: :mod:`pyradise.fileio.modality_config`

.. module:: pyradise.fileio.modality_config
    :noindex:

One drawback of the DICOM standard is that it does only provide the imaging modality in a minimal way (i.e. MR, CT,
etc.) and little extra information about the acquisition parameters. If working with multiple images from the
same modality, this information may not sufficient to distinguish between the different images. Therefore,
the :mod:`~pyradise.fileio.modality_config` module provides functionality to build a persistent mapping between the
DICOM images and their modality. This mapping is stored in a JSON file and may need to be modified manually. If there
are not multiple images of the same DICOM modality, the modality configuration file is not required and can be omitted.

This mechanism of identifying the different images is especially useful when working recurrently with the same DICOM
data or if the DICOM attributes are guaranteed to be named consistently. If the DICOM attributes are not guaranteed
to be named consistently, the user may want to use the :class:`~pyradise.fileio.extraction.ModalityExtractor` mechanism
to extract the modality information directly from the DICOM data based on user defined rules or by accessing a different
data source which provides the necessary modality information. The extractor approach is more flexible but requires the
user to implement a custom set of rules to extract the modality information. The modality configuration approach is more
convenient but requires the user to manually modify the configuration files if there is more than one DICOM image of the
same DICOM modality.

If using the modality configuration approach, the modality configuration file skeleton filled with the DICOM modalities
can be generated automatically using the appropriate DICOM :class:`~pyradise.fileio.crawling.Crawler`. The generated
modality configuration files are stored in the same directory as the DICOM files. After modifying the generated
modality configuration files manually or via an appropriate script, the data can be loaded using the appropriate
:class:`~pyradise.fileio.loading.Loader`.


.. automodule:: pyradise.fileio.modality_config
    :members:
    :show-inheritance:
    :inherited-members:


.. _extraction_module:

Extraction Module
-----------------
Module: :mod:`pyradise.fileio.extraction`

.. module:: pyradise.fileio.extraction
    :noindex:

The :mod:`~pyradise.fileio.extraction` module provides class prototypes, simple implementations, and examples
of extractors which are intended to be used to retrieve information from file paths or DICOM files to construct
:class:`~pyradise.data.modality.Modality`, :class:`~pyradise.data.organ.Organ`, and :class:`~pyradise.data.rater.Rater`
instances. Typically, extractors are used in combination with a :class:`~pyradise.fileio.crawling.Crawler` to retrieve
the necessary information for :class:`~pyradise.data.subject.Subject` construction which happens during
loading.

If working with DICOM data, extractors provide an alternative to generating and maintaining modality configuration
files. This alternative is especially useful if the data is well organized and the necessary information can be
retrieved easily from the data. If the data varies and contains ambiguous content we recommend to use modality
configuration files instead because they are more flexible.

The following abstract :class:`Extractor` classes are provided by the :mod:`~pyradise.fileio.extraction` module:

+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| Class                                   | Description                                                                                                                |
+=========================================+============================================================================================================================+
| :class:`Extractor`                      | Base class for all :class:`Extractor` subclasses                                                                           |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`ModalityExtractor`              | Prototype :class:`Extractor` for :class:`~pyradise.data.modality.Modality` extraction on discrete images and DICOM images. |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`OrganExtractor`                 | Prototype :class:`Extractor` for :class:`~pyradise.data.organ.Organ` extraction from discrete images.                      |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`RaterExtractor`                 | Prototype :class:`Extractor` for :class:`~pyradise.data.rater.Rater` extraction from discrete images.                      |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+

|

The following concrete :class:`Extractor` classes are provided by the :mod:`~pyradise.fileio.extraction` module:

+-----------------------------------------+---------------------------------------+
| Class                                   | Description                           |
+=========================================+=======================================+
| :class:`SimpleModalityExtractor`        | A simple :class:`ModalityExtractor`.  |
+-----------------------------------------+---------------------------------------+
| :class:`SimpleOrganExtractor`           | A simple :class:`OrganExtractor`.     |
+-----------------------------------------+---------------------------------------+
| :class:`SimpleRaterExtractor`           | A simple :class:`RaterExtractor`.     |
+-----------------------------------------+---------------------------------------+

|

.. automodule:: pyradise.fileio.extraction
    :members:
    :show-inheritance:
    :inherited-members:

.. _series_info_module:

Series Information Module
-------------------------
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


.. _selection_module:

Selection Module
----------------
Module: :mod:`pyradise.fileio.selection`

.. module:: pyradise.fileio.selection
    :noindex:

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

|

.. automodule:: pyradise.fileio.selection
    :members:
    :show-inheritance:
    :inherited-members:


.. _loading_module:

Loading Module
--------------
Module: :mod:`pyradise.fileio.loading`

.. module:: pyradise.fileio.loading
    :noindex:

The :mod:`~pyradise.fileio.loading` module provides extensible functionality to load a
:class:`~pyradise.data.subject.Subject` (including all selected and associated :class:`~pyradise.data.image.Image`)
specified by a list of :class:`~pyradise.fileio.series_info.SeriesInfo` entries. This module provides the
:class:`SubjectLoader` class to load a single subject and an iterative loader variant called
:class:`IterableSubjectLoader` to load multiple subjects iteratively. Furthermore, the module provides the
:class:`Loader` and the :class:`ExplicitLoader` base classes from which new loader can be derived.


.. automodule:: pyradise.fileio.loading
    :members:
    :show-inheritance:
    :inherited-members:


.. _dicom_conversion_module:

DICOM Conversion Module
-----------------------
Module: :mod:`pyradise.fileio.dicom_conversion`

.. module:: pyradise.fileio.dicom_conversion
    :noindex:

The :mod:`~pyradise.fileio.dicom_conversion` module provides functionality to convert from and to DICOM data. This
functionality is especially useful if working in the clinical context where the input and / or output data is provided
in the DICOM format.

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

|

.. automodule:: pyradise.fileio.dicom_conversion
    :members:
    :show-inheritance:
    :inherited-members:


.. _writing_module:

Writing Module
--------------
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
