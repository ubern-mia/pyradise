.. _crawling_module:

Crawling Module
===============
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

