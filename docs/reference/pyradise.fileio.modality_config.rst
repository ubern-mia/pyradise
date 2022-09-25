.. _modality_config_module:

Modality Configuration Module
=============================
Module: :mod:`pyradise.fileio.modality_config`

.. module:: pyradise.fileio.modality_config
    :noindex:

General
-------

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

Class Overview
--------------

The following class is provided by the :mod:`~pyradise.fileio.modality_config` module:

+------------------------------------+-------------------------------------------------------------------------+
| Class                              | Description                                                             |
+====================================+=========================================================================+
| :class:`ModalityConfiguration`     | Class handling all information to identify a modality and its details.  |
+------------------------------------+-------------------------------------------------------------------------+


Details
-------

.. automodule:: pyradise.fileio.modality_config
    :members:
    :show-inheritance:
    :inherited-members:

