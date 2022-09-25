.. _extraction_module:

Extraction Module
=================
Module: :mod:`pyradise.fileio.extraction`

.. module:: pyradise.fileio.extraction
    :noindex:

General
-------

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

Class Overview
--------------

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

Details
-------

.. automodule:: pyradise.fileio.extraction
    :members:
    :show-inheritance:
    :inherited-members:

