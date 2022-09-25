.. _loading_module:

Loading Module
==============
Module: :mod:`pyradise.fileio.loading`

.. module:: pyradise.fileio.loading
    :noindex:

General
-------

The :mod:`~pyradise.fileio.loading` module provides extensible functionality to load a
:class:`~pyradise.data.subject.Subject` (including all selected and associated :class:`~pyradise.data.image.Image`)
specified by a list of :class:`~pyradise.fileio.series_info.SeriesInfo` entries. This module provides the
:class:`SubjectLoader` class to load a single subject and an iterative loader variant called
:class:`IterableSubjectLoader` to load multiple subjects iteratively. Furthermore, the module provides the
:class:`Loader` and the :class:`ExplicitLoader` base classes from which new loader can be derived.

Class Overview
--------------

The following :class:`Loader` classes are provided by the :mod:`~pyradise.fileio.loading` module:

+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| Class                                   | Description                                                                                                                |
+=========================================+============================================================================================================================+
| :class:`Loader`                         | Base class for all :class:`Loader` subclasses                                                                              |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`ExplicitLoader`                 | Base class for :class:`Loader` subclasses containing an explicit :meth:`~ExplicitLoader.load` method.                      |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`SubjectLoader`                  | A :class:`Loader` class for explicit loading :class:`~pyradise.data.subject.Subject` instances.                            |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`IterableSubjectLoader`          | A :class:`Loader` class for iterative loading of :class:`~pyradise.data.subject.Subject` instances.                        |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+

Details
-------

.. automodule:: pyradise.fileio.loading
    :members:
    :show-inheritance:
    :inherited-members:

