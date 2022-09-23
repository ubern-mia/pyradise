.. _loading_module:

Loading Module
==============
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

