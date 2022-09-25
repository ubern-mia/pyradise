Filter Base Module
==================
Module: :mod:`pyradise.process.base`

.. module:: pyradise.process.base
    :noindex:

General
-------

The filter base module provides the filter base classes (i.e. :class:`~pyradise.process.base.Filter` and
:class:`~pyradise.process.base.LoopEntryFilter`) and their associated filter parameter classes (i.e.
:class:`~pyradise.process.base.FilterParams` and :class:`~pyradise.process.base.LoopEntryFilterParams`). Furthermore,
this module contains the implementation of the filter pipeline class (i.e.
:class:`~pyradise.process.base.FilterPipeline`), which is used to chain multiple parameterized filters such that they
can be executed sequentially on the same subject.

Class Overview
--------------

The following abstract base classes are provided by the :mod:`~pyradise.process.base` module:

+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| Class                                   | Description                                                                                                                |
+=========================================+============================================================================================================================+
| :class:`FilterParams`                   | Base class for :class:`FilterParams` subclasses that parameterize :class:`LoopEntryFilter` subclasses.                     |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`Filter`                         | Base class for :class:`Filter` subclasses that process the provided :class:`~pyradise.data.subject.Subject` instances.     |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`LoopEntryFilterParams`          | Base class for :class:`LoopEntryFilterParams` subclasses that parametrize :class:`LoopEntryFilter` subclasses.             |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`LoopEntryFilter`                | Base class for :class:`LoopEntryFilter` subclasses that allow loop-based processing of image subsets.                      |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| :class:`FilterPipeline`                 | A class for combining multiple :class:`Filter` instances into a common pipeline.                                           |
+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------------+

Details
-------

.. automodule:: pyradise.process.base
    :members:
    :show-inheritance:
    :inherited-members:

