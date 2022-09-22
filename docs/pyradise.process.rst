.. role:: hidden
    :class: hidden-section

.. module:: pyradise.process

Process Package
===============

The :mod:`~pyradise.process` package provides functionality for pre-processing, DL-model inference, and post-processing
in combination with the provided data model. The main building block of this package is the
:class:`~pyradise.process.base.Filter` and its associated :class:`~pyradise.process.base.FilterParams` which process
:class:`~pyradise.data.subject.Subject` instances. Due to the standardized interface, the chaining of multiple filters
in a :class:`~pyradise.process.base.FilterPipeline` is feasible, improving clarity and reproducibility. Furthermore,
this package provides an invertibility mechanism for filters that implement invertible process steps. This feature
renders feasibility to restore the original physical orientation of the processed :class:`~pyradise.data.image.Image`,
which may be crucial when processing medical imaging data. However, subsequent data processing with multiple filters
limits the invertibility because the data experiences information loss.

This package provides a basic set of extensible filter implementations. Currently, the process package includes
exclusively filters often applied in auto-segmentation development. However, we want to encourage the community to
implement and share their filters (e.g., via pull requests to the PyRaDiSe GitHub repository). The recommended workflow
for implementing new filters is documented in the documentation of the :class:`~pyradise.process.base.Filter` class.


Filter Base Module
-------------------
Module: :mod:`pyradise.process.base`

.. module:: pyradise.process.base
    :noindex:

The filter base module provides the filter base classes (i.e. :class:`~pyradise.process.base.Filter` and
:class:`~pyradise.process.base.LoopEntryFilter`) and their associated filter parameter classes (i.e.
:class:`~pyradise.process.base.FilterParams` and :class:`~pyradise.process.base.LoopEntryFilterParams`). Furthermore,
this module contains the implementation of the filter pipeline class (i.e.
:class:`~pyradise.process.base.FilterPipeline`), which is used to chain multiple parameterized filters such that they
can be executed sequentially on the same subject.


.. automodule:: pyradise.process.base
    :members:
    :show-inheritance:
    :inherited-members:

Intensity Module
--------------------
Module: :mod:`pyradise.process.intensity`

.. module:: pyradise.process.intensity
    :noindex:

The intensity module provides intensity-modifying filters, such as a
:class:`~pyradise.process.intensity.ZScoreNormFilter`. Furthermore, the intensity module
provides separate intensity filter base classes (i.e., :class:`~pyradise.process.intensity.IntensityFilter` and
:class:`~pyradise.process.intensity.IntensityLoopFilter`) to facilitate the user-driven implementation of
additional filters, including SimpleITK- or ITK-based filters.

.. automodule:: pyradise.process.intensity
    :members:
    :show-inheritance:


Orientation Module
------------------
Module: :mod:`pyradise.process.orientation`

.. module:: pyradise.process.orientation
    :noindex:

The orientation module provides functionality to reorient images such that the anatomical orientation of the data is
standardized.


.. automodule:: pyradise.process.orientation
    :members:
    :show-inheritance:


Registration Module
-------------------
Module: :mod:`pyradise.process.registration`

.. module:: pyradise.process.registration
    :noindex:

The registration module provides functionality for inter-subject and intra-subject registration of subject images.


.. automodule:: pyradise.process.registration
    :members:
    :show-inheritance:


Resampling Module
-----------------
Module: :mod:`pyradise.process.resampling`

.. module:: pyradise.process.resampling
    :noindex:

The resampling module provides functionality for resampling images.


.. automodule:: pyradise.process.resampling
    :members:
    :show-inheritance:

Modification Module
-------------------
Module: :mod:`pyradise.process.modification`

.. module:: pyradise.process.modification
    :noindex:

The modification module provides functionality to add and remove images from a subject on a filter-based approach.
In addition, this module contains a filter to merge (i.e.
:class:`~pyradise.process.modification.MergeSegmentationFilter`) multiple
:class:`~pyradise.data.image.SegmentationImage` s into one common :class:`~pyradise.data.image.SegmentationImage`.


.. automodule:: pyradise.process.modification
    :members:
    :show-inheritance:

Inference Module
----------------
Module: :mod:`pyradise.process.inference`

.. module:: pyradise.process.inference
    :noindex:

The inference module provides a prototype implementation of a DL-framework agnostic and filter-based inference class
(i.e. :class:`~pyradise.process.inference.InferenceFilter`) which is left for implementation to the user such that
the installation of a DL-framework is not required when installing PyRaDiSe.

.. automodule:: pyradise.process.inference
    :members:
    :show-inheritance:

Post-Processing Module
----------------------
Module: :mod:`pyradise.process.postprocess`

.. module:: pyradise.process.postprocess
    :noindex:

The postprocess module includes a single component filter class (i.e.
:class:`~pyradise.process.postprocess.SingleConnectedComponentFilter`) which exclusively keeps the largest single
component on the provided :class:`~pyradise.data.image.SegmentationImage` s. Furthermore, this module provides a filter
for alphabetically sorting the subject-associated :class:`~pyradise.data.image.SegmentationImage` s via their
:class:`~pyradise.data.organ.Organ` name. This functionality is essential if the :class:`~pyradise.data.organ.Organ` s
in a subsequently generated DICOM-RTSS dataset must be sorted.

.. automodule:: pyradise.process.postprocess
    :members:
    :show-inheritance:
