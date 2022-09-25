Modification Module
===================
Module: :mod:`pyradise.process.modification`

.. module:: pyradise.process.modification
    :noindex:

General
-------

The modification module provides functionality to add and remove images from a subject on a filter-based approach.
In addition, this module contains a filter to merge (i.e.
:class:`~pyradise.process.modification.MergeSegmentationFilter`) multiple
:class:`~pyradise.data.image.SegmentationImage` s into one common :class:`~pyradise.data.image.SegmentationImage`.

Class Overview
--------------

The following classes are provided by the :mod:`~pyradise.process.modification` module:

+--------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| Class                                      | Description                                                                                                                                 |
+============================================+=============================================================================================================================================+
| :class:`AddImageFilterParams`              | Parameterization class for the :class:`AddImageFilter`.                                                                                     |
+--------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`AddImageFilter`                    | Filter to add new :class:`~pyradise.data.image.Image` instances to a :class:`~pyradise.data.subject.Subject` instance.                      |
+--------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`RemoveImageByOrganFilterParams`    | Parameterization class for the :class:`RemoveImageByOrganFilter`.                                                                           |
+--------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`RemoveImageByOrganFilter`          | Filter to remove a :class:`~pyradise.data.image.SegmentationImage` instances from a subject via its :class:`~pyradise.data.organ.Organ`.    |
+--------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`RemoveImageByRaterFilterParams`    | Parameterization class for the :class:`RemoveImageByRaterFilter`.                                                                           |
+--------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`RemoveImageByRaterFilter`          | Filter to remove a :class:`~pyradise.data.image.SegmentationImage` instances from a subject via its :class:`~pyradise.data.rater.Rater`.    |
+--------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`RemoveImageByModalityFilterParams` | Parameterization class for the :class:`RemoveImageByModalityFilter`.                                                                        |
+--------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`RemoveImageByModalityFilter`       | Filter to remove a :class:`~pyradise.data.image.IntensityImage` instances from a subject via its :class:`~pyradise.data.modality.Modality`. |
+--------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`MergeSegmentationFilterParams`     | Parameterization class for the :class:`MergeSegmentationFilter`.                                                                            |
+--------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| :class:`MergeSegmentationFilter`           | Filter to merge / combine multiple :class:`~pyradise.data.image.SegmentationImage` instances in to one multi-label instance.                |
+--------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------+

Details
-------

.. automodule:: pyradise.process.modification
    :members:
    :show-inheritance:

