.. automodule:: pyradise.data

Overview Data Package
=====================

The :mod:`~pyradise.data` package contains the data model for PyRaDiSe which will be used during loading
(see :mod:`pyradise.fileio.loading` module), processing (see :mod:`pyradise.process` package) and writing (see
:mod:`pyradise.fileio.writing` module). The goal of the data model design is to provide an simple, lightweight, and
extensible RT-oriented interface for the user to work with the data. First, simple because handling data with a
simple interface should be easy and intuitive. Second, lightweight because the data model should not add a lot of
overhead because processing of medical images typically requires large amounts of memory. Third, extensible because the
data model should be easily extendable to support new features and new data types such as for example DICOM Dose Plans.

The :class:`~pyradise.data.subject.Subject` is the top-level data holding container combining all necessary
subject-level information such as the subject's name, the intensity and segmentation images and additional user
defined data in one common data structure. Typically, the :class:`~pyradise.data.subject.Subject` is created directly
by a :class:`~pyradise.fileio.loading.SubjectLoader` when loading data from disk. However, it can also be constructed
manually in order to render feasibility for working with other libraries such as `MONAI <https://monai.io/>`_.

The :class:`~pyradise.data.subject.Subject` comprises of a list of :class:`~pyradise.data.image.IntensityImage` and
:class:`~pyradise.data.image.SegmentationImage` images with each image possessing additional information about the
image content such as the :class:`~pyradise.data.modality.Modality` or the :class:`~pyradise.data.organ.Organ`
segmented. Furthermore, each image contains a :class:`~pyradise.data.taping.TransformTape` which is used to keep
track of all necessary physical property (i.e. origin, direction, spacing, size) changes during processing. The
:class:`~pyradise.data.taping.TransformTape` provides also functionality to revert the changes to the original
physical properties by playback the recorded changes. Each :class:`~pyradise.data.image.Image` type posses
distinctive content-related information which are enlisted below:


.. figure:: ../_static/data_image_0.png
    :name: fig_data_subject
    :width: 600
    :align: center
    :alt: main concept data model

    Schematic illustration of the subject and the images.


Intensity Image
---------------

In addition to the image data and the transform tape, an :class:`~pyradise.data.image.IntensityImage` contains
information about the :class:`~pyradise.data.modality.Modality`. The :class:`~pyradise.data.modality.Modality` is used
to distinguish between different image modalities and their details such as CT, PET, or MR. The naming of the
different :class:`~pyradise.data.modality.Modality` instances is determined during the loading of
the :class:`~pyradise.data.subject.Subject` using either a modality configuration file or a
:class:`~pyradise.fileio.extraction.ModalityExtractor`.


Segmentation Image
------------------

Additionally to the image data and the transform tape, a :class:`~pyradise.data.image.SegmentationImage` contains
information about the :class:`~pyradise.data.organ.Organ` segmented on the image and the
:class:`~pyradise.data.rater.Rater` who generated the segmentations / contours. By design, each
:class:`~pyradise.data.image.SegmentationImage` instance should contain a single organ / label to allow for simple
processing. As explained earlier this is not a hard constraint and can be circumvented in appropriate cases such as
for example if one needs to output multi-label segmentations.



