.. automodule:: pyradise.fileio

Overview FileIO Package
=======================

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
