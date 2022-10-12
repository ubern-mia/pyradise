API Reference
=============

The functionality of PyRaDiSe is split into the following packages:

* | **Data Package**
  | The data package provides a lightweight, reproducible, and extensible RT-oriented data model.
  | First, lightweight because working with medical image data requires large amounts of memory;
  | thus, the introduced functionality should consume a minimum of additional memory. Second,
  | reproducible because the data model should not bias or influence the evaluation of different
  | auto-segmentation models on the same data. Third, extensible so that additional data entities
  | (e.g., DICOM RT Dose) can be added in the future to extend the functionality and scope of PyRaDiSe.
* | **FileIO Package**
  | The fileio package provides highly automated, versatile, and easy-to-use functionality for data
  | import, export, and DICOM-RTSS conversion. First, automated because importing and converting medical
  | image data, particularly DICOM data, can be complex, confusing, and time-consuming if, for instance,
  | referenced DICOM registrations need to be applied to DICOM images during import. Second, versatile
  | because loading, converting, and serializing medical image data requires many computational resources,
  | and thus a streamlined import and export process featuring low computational complexity is important.
  | Furthermore, versatile because the fileio package can handle input and output data of various forms
  | and structures. Third, easy-to-use because data ingestion, conversion, and serialization should be
  | as simple as possible for developers, reducing development time and overall improved code readability.
* | **Process Package**
  | The process package offers a selection of pre-processing and post-processing procedures. These
  | procedures are either implemented by the package authors, or via wrapped versions of procedures found
  | in other popular frameworks such as SimpleITK and ITK packages. For deep learning model inference,
  | PyRaDiSe incorporates prototypes and examples exclusively to stay deep learning framework agnostic such
  | that the user can use the framework of his choice. Furthermore, the optional use of a deep learning
  | framework drastically reduces the memory footprint of PyRaDiSe itself, which is beneficial if PyRaDiSe
  | is used for conversion purposes only.

.. toctree::
    :maxdepth: 4
    :caption: Data Package
    :hidden:

    reference/pyradise.data
    reference/pyradise.data.subject
    reference/pyradise.data.image
    reference/pyradise.data.taping
    reference/pyradise.data.modality
    reference/pyradise.data.organ
    reference/pyradise.data.annotator


.. toctree::
    :maxdepth: 4
    :caption: FileIO Package
    :hidden:

    reference/pyradise.fileio
    reference/pyradise.fileio.crawling
    reference/pyradise.fileio.modality_config
    reference/pyradise.fileio.extraction
    reference/pyradise.fileio.series_info
    reference/pyradise.fileio.selection
    reference/pyradise.fileio.loading
    reference/pyradise.fileio.dicom_conversion
    reference/pyradise.fileio.writing


.. toctree::
    :maxdepth: 4
    :caption: Process Package
    :hidden:

    reference/pyradise.process
    reference/pyradise.process.base
    reference/pyradise.process.intensity
    reference/pyradise.process.orientation
    reference/pyradise.process.registration
    reference/pyradise.process.resampling
    reference/pyradise.process.modification
    reference/pyradise.process.inference
    reference/pyradise.process.postprocess
    reference/pyradise.process.invertibility

