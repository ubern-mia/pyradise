API
===

The functionality of PyRaDiSe is split into the following packages:

* | **Conversion**
  | The conversion package is functionality for converting DICOM images, DICOM-RT and DICOM registration files to a Subject and/or to SimpleITK images and vice-versa.
* | **Curation**
  | The curation package contains functionality for pre-processing, deep learning model inference, and post-processing.
* | **Data**
  | The data package contains the data structures for PyRaDiSe.
* | **Loading**
  | The conversion package provides functionality for loading discretized image data (e.g. NIFTI files).
* | **Serialization**
  | The serialization package contains functionality for serialization of processed data as DICOM files or discretized image formats (e.g. NIFTI).


.. toctree::
    :maxdepth: 3
    :caption: Packages

    pyradise.conversion
    pyradise.curation
    pyradise.data
    pyradise.loading
    pyradise.serialization