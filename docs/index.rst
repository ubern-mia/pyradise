.. PyRaDiSe documentation master file, created by
   sphinx-quickstart on Wed Jul 27 18:01:57 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyRaDiSe's Documentation!
====================================

PyRaDiSe is an open-source Python (Py) library focusing on handling radiotherapy (Ra) DICOM (Di) data for segmentation (Se) tasks. This library addresses three main parts of machine learning or deep learning-based segmentation pipelines:

 * | **Data Conversion:**
   | Converting clinical DICOM images and segmentations to and from computation-oriented image formats (i.e. NIFTI) is a well-known challenge for data scientists in the field of radio-oncology. PyRaDiSe provides functionality to convert DICOM images and RT-STRUCTS to NIFTI images and to generate DICOM RT-STRUCTS from NIFTI or ITK/SimpleITK segmentation masks.
 * | **Data Curation:**
   | The reproducible and fast curation of clinical data demands for suitable pre-processing pipelines. The functionality included in PyRaDiSe renders the generation of extensible, understandable and reproducible pre-processing pipelines feasible.
 * | **Model Inference:**
   | The inference of deep learning models on clinical data for segmentation purposes requires different implementations than for model training. PyRaDiSe provides a simple interface for including model inference in one common pipeline with pre- and post-processing and serialization of clinical data.

Therefore, PyRaDiSe is highly flexible, reduces burdens on data conversion, pre-processing, post-processing, and serialization to DICOM and renders agile model deployment for clinical evaluation feasible. Furthermore, PyRaDiSe is easily extensible and contributions are highly appreciated.

Main Features
=============
The main feature of PyRaDiSe is the simplification of deep learning models inference feasible  CONTINUE HERE

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   examples
   api
   change_history
   acknowledgment


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
