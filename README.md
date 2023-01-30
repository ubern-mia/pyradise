PyRaDiSe
========

[![Documentation Status](https://readthedocs.org/projects/pyradise/badge/?version=latest)](https://pyradise.readthedocs.io/en/latest/?badge=latest)

PyRaDiSe is an open-source Python (Py) package for developing deployable, radiotherapy-oriented (Ra), DICOM-based (Di) 
auto-segmentation (Se) solutions. PyRaDiSe is DL framework-independent but can easily integrate most DL frameworks, 
such as PyTorch or TensorFlow. The package addresses the following challenges for building radiotherapy-oriented 
auto-segmentation solutions: handling DICOM data, managing and converting DICOM-RTSS data (incl. a 2D-based and 
a 3D-based conversion algorithm), invertible pre-processing, and post-processing. In addition to building 
auto-segmentation solutions, PyRaDiSe allows for converting and curating DICOM image series and DICOM-RTSS data to 
simplify segmentation training dataset construction. Therefore, PyRaDiSe is highly flexible, allows for fast 
prototyping, and facilitates a fast transition of data science research results into clinical radiotherapy research.

<img alt="PyRaDiSe_Meme" src="https://github.com/ubern-mia/pyradise/raw/main/docs/_static/meme.jpg" width="300">

Main Features
-------------
The main features of PyRaDiSe are data handling, conversion from and to DICOM-RTSS, and data processing, including deep 
learning model inference. The intended use of PyRaDiSe in the radiotherapy environment is depicted below. The 
DICOM and other discrete medical image file formats, such as NIfTI, are imported into the provided data model using 
the [`fileio` package](https://pyradise.readthedocs.io/en/latest/reference/pyradise.fileio.html). In contrast to the 
standard way of loading DICOM data, this package provides comprehensive and flexible import routines that consider 
data relation details and automate import steps, such as registering DICOM images if DICOM registration files are 
available. However, in some cases, the DICOM standard does not provide sufficient information for automation, 
requiring minimal human interaction for resolution. In addition, discrete medical images also suffer from the lack of 
identification data needed for automation. However, the [`fileio` package](https://pyradise.readthedocs.io/en/latest/reference/pyradise.fileio.html) 
package offers the necessary methods to address these issues with flexible approaches and prototypes. Furthermore, 
the [`fileio` package](https://pyradise.readthedocs.io/en/latest/reference/pyradise.fileio.html) provides 
routines to select specific entities from the available data before loading by generating filterable pre-loading 
information (so-called [`SeriesInfo`](https://pyradise.readthedocs.io/en/latest/reference/pyradise.fileio.series_info.html#pyradise.fileio.series_info.SeriesInfo))
so that the computation time and memory usage for loading is minimal. Finally, after the data is loaded, it is 
represented using the data model implemented in the [`data` package](https://pyradise.readthedocs.io/en/latest/reference/pyradise.data.html). 
All downstream tasks are performed using the simple and extensible radiotherapy-oriented data model from this step on.

After loading, the data is either converted and written to a file or processed using routines from the 
[`process` package](https://pyradise.readthedocs.io/en/latest/reference/pyradise.process.html). This package includes 
functionality and prototypes for pre-processing, deep learning model inference, and post-processing with a similar mode 
of operations as well-known medical image libraries, such as SimpleITK or ITK. However, in contrast to other libraries, 
the process package offers a mechanism for guaranteeing reproducibility and limited invertibility.

After processing or loading, the altered data can be written to disk using a versatile writer from the 
[`fileio` package](https://pyradise.readthedocs.io/en/latest/reference/pyradise.fileio.html) to save the data as either 
a discrete image file or as DICOM-RTSS. In addition, specific writers provide the additional functionality to copy 
the input data from the source to the target directory. This feature is handy if the developed auto-segmentation 
solution will be deployed to the clinical environment or the cloud, where the original input data should remain 
unmodified.

<img src="https://github.com/ubern-mia/pyradise/raw/main/docs/_static/architecture_overview_v2.png" alt="Schematic illustration of PyRaDiSe in the radiotherapy environment">


Getting Started
---------------

If you are new to PyRaDiSe, here are a few guides to get you up to speed right away:

 - [Installation](https://pyradise.readthedocs.io/en/latest/installation.html) for installation instructions - or simply run `pip install pyradise`
 - [Examples](https://pyradise.readthedocs.io/en/latest/examples.html) give you an overview of PyRaDiSe's intended use. Jupyter notebooks are available in the directory [./examples](https://github.com/ubern-mia/pyradise/tree/main/examples/).
 - [Change history](https://pyradise.readthedocs.io/en/latest/change_history.html)
 - [Acknowledgments](https://pyradise.readthedocs.io/en/latest/acknowledgment.html)


Citation
--------

If you use PyRaDiSe for your research, please acknowledge it accordingly by citing our paper:

BibTeX entry:

    @article{Ruefenacht2023,
    author = {Rüfenacht, Elias and Kamath, Amith and Suter, Yannick and Poel, Robert and Ermis, Ekin and Scheib, Stefan and Reyes, Mauricio},
    title = {{PyRaDiSe: A Python package for DICOM-RT-based auto-segmentation pipeline construction and DICOM-RT data conversion}},
    journal = {Computer Methods and Programs in Biomedicine},
    doi = {10.1016/j.cmpb.2023.107374},
    issn = {0169-2607},
    year = {2023}
    }
