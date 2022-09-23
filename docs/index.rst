.. toctree::
   :maxdepth: 3
   :caption: Contents:
   :hidden:

   installation
   examples
   api
   change_history
   acknowledgment

Welcome to PyRaDiSe's Documentation
===================================

PyRaDiSe (Python package for Radiotherapy-oriented and DICOM-based auto-Segmentation) is an open-source Python package
for building clinically applicable  radiotherapy-oriented auto-segmentation solutions. This package addresses two main
challenges of auto-segmentation in clinical radiotherapy: data handling and conversion from and to DICOM-RTSS. Besides a
radiotherapy-oriented data model and conversion capabilities, PyRaDiSe provides a profound set of extensible processing
filters for fast prototyping and development of clinically deployable auto-segmentation pipelines. Therefore, PyRaDiSe
is a highly flexible and extensible toolbox, allowing for narrowing the gap between data science and clinical
radiotherapy research and speeding up development cycles.

Main Features
-------------
The main features of PyRaDiSe are data handling, conversion from and to DICOM-RTSS, and data processing,
including deep learning model inference. The intended use of PyRaDiSe in the radiotherapy environment is depicted in
:numref:`Fig. %s <fig-overview>`.

.. figure:: ./_static/architecture_overview_v2.png
    :name: fig-overview
    :figwidth: 80 %
    :align: center
    :alt: Schematic illustration of PyRaDiSe in the radiotherapy environment

    Schematic illustration of PyRaDiSe in the radiotherapy environment


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
    :maxdepth: 3
    :caption: API Reference:
    :hidden:

    pyradise.data
    pyradise.fileio