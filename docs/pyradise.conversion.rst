.. role:: hidden
    :class: hidden-section


Conversion Package
==================

The :mod:`pyradise.conversion` package provides functionality for converting DICOM images, DICOM-RT and DICOM
registration files to a :class:`Subject` and/or to :class:`itk.Image` / :class:`SimpleITK.Image` images and
vice-versa (excluding the conversion of :class:`IntensityImage` to DICOM images).

The main concept of the :mod:`pyradise.conversion` package is illustrated in the figures below.

.. image:: _static/conversion_image_0.png
    :width: 800
    :align: center
    :alt: main concept conversion module

|

Configuration Module
--------------------


One drawback of working with DICOM images in radiotherapy is that the metadata used to identify a specific MR-sequence
(e.g. T1w or T1c) can  be ambiguous or may not be provided. Furthermore, the naming of the necessary metadata may be
specific on the vendor of MR equipment or the clinic. Thus, for a reliable conversion of multiple single patient DICOM
images additional information must be provided semi-automatically as a separate *modality configuration file*
(by default named: :file:`modality_config.json`).

The :class:`ModalityConfiguration` is the responsible class for handling the modality configuration and in combination
with a DICOM-specific :class:`Crawler` provides functionality to generate the *modality configuration file* skeleton
which can then be modified by the user or a separate script. Furthermore, the :class:`ModalityConfiguration` can load
existing *modality configuration files* and provide the data to the :class:`DicomSeriesImageInfo`.

.. note::
    Typically, the :class:`ModalityConfiguration` is used inside of a DICOM-specific :class:`Crawler` and for most use
    cases it's sufficient to be aware that additional information on the sequence naming must be provided via a
    *modality configuration file*. Detailed information see for example :class:`DicomSubjectDirectoryCrawler`.

**Example Modality Configuration File**

The following *modality configuration file* was automatically generated using :class:`DicomSubjectDirectoryCrawler`
and was completed filling the tag ``Modality`` with values defined in :class:`curation.data.modality.Modality`.

.. code-block:: json-object

    [
        {
            "SOPClassUID": "1.2.840.10008.5.1.4.1.1.4",
            "StudyInstanceUID": "1.3.6.1.4.1.5962.99.1.1856959841.1925150802.1556635153761.6.0",
            "SeriesInstanceUID": "1.3.6.1.4.1.5962.99.1.1856959841.1925150802.1556635153761.239.0",
            "SeriesDescription": "t1_mpr_sag_we_p2_iso",
            "SeriesNumber": 7,
            "DICOM_Modality": "MR",
            "Modality": "T1c"
        },
        {
            "SOPClassUID": "1.2.840.10008.5.1.4.1.1.2",
            "StudyInstanceUID": "1.3.6.1.4.1.5962.99.1.2628079426.196750453.1557406273346.1015.0",
            "SeriesInstanceUID": "1.3.6.1.4.1.5962.99.1.2628079426.196750453.1557406273346.1016.0",
            "SeriesDescription": "Unnamed_Series",
            "SeriesNumber": 2,
            "DICOM_Modality": "CT",
            "Modality": "CT"
        }
    ]

.. note::
    A detailed example for generating a *modality configuration file* skeleton is provided in the example section.

|



Crawling Module
---------------


The :mod:`crawling` module is responsible for retrieving :class:`DicomSeriesInfo` objects from DICOM files and
*modality configuration files*, which contain all the necessary information for the subsequent conversion process.
Furthermore, the :mod:`crawling` module provides functionality to generate the *modality configuration files* if they
are not existing. For details about the *modality configuration files* see `Configuration Module`_.

**Example Modality Configuration File Generation**

The generation of the *modality configuration files* is typically the first step for converting DICOM files to a
discretized image format (e.g. NIFTI format). The following example demonstrates this generation step for a single
subject:

.. code-block:: python

    from argparse import ArgumentParser

    from pyradise.conversion.crawling import DicomSubjectDirectoryCrawler


    def main(subject_directory: str) -> None:
        print(f'Crawling at {subject_directory}...')
        crawler = DicomSubjectDirectoryCrawler(subject_directory, write_modality_config=True)
        crawler.execute()
        print('Crawling finished!')


    if __name__ == '__main__':
        parser = ArgumentParser()
        parser.add_argument('-subject_directory', type=str)
        args = parser.parse_args()

        main(args.subject_directory)

After the *modality configuration file* has been generated it needs to be adjusted manually or with a separate
script such that it contains the correct modality information for each sequence. If the subject contains just one
sequence you may leave the modality config file as it is (default assigned modality: *UNKNOWN*).

**Example Retrieval of Series Infos from DICOM Files and a Modality Configuration File**

The following example demonstrates the retrieval of :class:`DicomSeriesInfo`'s for a subject for which the
*modality configuration file* is already existing.

.. code-block:: python

    from argparse import ArgumentParser
    from typing import Tuple

    from pyradise.conversion.crawling import DicomSubjectDirectoryCrawler
    from pyradise.conversion.series_information import DicomSeriesInfo


    def main(subject_directory: str) -> None:
        crawler = DicomSubjectDirectoryCrawler(subject_directory)
        series_infos: Tuple[DicomSeriesInfo] = crawler.execute()

        for series_info in series_infos:
            print(f'Retrieved series {series_info.series_description} '
                  f'with SeriesInstanceUID {series_info.series_instance_uid}')


    if __name__ == '__main__':
        parser = ArgumentParser()
        parser.add_argument('-subject_directory', type=str)
        args = parser.parse_args()

        main(args.subject_directory)

**Data Structure and Extensibility**

The current release of PyRaDiSe contains functionality to process DICOM data from a single subject which is contained
in one subject folder (incl. subfolders, use :class:`DicomSubjectDirectoryCrawler`) or DICOM data from multiple
subjects in separated subject folders (incl. subfolders, use :class:`DicomDatasetDirectoryCrawler`). However, you can
extend the functionality of the crawlers by inheriting from the abstract :class:`Crawler` to adopt for your data
structure.

.. note::
    In case of large datasets or limited memory we recommend to use the :class:`IterableDicomDatasetDirectoryCrawler`
    which loads the sequentially and reduces memory usage.

|




Series Information Module
-------------------------


The :mod:`series_information` module is responsible for providing all necessary information to the conversion process.
Because DICOM images, registrations, and RT-STRUCTs contain different information necessary for the conversion process
the :mod:`series_information` module contains separate objects for all these classes of DICOM datasets.

Additionally, the :mod:`series_information` module contains filter classes (e.g. :class:`DicomSeriesImageInfoFilter`)
to filter :class:`DicomSeriesInfo` such that for example unused :class:`Modality` 's can be excluded from the
subsequent conversion process.

|



DICOM Conversion Module
-----------------------


The :mod:`dicom_conversion` module is responsible for the conversion of the data based on the information provided by
the :class:`DicomSeriesInfo`. Different interfaces for converting from and to DICOM are available. In general, classes
starting with a *Dicom* or *Subject* in their name are predominantly implemented with a focus on the data model of
PyRaDiSe whereas the remaining classes are designed for a more flexible use.

**Overview of Converters**

+-----------------------------------------------+--------------------+------------------------------------------------------+----------------------------+
| Class                                         | Data Model         | Input                                                | Output                     |
+===============================================+====================+======================================================+============================+
| :class:`DicomSubjectConverter`                | PyRaDiSe           | :class:`DicomSeriesInfo` subtypes                    | :class:`Subject`           |
+-----------------------------------------------+--------------------+------------------------------------------------------+----------------------------+
| :class:`DicomSeriesImageConverter`            | PyRaDiSe           | :class:`DicomSeriesInfo` subtypes                    | :class:`IntensityImage`    |
+-----------------------------------------------+--------------------+------------------------------------------------------+----------------------------+
| :class:`DicomSeriesRTStructureSetConverter`   | PyRaDiSe           | :class:`DicomSeriesInfo` subtypes                    | :class:`SegmentationImage` |
+-----------------------------------------------+--------------------+------------------------------------------------------+----------------------------+
| :class:`SubjectRTStructureSetConverter`       | PyRaDiSe           | :class:`Subject` & :class:`DicomSeriesInfoImage`     | :class:`pydicom.Dataset`   |
+-----------------------------------------------+--------------------+------------------------------------------------------+----------------------------+
| :class:`RTSSToImageConverter`                 | General Purpose    | :class:`pydicom.Dataset`                             | :class:`SimpleITK.Image`   |
+-----------------------------------------------+--------------------+------------------------------------------------------+----------------------------+
| :class:`ImageToRTSSConverter`                 | General Purpose    | :class:`SimpleITK.Image` & :class:`pydicom.Dataset`  | :class:`pydicom.Dataset`   |
+-----------------------------------------------+--------------------+------------------------------------------------------+----------------------------+

.. note::
    In radiotherapy DICOM images often need to be registered to each other before processing. The converters of
    PyRaDiSe are feasible to process DICOM registration files and will automatically apply the appropriate transformations
    to the images if the DICOM registration files are provided.

|



Utilities Module
----------------


The :mod:`utils` module provides functionality which is used at multiple locations in the :mod:`pyradise.conversion`
package.

|


