.. module:: pyradise.conversion

Conversion (:mod:`pyradise.conversion` package)
===============================================

This data package provides data handling functionality for machine learning (especially deep learning) projects.
The concept of the data package is illustrated in the figure below.


Configuration (:mod:`pyradise.conversion.configuration` module)
---------------------------------------------------------------

The :class:`ModalityConfiguration` is responsible for loading and applying the correct modality information to the :class:`DicomSeriesImageInfo` or to generate the modality config file skeleton (default: :file:`modality_config.json`). This step is necessary because DICOM image files for certain image modalities (i.e. MR) do not contain robust information to retrieve the detailed modality (e.g. T1c, T1w, FLAIR).

If the :class:`ModalityConfiguration` class is used to generate the modality config file skeleton its content must be modified manually or with an appropriate automation script. For generating the modality config file skeleton you can use a comparable script as in the following example:



.. automodule:: pyradise.conversion.configuration
    :members: ModalityConfiguration

Series Information (:mod:`pyradise.conversion.series_information` module)
-------------------------------------------------------------------------

SHORT DESCRIPTION

.. automodule:: pyradise.conversion.series_information
    :members:

.. automodule:: pyradise.conversion.crawling
    :members:

.. automodule:: pyradise.conversion.base_conversion
    :members:

.. automodule:: pyradise.conversion.dicom_conversion
    :members:

.. automodule:: pyradise.conversion.rtss_conversion
    :members:

.. automodule:: pyradise.conversion.nifti_conversion
    :members:

.. automodule:: pyradise.conversion.utils
    :members:
