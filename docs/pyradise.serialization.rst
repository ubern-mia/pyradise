.. role:: hidden
    :class: hidden-section

.. module:: pyradise.serialization

Serialization Package
=====================

The :mod:`pyradise.serialization` package provides functionality for the serialization of processed data as DICOM files or discretized image formats (e.g. NIFTI).
Furthermore, the package provides functionality to generate HDF5 datasets which can be used for training deep learning in combination with for example `pymia <https://pymia.readthedocs.io/en/latest/index.html>`_.

|

Subject Serialization Module
----------------------------
Module: :mod:`pyradise.serialization.subject_serialization`

The :mod:`subject_serialization` module provides functionality to store processed subject data as DICOM or as a
discretized image format (i.e. NIFTI and NRRD, see :class:`ImageFileFormat`). The following writers are currently available:

+-----------------------------------+------------------------------------------------------+------------------------------------+
| Class                             | Input                                                | Output                             |
+===================================+======================================================+====================================+
| :class:`SubjectWriter`            | :class:`Subject`                                     | Discrete Image Files (e.g. NIFTI)  |
+-----------------------------------+------------------------------------------------------+------------------------------------+
| :class:`DirectorySubjectWriter`   | :class:`pydicom.Dataset` & path                      | Generated: DICOM, Copied: Any      |
+-----------------------------------+------------------------------------------------------+------------------------------------+
| :class:`DicomSeriesSubjectWriter` | :class:`pydicom.Dataset` & :class:`DicomSeriesInfo`  | Generated: DICOM, Copied: DICOM    |
+-----------------------------------+------------------------------------------------------+------------------------------------+

|

.. automodule:: pyradise.serialization.subject_serialization
    :show-inheritance:
    :members:

Directory Building Module
-------------------------
Module: :mod:`pyradise.serialization.directory_building`

The :mod:`directory_building` module provides functionality to generate a folder structure for storing subject data
in a common dataset.

|

.. automodule:: pyradise.serialization.directory_building
    :show-inheritance:
    :members:

HDF5 Building Module
--------------------
Module: :mod:`pyradise.serialization.h5_building`

The HDF5 file format is a fast and reliable alternative to reading single files from disk for deep learning model training.
Because PyRaDiSe can also be used for pre-processing data for deep learning model training we included this `pymia <https://pymia.readthedocs.io/en/latest/index.html>`_-based module with the aim to speed-up HDF5 generation when using PyRaDiSe for pre-processing.

**Example Serialization of Pre-Processed Data to a HDF5 Dataset**

The following example illustrates the usage of the HDF5 generation process.

.. code-block:: python

    from argparse import ArgumentParser

    from pyradise.serialization import (SimpleFilePathGenerator, FileSystemDatasetCreator)
    from pyradise.data import (OrganRaterCombination, Modality)


    def main(input_dir_path: str,
             output_dir_path: str
             ) -> None:
        """Generate a HDF5 dataset file from multiple subjects with the following input data structure:

            input_dir
            |
            |- SUBJECT_001
            | |- img_SUBJECT_001_T1c.nii.gz
            | |- img_SUBJECT_001_T1w.nii.gz
            | |- img_SUBJECT_001_T2w.nii.gz
            | |- img_SUBJECT_001_FLAIR.nii.gz
            | |- seg_SUBJECT_001_Combination_all.nii.gz
            |
            |- SUBJECT_002
            | |- img_SUBJECT_002_T1c.nii.gz
            | |- img_SUBJECT_002_T1w.nii.gz
            | |- img_SUBJECT_002_T2w.nii.gz
            | |- img_SUBJECT_002_FLAIR.nii.gz
            | |- seg_SUBJECT_002_Combination_all.nii.gz
            |
            | - ...


        Args:
            input_dir_path (str): The path to the data directory where each subject has a separate folder.
            output_dir_path (str): The path to the output HDF5 file (i.e. /YOUR/DESIRED/OUTPUT_PATH/dataset.h5).

        Returns:
            None
        """
        label_identifiers = {'LB': OrganRaterCombination('all', 'Combination')}
        image_identifiers = {'T1c': Modality.T1c,
                             'T1w': Modality.T1w,
                             'T2w': Modality.T2w,
                             'FLAIR': Modality.FLAIR}

        file_path_generator = SimpleFilePathGenerator(tuple(label_identifiers.values()),
                                                      tuple(image_identifiers.values()))
        creator = FileSystemDatasetCreator(input_dir_path, output_dir_path, file_path_generator,
                                           label_identifiers, image_identifiers)
        creator.create()


    if __name__ == '__main__':
        parser = ArgumentParser()
        parser.add_argument('-input_dir_path', type=str)
        parser.add_argument('-output_dir_path', type=str)
        args = parser.parse_args()

        main(args.input_dir_path, args.output_dir_path)

|

.. automodule:: pyradise.serialization.h5_building
    :show-inheritance:
    :special-members:
    :exclude-members: __init__, __dict__, __weakref__
    :members:
