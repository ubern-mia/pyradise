Examples
========

The following examples illustrate the intended use of PyRaDiSe:

.. toctree::
    :maxdepth: 1

    examples.conversion.dicom_to_nii
    examples.conversion.nii_to_dicom
    examples.processing.basic_processing
    examples.inference.basic
    examples.inference.container



Example Data
------------
The data from the "Segmentation of Vestibular Schwannoma from Magnetic Resonance Imaging" dataset was used for the
given examples. Because this dataset is large with 242 patients in total, a subset of data from 5 patients is made
available in a separate `GitHub repository <https://github.com/ruefene/pyradise-example-data>`_ such that the reader
can execute the given examples in his/her setup. The data is available as DICOM files and as NIfTI files.

The example data contains for each patient a T1-weighted Gd-enhanced MR image, a T2-weighted MR image, and a DICOM-RTSS
with segmentations of the current tumor volume, the left cochlea, the skull, and an older version of the tumor volume.
The original data includes in addition a second RT Structure Set, two RT Dose, and two RT Plans for each patient that
were removed to lower the size of the example data. All images were acquired with a 32-channel 1.5T Siemens Avanto
scanner between the years 2012 and 2018. The RT Structure Sets were created manually in consensus between the
neurosurgeon and physician using both the T1- and T2-weighted images. The segmentation were performed using the
Leksell GammaPlan software from Elekta, Sweden.

In addition to the example patient data, a PyTorch-based DL-model for skull segmentation is provided that is used in
the `inference example <examples.inference.basic.ipynb>`_. The model was trained on 50 patients from the mentioned
dataset for 15 epochs with a fixed learning rate of 0.0001 and a batch size of 4 images. The Binary Cross Entropy loss
function was used for training. Because this model is used for demonstration purposes only and is not intended for
clinical use, the model is not trained until convergence.

.. seealso::
    Shapey, J., Kujawa, A., Dorent, R., Wang, G., Bisdas, S., Dimitriadis, A., Grishchuck, D., Paddick, I.,
    Kitchen, N., Bradford, R., Saeed, S., Ourselin, S., & Vercauteren, T. (2021). Segmentation of Vestibular
    Schwannoma from Magnetic Resonance Imaging: An Open Annotated Dataset and Baseline Algorithm [Data set].
    The Cancer Imaging Archive. `https://doi.org/10.7937/TCIA.9YTJ-5Q73 <https://doi.org/10.7937/TCIA.9YTJ-5Q73>`_.

.. seealso::
    Shapey, J., Kujawa, A., Dorent, R., Wang, G., Dimitriadis, A., Grishchuk, D., Paddick, I., Kitchen, N.,
    Bradford, R., Saeed, S. R., Bisdas, S., Ourselin, S., & Vercauteren, T. (2021). Segmentation of vestibular
    schwannoma from MRI, an open annotated dataset and baseline algorithm. In Scientific Data (Vol. 8, Issue 1).
    Springer Science and Business Media LLC. `https://doi.org/10.1038/s41597-021-01064-w
    <https://doi.org/10.1038/s41597-021-01064-w>`_.

.. seealso::
    Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., Moore, S., Phillips, S., Maffitt, D.,
    Pringle, M., Tarbox, L., & Prior, F. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a
    Public Information Repository. Journal of Digital Imaging, 26(6), 1045–1057.
    `https://doi.org/10.1007/s10278-013-9622-7 <https://doi.org/10.1007/s10278-013-9622-7>`_.