TCIA Glioma Image Segmentation for Radiotherapy (GLIS-RT) Dataset Tutorial
==========================================================================

The `Glioma Image Segmentation for Radiotherapy (GLIS-RT) <https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=95224486>`_
dataset contains a collection of 230 cases of glioblastoma and low-grade glioma patients treated with surgery and
adjuvant radiotherapy at Massachusetts General Hospital. The patients underwent routine post-surgical MRI examination
by acquiring two MR sequences, contrast-enhanced 3D-T1 and 2D multislice-T2 FLAIR, required to define target volumes
for radiotherapy treatment. CT scans were acquired after diagnostic imaging to use in radiotherapy treatment planning.
All cases in the image set are provided with the radiotherapy targets, gross tumor volume (GTV), and clinical target
volume (CTV) manually delineated by the treating radiation oncologist. The set includes glioblastoma (GBM) - 198 cases,
anaplastic astrocytoma (AAC) - 23 cases, astrocytoma (AC) - 5 cases, anaplastic oligodendroglioma (AODG) - 2 cases, and
oligodendroglioma (ODG) - 2 cases.

The following tutorial demonstrates converting the DICOM-based GLIS-RT dataset to the NIfTI format.

.. note::

    The following tutorial presumes that PyRaDiSe is installed. If not, heed the :ref:`installation section's <installation_section>`
    instructions or just run ``pip install pyradise`` in the appropriate terminal session.


Download the Dataset
--------------------

Downloading datasets from the Tumor Cancer Imaging Archive (TCIA) requires the installation of the NBIA Data Receiver
from `here <https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+the+NBIA+Data+Retriever+7.7>`_. After
installation, the GLIS-RT dataset can be downloaded by downloading the corresponding NBIA Data Receiver file from the
`TCIA website <https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=95224486>`_.

.. attention::

    Access to the dataset requires registration at the TCIA website before starting the download process.

.. note::

    Be aware that the download procedure may take hours, depending on the internet connection.


After the download procedure is finished successfully, one should have a data folder (i.e., dataset_glis) containing
the following sub-structure:

.. code-block:: bash

    dataset_glis
    └── manifest-1636603674498
        ├── metadata.csv
        └── GLIS-RT
            ├── GLI_001_GBM
            │   └── <subject-specific folder structure>
            ├── GLI_002_GBM
            │   └── <subject-specific folder structure>
            └── ...


Generating the Modality Configuration Files
-------------------------------------------

PyRaDiSe provides multiple approaches for identifying (uni-modal) image series in a dataset. Because the conversion of
a dataset may be performed multiple times with different pre-processing procedures, using modality configuration files
for discriminating image series is best suited. So, the next step is the generation of these modality configuration
files for each subject in the dataset. To accomplish this, one needs to write a script reading the content of the
`metadata.csv` file and generating the appropriate modality configuration files.

The following script performs the modality configuration generation for the GLIS-RT dataset:

.. code-block:: python

    import typing as t
    import os
    import csv

    from pyradise.fileio import ModalityConfiguration


    class GLISRTModalityConfigGenerator:

        def __init__(self, manifest_dir_path: str):
            self.manifest_dir_path = manifest_dir_path

        def _read_meta_data(self) -> t.Dict[str, t.List[t.Dict[str, t.Any]]]:
            # read the meta data csv
            meta_data_path = os.path.join(self.manifest_dir_path, 'metadata.csv')
            with open(meta_data_path, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                raw_data_meta_data = list(reader)

            # sort the entries
            data = {}
            header_meta_data = raw_data_meta_data[0]
            for entry in raw_data_meta_data[1:]:
                entry_data = {}

                for identifier, value in zip(header_meta_data, entry):
                    entry_data[identifier] = value

                if data.get(entry_data['Subject ID'], None) is None:
                    data[entry_data['Subject ID']] = []
                data[entry_data['Subject ID']].append(entry_data)

            return data

        @staticmethod
        def _extract_sequence_info(meta_data: t.Dict[str, t.List[t.Dict[str, t.Any]]]
                                   ) -> t.Dict[str, t.List[t.Dict[str, t.Any]]]:
            # extract the sequence info for the MR images and the CT image
            for subject, entry in meta_data.items():
                for entity_entry in entry:
                    if entity_entry.get('SOP Class Name') == 'MR Image Storage':
                        description = entity_entry.get('Series Description')

                        if any(x in description.upper() for x in ('FLAIR', 'T2', 'AX 3MM')):
                            entity_entry['AssignedModality'] = 'T2'
                        else:
                            entity_entry['AssignedModality'] = 'T1'

                    if entity_entry.get('SOP Class Name') == 'CT Image Storage':
                        entity_entry['AssignedModality'] = 'CT'

            return meta_data

        def _generate_modality_config(self,
                                      meta_data: t.Dict[str, t.List[t.Dict[str, t.Any]]]
                                      ) -> None:
            # get the subject base directory
            subject_base_path = [folder for folder in os.scandir(self.manifest_dir_path)
                                 if folder.is_dir()][0].path

            # get all the subject directories
            subject_paths = [folder for folder in os.scandir(subject_base_path)
                             if folder.is_dir()]

            # loop through the subject directories
            for subject_path in subject_paths:
                print(f'Generating modality config for subject {subject_path.name}...')

                # get the subject-associated metadata
                subject_data = meta_data.get(subject_path.name)

                # build the modality config
                config = ModalityConfiguration()
                for entry in subject_data:

                    if entry.get('SOP Class Name') not in ('MR Image Storage', 'CT Image Storage'):
                        continue

                    config.add_modality_entry(
                        sop_class_uid=entry.get('SOP Class UID'),
                        study_instance_uid=entry.get('Study UID'),
                        series_instance_uid=entry.get('Series UID'),
                        series_description=entry.get('Series Description'),
                        series_number='1',
                        dicom_modality=entry.get('SOP Class Name', ' ').split(' ')[0],
                        modality=entry.get('AssignedModality')
                    )

                # write the modality config file
                modality_config_path = os.path.join(subject_path.path, 'modality_config.json')
                config.to_file(modality_config_path, True)

        def generate(self) -> None:
            # read the meta data
            print('Reading the meta data file...')
            meta_data = self._read_meta_data()

            # extract the sequence info
            print('Extracting and combining the sequence info...')
            meta_data = self._extract_sequence_info(meta_data)

            # generate the modality config
            print('Generating the modality config files...')
            self._generate_modality_config(meta_data)


    if __name__ == '__main__':
        meta_data_path_ = r'PATH_TO_THE_MANIFEST_FOLDER'

        GLISRTModalityConfigGenerator(meta_data_path_).generate()

After the modality configuration files are generated, the folder structure should look as follows:

.. code-block:: bash

    dataset_glis
    └── manifest-1636603674498
        ├── metadata.csv
        └── GLIS-RT
            ├── GLI_001_GBM
            │   ├── modality_config.json
            │   └── <subject-specific folder structure>
            ├── GLI_002_GBM
            │   ├── modality_config.json
            │   └── <subject-specific folder structure>
            └── ...


Converting the Dataset
----------------------

Now the dataset is prepared for implementing the conversion script. This tutorial demonstrates a minimal conversion
procedure that incorporates the co-registration of the intra-subject image series using the downloaded DICOM
registration files. However, due to inappropriate references in the DICOM registration files that do not match the
provided DICOM image series, the co-registration procedures are not executed automatically by PyRaDiSe and require some
additional code. This additional code extracts the registration matrices from the appropriate DICOM registration
files and applies them during the conversion procedure. The following script performs this task and is stored as
a separate file called `registration.py`:

.. code-block:: python

    import typing as t

    import numpy as np
    import SimpleITK as sitk
    import pyradise.fileio as fio
    import pyradise.process as proc
    import pyradise.data as dat


    class RegistrationFilterParams(proc.FilterParams):

        def __init__(self,
                     registration_infos: t.Sequence[fio.DicomSeriesRegistrationInfo],
                     ) -> None:
            # extract the transformation matrices from the DICOM registration files
            self.transforms = {}
            for info in registration_infos:
                if 'T1/T2' not in info.series_description.upper():
                    target, transform = self._get_transform_matrix(info)
                    self.transforms[target] = transform

            # store the reference modality to which the registration is performed
            self.reference_modality = 'CT'

        @staticmethod
        def _get_transform_matrix(info: fio.DicomSeriesRegistrationInfo
                                  ) -> t.Tuple[str, sitk.Transform]:
            # get the DICOM dataset from the DicomSeriesRegistrationInfo
            dataset = info.dataset

            # extract the modality to which the transformation should be applied
            target = dataset.SeriesDescription.split('/')[-1]

            # search the transformation matrix
            transform_np = None
            for item_0 in dataset.RegistrationSequence:
                for item_1 in item_0.MatrixRegistrationSequence:
                    for item_2 in item_1.MatrixSequence:
                        transform_np = np.array(item_2.FrameOfReferenceTransformationMatrix).reshape(4, 4)
                        if not np.any(transform_np == np.eye(4)):
                            break
                        break
                    break

            # generate a SimpleITK transform from the transformation matrix
            transform_obj = sitk.AffineTransform(3)
            transform_obj.SetMatrix(transform_np[:3, :3].flatten())
            transform_obj.SetTranslation(transform_np[:3, 3])
            return target, transform_obj


    class RegistrationFilter(proc.Filter):

        @staticmethod
        def is_invertible() -> bool:
            return False

        def _apply_transform(self,
                             subject: dat.Subject,
                             target: str,
                             transform: sitk.Transform,
                             params: RegistrationFilterParams
                             ) -> dat.Subject:
            # get the reference image
            ref_image = subject.get_image_by_modality(params.reference_modality)
            ref_image_sitk = ref_image.get_image_data()

            # get the target image
            target_image = subject.get_image_by_modality(target)
            pre_image_sitk = target_image.get_image_data()
            min_intensity = float(np.min(sitk.GetArrayFromImage(pre_image_sitk)))

            # apply the transform
            post_image_sitk = sitk.Resample(pre_image_sitk,
                                            ref_image_sitk,
                                            transform.GetInverse(),
                                            sitk.sitkBSpline,
                                            min_intensity)
            target_image.set_image_data(post_image_sitk)

            # track the registration transformation on the transform tape
            self._register_tracked_data(target_image, pre_image_sitk, post_image_sitk, None, transform)

            return subject

        def execute(self,
                    subject: dat.Subject,
                    params: t.Optional[RegistrationFilterParams]
                    ) -> dat.Subject:

            # apply the registration matrices to the corresponding images
            for target, transform in params.transforms.items():
                subject = self._apply_transform(subject, target, transform, params)

            return subject

        def execute_inverse(self,
                            subject: dat.Subject,
                            transform_info: dat.TransformInfo,
                            target_image: t.Optional[t.Union[dat.SegmentationImage, dat.IntensityImage]] = None
                            ) -> dat.Subject:
            # The registration is invertible. However, because we use this filter only for conversion
            # we do not implement the inverse transform and return the original subject.
            return subject


After implementing the registration filter and its parameter class, the conversion script is created. Because the
GLIS-RT dataset contains two DICOM image series that incorporate errors (i.e., inappropriate size and spacing of the
last image slice), the conversion script also must implement a routine that deletes those files automatically (see
method `remove_erroneous_data`).

In addition, the application of the newly implemented registration filter requires separating the pre-loading
information (i.e., :class:`~pyradise.fileio.series_info.SeriesInfo`) into two (i.e., registration info entries and
non-registration info entries) before loading the data to work around the aforementioned DICOM registration file errors.
Due to the separation of the pre-loading information, the new registration filter can use the pre-loaded DICOM
attributes for registration, what circumvents searching for the appropriate DICOM registration files again.

The remaining part of the conversion script is similar to other conversion scripts on this website. It thus
allows for extensive customization and extension (e.g., image resampling to a common size and spacing, and image
reorientation). The following code snipped demonstrates the basic conversion procedure:

.. code-block:: python

    import os
    import typing as t

    import pyradise.fileio as fio

    from registration import RegistrationFilterParams, RegistrationFilter


    def split_series_info_entries(infos: t.Sequence[fio.SeriesInfo]
                                  ) -> t.Tuple[t.Tuple[fio.SeriesInfo, ...], t.Tuple[fio.DicomSeriesRegistrationInfo, ...]]:
        reg_infos = []
        remaining_infos = []

        for info in infos:
            if isinstance(info, fio.DicomSeriesRegistrationInfo):
                reg_infos.append(info)
            else:
                remaining_infos.append(info)

        return tuple(remaining_infos), tuple(reg_infos)


    def remove_erroneous_data(input_dir: str) -> None:
        # files to remove
        files = (
            # remove one DICOM image file for GLI_101_GBM due to inappropriate properties
            os.path.join(input_dir,
                         'GLI_101_GBM/02-13-2007-NA-MRIBRNWWO-27331/'
                         '23.000000-AX RFT MPRAGE POST-58935/1-182.dcm'),
            # remove one DICOM image file for GLI_183_AAC due to inappropriate properties
            os.path.join(input_dir,
                         'GLI_183_AAC/11-27-2006-NA-MRIBRNWWO-29502/'
                         '12.000000-AX SPACE FLAIR REFORMAT-57013/1-183.dcm'),
        )

        for file in files:
            if os.path.exists(file):
                os.remove(file)


    def main(input_dir: str,
             output_dir: str
             ) -> None:
        # remove erroneous data
        remove_erroneous_data(input_dir)

        # get the subject list
        subjects = [subject for subject in os.scandir(input_dir) if subject.is_dir()]

        # loop through the subjects
        for i, subject in enumerate(subjects):
            print(f'[{i} / {len(subjects)}] Processing subject {subject.name}...')

            # crawl for the subject data
            crawler = fio.SubjectDicomCrawler(subject.path)
            series_info = crawler.execute()

            # split the series info into registration and non-registration
            series_info, reg_infos = split_series_info_entries(series_info)

            # load subject
            subject = fio.SubjectLoader().load(series_info)

            # perform the registration
            reg_params = RegistrationFilterParams(reg_infos)
            reg_filter = RegistrationFilter()
            subject = reg_filter.execute(subject, reg_params)

            # add a filter pipeline for adjusting the data properties,
            # such as image size, spacing, orientation, etc.

            # write the subject to nifti
            writer = fio.SubjectWriter()
            writer.write_to_subject_folder(output_dir, subject, True)


    if __name__ == '__main__':
        input_path = r'PATH_TO_THE_FOLDER_CONTAINING_THE_SUBJECTS'
        output_path = r'PATH_TO_AN_EMPTY_OUTPUT_FOLDER'

        main(input_path, output_path)


Congratulations :) You have successfully converted the GLIS-RT dataset to the NIfTI format.
