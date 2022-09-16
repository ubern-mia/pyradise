import os
from typing import (
    Any,
    Dict,
    Optional)

import SimpleITK as sitk
import numpy as np

from pyradise.data import Modality, IntensityImage
from pyradise.fileio import (IterableDicomCrawler, ModalityExtractor, SubjectWriter,
                             SubjectLoader, SubjectDicomCrawler, Tag)


class ExampleModalityExtractor(ModalityExtractor):

    @staticmethod
    def _get_mr_modality(dataset_dict: Dict[str, Any]) -> Optional[Modality]:
        # check for different variants of attributes to get the sequence identification
        scanning_sequence = dataset_dict.get('Scanning Sequence', {}).get('value', [])
        scanning_sequence = [scanning_sequence] if isinstance(scanning_sequence, str) else scanning_sequence
        contrast_bolus = dataset_dict.get('Contrast/Bolus Agent', {}).get('value', '')

        if all(entry in scanning_sequence for entry in ('SE', 'IR')):
            return Modality('FLAIR')
        elif all(entry in scanning_sequence for entry in ('GR', 'IR')) and len(contrast_bolus) > 0:
            return Modality('T1c')
        elif all(entry in scanning_sequence for entry in ('GR', 'IR')) and len(contrast_bolus) == 0:
            return Modality('T1w')
        elif all(entry == 'SE' for entry in scanning_sequence):
            return Modality('T2w')
        else:
            return None

    def extract_from_dicom(self, path: str) -> Optional[Modality]:
        # extract the necessary attributes from the file
        tags = (Tag(0x0008, 0x0060),  # Modality
                Tag(0x0018, 0x0010),  # ContrastBolusAgent
                Tag(0x0018, 0x0020))  # ScanningSequence
        dataset_dict = self._load_dicom_attributes(tags, path)

        # identify the modality
        extracted_modality = dataset_dict.get('Modality', {}).get('value', None)
        if extracted_modality == 'CT':
            return Modality('CT')
        elif extracted_modality == 'MR':
            return self._get_mr_modality(dataset_dict)
        else:
            return None

    def extract_from_path(self, path: str) -> Optional[Modality]:
        file_name = os.path.basename(path)
        if 'T1c' in file_name:
            return Modality('T1c')
        elif 'T1w' in file_name:
            return Modality('T1w')
        elif 'T2w' in file_name:
            return Modality('T2w')
        elif 'FLAIR' in file_name:
            return Modality('FLAIR')
        elif 'CT' in file_name:
            return Modality('CT')
        else:
            return None


def main():
    input_path = 'D:/temp/dicom_test_data_v2/'
    output_path = 'D:/temp/test_data_output'

    crawler = IterableDicomCrawler(input_path, ExampleModalityExtractor())

    writer = SubjectWriter()

    for subject_info in crawler:
        if int(subject_info[0].patient_name.split('_')[-1]) >= 43:
            subject = SubjectLoader().load(subject_info)
            print(subject.name)
            writer.write_to_subject_folder(output_path, subject, False)


def main2():

    for i in range(48, 99):
        input_path = f'D:/temp/dicom_test_data_v2/ISAS_GBM_0{i}'
        output_path = 'D:/temp/test_data_output'

        writer = SubjectWriter()
        crawler = SubjectDicomCrawler(input_path, ExampleModalityExtractor())
        info = crawler.execute()

        subject = SubjectLoader().load(info)
        print(subject.name)
        # writer.write_to_subject_folder(output_path, subject, False)

        image_sitk = sitk.GetImageFromArray(np.ones((10, 10, 10), dtype=np.float32))
        new_image = IntensityImage(image_sitk, Modality('T1c'))

        # print(f'New image: {new_image}')
        # print(f'Old image: {subject.intensity_images[0]}')

        # subject.replace_image_(new_image, subject.intensity_images[0])
        subject.replace_image(new_image)

        # subject.remove_image(new_image)
        # subject.remove_image_by_modality('CT')
        # subject.remove_image_by_organ('Edema')

        subject.add_image(new_image)


        print(subject.name)


if __name__ == '__main__':
    # main()
    main2()
