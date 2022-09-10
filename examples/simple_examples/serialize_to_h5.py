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
    image_identifiers = {'T1c': Modality('T1c'),
                         'T1w': Modality('T1w'),
                         'T2w': Modality('T2w'),
                         'FLAIR': Modality('FLAIR')}

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
