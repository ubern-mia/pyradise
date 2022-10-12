from argparse import ArgumentParser
from typing import Tuple
import os

import SimpleITK as sitk

from pyradise.data import Subject, IntensityImage, SegmentationImage, Modality, Organ, Annotator
from pyradise.fileio import SubjectWriter, ImageFileFormat


def get_segmentation_file_paths(path: str,
                                valid_organs: Tuple[Organ, ...]
                                ) -> Tuple[str]:
    file_paths = []

    for file in os.listdir(path):
        if any(organ.get_name() in file for organ in valid_organs) and file.endswith('.nii.gz'):
            file_paths.append(os.path.join(path, file))

    return tuple(sorted(file_paths))


def get_intensity_file_paths(path: str,
                             valid_modalities: Tuple[Modality, ...]
                             ) -> Tuple[str]:
    file_paths = []

    for file in os.listdir(path):
        if any(modality.get_name() in file for modality in valid_modalities) and file.endswith('.nii.gz'):
            file_paths.append(os.path.join(path, file))

    return tuple(sorted(file_paths))


def main(input_dir: str,
         output_dir: str
         ) -> None:
    # Retrieve image file paths
    organs = (Organ('Brainstem'), Organ('Eyes'), Organ('Hippocampi'), Organ('OpticNerves'))
    modalities = (Modality('CT'), Modality('T1c'), Modality('T1w'), Modality('T2w'))

    segmentation_file_paths = get_segmentation_file_paths(input_dir, organs)
    intensity_file_paths = get_intensity_file_paths(input_dir, modalities)

    # Load the segmentation image files
    images = []
    for path, organ in zip(segmentation_file_paths, organs):
        image = SegmentationImage(sitk.ReadImage(path, sitk.sitkUInt8), organ, Annotator.get_default())
        images.append(image)

    # Load the intensity image files
    for path, modality in zip(intensity_file_paths, modalities):
        image = IntensityImage(sitk.ReadImage(path, sitk.sitkFloat32), modality)
        images.append(image)

    # Construct the subject
    subject = Subject(os.path.basename(input_dir), images)

    # Display the subject name and properties of the intensity and segmentation images
    print(f'Subject {subject.get_name()} contains the following images:')

    for image in subject.intensity_images:
        print(f'Intensity image of modality {image.get_modality(True)} with size: {image.get_size()}')

    for image in subject.segmentation_images:
        print(f'Segmentation image of {image.get_organ(True)} with size: {image.get_size()}')

    # Write the subject to disk
    SubjectWriter(ImageFileFormat.NRRD).write(output_dir, subject, write_transforms=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-input_dir', type=str)
    parser.add_argument('-output_dir', type=str)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
