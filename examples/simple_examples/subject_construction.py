from argparse import ArgumentParser
import os

import SimpleITK as sitk

from pyradise.data import Subject, IntensityImage, SegmentationImage, Modality, Organ, Rater
from pyradise.serialization import SubjectWriter


def main(input_dir: str,
         output_dir: str
         ) -> None:
    # Retrieve image file paths
    segmentation_file_paths = [entry.path for entry in os.scandir(input_dir)
                               if 'seg_' in entry.name and '.nii.gz' in entry.name]
    intensity_file_paths = [entry.path for entry in os.scandir(input_dir)
                            if 'img_' in entry.name and '.nii.gz' in entry.name]

    # Load the segmentation image files
    images = []
    organs = [Organ('Brain'), Organ('Lung'), Organ('Liver'), Organ('Heart')]
    raters = [Rater('RaterName'), Rater('RaterName'), Rater('RaterName'), Rater('RaterName')]
    for path, organ, rater in zip(segmentation_file_paths, organs, raters):
        image = SegmentationImage(sitk.ReadImage(path, sitk.sitkUInt8), organ, rater)
        images.append(image)

    # Load the intensity image files
    modalities = [Modality.CT, Modality.T1c, Modality.T1w, Modality.T2w]
    for path, modality in zip(intensity_file_paths, modalities):
        image = IntensityImage(sitk.ReadImage(path, sitk.sitkFloat32), modality)
        images.append(image)

    # Construct the subject
    subject = Subject('subject_1', images)

    # Display the subject name and properties of the intensity and segmentation images
    print(f'Subject {subject.get_name()} contains the following images:')

    for image in subject.intensity_images:
        print(f'Intensity image of modality {image.get_modality(True)} with size: {image.get_size()}')

    for image in subject.segmentation_images:
        print(f'Segmentation image of {image.get_organ(True)} with size: {image.get_size()}')

    # Write the subject to disk
    SubjectWriter().write(output_dir, subject, write_transforms=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-input_dir', type=str)
    parser.add_argument('-output_dir', type=str)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
