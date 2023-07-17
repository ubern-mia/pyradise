import os

import pytest
import SimpleITK as sitk

from pyradise.data import (
    ImageProperties,
    IntensityImage,
    SegmentationImage,
    Subject,
    TransformInfo,
)
from pyradise.fileio.writing import ImageFileFormat, SubjectWriter
from tests.conftest import get_sitk_image

sitk_img_1 = get_sitk_image(seed=0, low=0, high=101, meta="nii")
sitk_seg_1 = get_sitk_image(seed=0, low=0, high=2, meta="nii")


def test__init__():
    subject_writer = SubjectWriter()
    assert subject_writer.image_file_format == ImageFileFormat.NIFTI_GZ
    assert subject_writer.allow_override is False


def test__generate_image_file_name_1(img_file_nii):
    subject_writer = SubjectWriter()
    subject = Subject("subject")
    image = IntensityImage(sitk_img_1, "modality")
    assert subject_writer._generate_image_file_name(subject, image, False) == "img_subject_modality"


def test__generate_image_file_name_2(seg_file_nii):
    subject_writer = SubjectWriter()
    subject = Subject("subject")
    image = SegmentationImage(sitk.ReadImage(seg_file_nii), "organ", "annotation")
    assert subject_writer._generate_image_file_name(subject, image, False) == "seg_subject_annotation_organ"


def test__generate_image_file_name_3(img_file_nii):
    subject_writer = SubjectWriter()
    subject = Subject("subject")
    image = IntensityImage(sitk_img_1, "modality")
    assert subject_writer._generate_image_file_name(subject, image, True) == "img_subject_modality.nii.gz"


def test__generate_image_file_name_4(seg_file_nii):
    subject_writer = SubjectWriter()
    subject = Subject("subject")
    image = SegmentationImage(sitk_seg_1, "organ", "annotation")
    assert subject_writer._generate_image_file_name(subject, image, True) == "seg_subject_annotation_organ.nii.gz"


def test__generate_image_file_name_5():
    subject_writer = SubjectWriter()
    subject = Subject("subject")
    with pytest.raises(ValueError):
        subject_writer._generate_image_file_name(subject, None, False)


def test__generate_transform_file_name_1():
    subject_writer = SubjectWriter()
    subject = Subject("subject")
    image = IntensityImage(sitk_img_1, "modality")
    assert subject_writer._generate_transform_file_name(subject, image, 0, ".tfm") == "tfm_img_subject_modality_0.tfm"
    assert isinstance(subject_writer._generate_transform_file_name(subject, image, 0, ".tfm"), str)


def test__generate_transform_file_name_2():
    subject_writer = SubjectWriter()
    subject = Subject("subject")
    image = SegmentationImage(sitk_seg_1, "organ", "annotation")
    assert (
        subject_writer._generate_transform_file_name(subject, image, -1, ".tfm")
        == "tfm_seg_subject_annotation_organ_-1.tfm"
    )
    assert isinstance(subject_writer._generate_transform_file_name(subject, image, -1, ".tfm"), str)


def test__generate_transform_file_name_3():
    subject_writer = SubjectWriter()
    subject = Subject("subject")
    with pytest.raises(ValueError):
        subject_writer._generate_transform_file_name(subject, None, 1, ".tfm")


def test__check_file_path_1():
    subject_writer = SubjectWriter(allow_override=False)
    assert subject_writer._check_file_path("fantasy_path/fantasy_file.nii.gz") is None


def test__check_file_path_2(img_file_nii):
    subject_writer = SubjectWriter(allow_override=False)
    with pytest.raises(FileExistsError):
        subject_writer._check_file_path(img_file_nii)


def test__check_file_path_3(img_file_nii):
    subject_writer = SubjectWriter(allow_override=True)
    assert subject_writer._check_file_path(img_file_nii) is None


def test_write_1(empty_folder):
    subject_writer = SubjectWriter()
    subject = Subject("subject")
    with pytest.raises(NotADirectoryError):
        subject_writer.write("fantasy_path", subject, False)


def test_write_2(empty_folder):
    subject_writer = SubjectWriter()
    image = IntensityImage(sitk_img_1, "modality")
    seg = SegmentationImage(sitk_seg_1, "organ", "annotation")
    subject = Subject("subject", [image, seg])
    subject_writer.write(empty_folder, subject, False)
    assert os.path.exists(os.path.join(empty_folder, "img_subject_modality.nii.gz"))
    assert os.path.exists(os.path.join(empty_folder, "seg_subject_annotation_organ.nii.gz"))
    assert sitk.ReadImage(os.path.join(empty_folder, "img_subject_modality.nii.gz")) == image.image
    assert sitk.ReadImage(os.path.join(empty_folder, "seg_subject_annotation_organ.nii.gz")) == seg.image
    assert "tfm" not in os.listdir(empty_folder)


def test_write_3(empty_folder):
    subject_writer = SubjectWriter()
    image = IntensityImage(sitk_img_1, "modality")
    image_property = ImageProperties(sitk_img_1)
    info = TransformInfo("trans_name", "filter", image_property, image_property)
    image.add_transform_info(info)
    seg = SegmentationImage(sitk_seg_1, "organ", "annotation")
    seg.add_transform_info(info)
    subject = Subject("subject", [image, seg])
    subject_writer.write(empty_folder, subject, True)
    assert os.path.exists(os.path.join(empty_folder, "img_subject_modality.nii.gz"))
    assert os.path.exists(os.path.join(empty_folder, "seg_subject_annotation_organ.nii.gz"))
    assert sitk.ReadImage(os.path.join(empty_folder, "img_subject_modality.nii.gz")) == image.image
    assert sitk.ReadImage(os.path.join(empty_folder, "seg_subject_annotation_organ.nii.gz")) == seg.image
    assert os.path.exists(os.path.join(empty_folder, "tfm_img_subject_modality_0.tfm"))
    assert os.path.exists(os.path.join(empty_folder, "tfm_seg_subject_annotation_organ_0.tfm"))


def test_write_to_subject_folder_1(empty_folder):
    subject_writer = SubjectWriter()
    subject = Subject("subject")
    subject_writer.write_to_subject_folder(empty_folder, subject, False)
    assert os.path.exists(os.path.join(empty_folder, "subject"))


def test_write_to_subject_folder_2(empty_folder):
    subject_writer = SubjectWriter()
    subject = Subject("subject")
    os.makedirs(os.path.join(empty_folder, "subject"))
    with pytest.raises(FileExistsError):
        subject_writer.write_to_subject_folder(empty_folder, subject, False)
