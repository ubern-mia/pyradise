from collections import abc as col_abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np
import SimpleITK as sitk

np.random.seed(42)

from pyradise.data.annotator import Annotator
from pyradise.data.image import Image, IntensityImage, SegmentationImage
from pyradise.data.modality import Modality
from pyradise.data.organ import Organ
from pyradise.data.subject import Subject
from tests.unit.helpers.image_helpers import (
    get_sitk_intensity_image,
    get_sitk_segmentation_image,
)

img_1 = IntensityImage(get_sitk_intensity_image(1), Modality("modality_1"))
img_2 = IntensityImage(get_sitk_intensity_image(2), Modality("modality_2"))
seg_1 = SegmentationImage(
    get_sitk_segmentation_image(3), Organ("organ_1"), Annotator("annotator_1")
)
seg_2 = SegmentationImage(
    get_sitk_segmentation_image(4), Organ("organ_2"), Annotator("annotator_2")
)


def test__init__1():
    s = Subject("subject")
    assert s.name == "subject"
    assert s.intensity_images == []
    assert s.segmentation_images == []
    assert s.data == {}


def test__init__2():
    data = {"test_1": "test_1", "test_2": None, "test_3": {"test_3_1": "test_3_1"}}
    s = Subject("subject", [img_1, img_2, seg_1, seg_2], data)

    assert s.name == "subject"
    assert s.intensity_images == [img_1, img_2]
    assert s.segmentation_images == [seg_1, seg_2]
    assert s.data == data


def test_check_for_single_candidate():
    s = Subject("subject")
    assert (
        s._check_for_single_candidate([], "test_1", return_first_on_multiple=False)
        is None
    )
    assert (
        s._check_for_single_candidate([1], "test_1", return_first_on_multiple=False)
        == 1
    )
    assert (
        s._check_for_single_candidate([2, 3], "test_2", return_first_on_multiple=True)
        == 2
    )


def test_get_name():
    s = Subject("subject")
    assert s.get_name() == "subject"
    assert isinstance(s.get_name(), str)


def test_get_modalities():
    s = Subject("subject", [img_1, img_2, seg_1, seg_2])
    assert s.get_modalities() == (Modality("modality_1"), Modality("modality_2"))
    assert isinstance(s.get_modalities(), tuple)
    assert s.get_modalities(as_str=True) == ("modality_1", "modality_2")
    assert isinstance(s.get_modalities(as_str=True), tuple)


def test_get_organs():
    s = Subject("subject", [img_1, img_2, seg_1, seg_2])
    assert s.get_organs() == (Organ("organ_1"), Organ("organ_2"))
    assert isinstance(s.get_organs(), tuple)
    assert s.get_organs(as_str=True) == ("organ_1", "organ_2")
    assert isinstance(s.get_organs(as_str=True), tuple)


def test_get_annotators():
    s = Subject("subject", [img_1, img_2, seg_1, seg_2])
    assert s.get_annotators() == (Annotator("annotator_1"), Annotator("annotator_2"))
    assert isinstance(s.get_annotators(), tuple)
    assert s.get_annotators(as_str=True) == ("annotator_1", "annotator_2")
    assert isinstance(s.get_annotators(as_str=True), tuple)


def test_add_image_1():
    s = Subject("subject")
    s.add_image(img_1)
    s.add_image(seg_1)
    s.add_image(img_2)
    s.add_image(seg_2)
    assert s.intensity_images == [img_1, img_2]
    assert s.segmentation_images == [seg_1, seg_2]
    assert s.get_modalities() == (Modality("modality_1"), Modality("modality_2"))
    assert s.get_organs() == (Organ("organ_1"), Organ("organ_2"))
    assert s.get_annotators() == (Annotator("annotator_1"), Annotator("annotator_2"))


def test_add_image_2():
    s = Subject("subject")
    s.add_image(img_1)
    s.add_image(img_1, force=True)

    s.add_image(seg_1)
    s.add_image(seg_1, force=True)
    s.add_image(seg_1, force=True)

    assert s.intensity_images == [img_1, img_1]
    assert s.segmentation_images == [seg_1, seg_1, seg_1]


def test_add_images_1():
    s = Subject("subject")
    s.add_images([img_1, img_2, seg_1, seg_2])
    assert s.intensity_images == [img_1, img_2]
    assert s.segmentation_images == [seg_1, seg_2]


def test_add_images_2():
    s = Subject("subject")
    s.add_images([img_1, img_1, seg_1, seg_1, seg_1], force=True)
    assert s.intensity_images == [img_1, img_1]
    assert s.segmentation_images == [seg_1, seg_1, seg_1]


def test_get_images():
    s = Subject("subject")
    s.add_images([img_1, img_2, seg_1, seg_2])
    assert s.get_images() == [img_1, img_2, seg_1, seg_2]
    assert isinstance(s.get_images(), list)


def test_get_image_by_modality():
    s = Subject("subject")
    s.add_images([img_1, img_1, img_2, seg_1, seg_1, seg_2], force=True)
    assert s.get_image_by_modality("modality_2") == img_2
    assert s.get_image_by_modality(Modality("modality_2")) == img_2
    assert s.get_image_by_modality("modality_1", return_first_on_multiple=True) == img_1


def test_get_image_by_organ():
    s = Subject("subject")
    s.add_images([img_1, img_1, img_2, seg_1, seg_1, seg_2], force=True)
    assert s.get_image_by_organ("organ_2") == seg_2
    assert s.get_image_by_organ(Organ("organ_2")) == seg_2
    assert s.get_image_by_organ("organ_1", return_first_on_multiple=True) == seg_1


def test_get_images_by_annotator():
    s = Subject("subject")
    s.add_images([img_1, img_1, img_2, seg_1, seg_1, seg_2], force=True)
    assert isinstance(s.get_images_by_annotator("annotator_2"), tuple)
    assert s.get_images_by_annotator("annotator_2") == (seg_2,)
    assert s.get_images_by_annotator(Annotator("annotator_2")) == (seg_2,)
    assert s.get_images_by_annotator("annotator_1") == (seg_1, seg_1)


def test_get_image_by_organ_and_annotator():
    s = Subject("subject")
    s.add_images([img_1, img_1, img_2, seg_1, seg_1, seg_2], force=True)
    assert isinstance(
        s.get_image_by_organ_and_annotator("organ_2", "annotator_2"), SegmentationImage
    )
    assert s.get_image_by_organ_and_annotator("organ_2", "annotator_2") == seg_2
    assert (
        s.get_image_by_organ_and_annotator(Organ("organ_2"), Annotator("annotator_2"))
        == seg_2
    )
    assert (
        s.get_image_by_organ_and_annotator(
            "organ_1", "annotator_1", return_first_on_multiple=True
        )
        == seg_1
    )


def test_get_images_by_type():
    s = Subject("subject")
    s.add_images([img_1, img_1, img_2, seg_1, seg_1, seg_2], force=True)
    assert isinstance(s.get_images_by_type(IntensityImage), list)
    assert s.get_images_by_type(IntensityImage) == [img_1, img_1, img_2]
    assert s.get_images_by_type(SegmentationImage) == [seg_1, seg_1, seg_2]


def test_replace_image():
    s = Subject("subject")

    s.add_images([img_1, img_1, img_2, seg_1, seg_1, seg_2], force=True)
    s.replace_image(new_image=img_1, old_image=img_2)
    assert s.get_images_by_type(IntensityImage) == [img_1, img_1, img_1]
    s.replace_image(new_image=img_2, old_image=img_1)
    assert s.get_images_by_type(IntensityImage) == [img_2, img_1, img_1]

    s.replace_image(new_image=seg_1, old_image=seg_2)
    assert s.get_images_by_type(SegmentationImage) == [seg_1, seg_1, seg_1]
    s.replace_image(new_image=seg_2, old_image=seg_1)
    assert s.get_images_by_type(SegmentationImage) == [seg_2, seg_1, seg_1]


def test_remove_image_by_modality():
    s = Subject("subject")
    s.add_images([img_1, img_1, img_2, seg_1, seg_1, seg_2], force=True)
    s.remove_image_by_modality("modality_1")
    assert s.get_images_by_type(IntensityImage) == [img_2]
    assert s.get_images_by_type(SegmentationImage) == [seg_1, seg_1, seg_2]


def test_remove_image_by_organ():
    s = Subject("subject")
    s.add_images([img_1, img_1, img_2, seg_1, seg_1, seg_2], force=True)
    s.remove_image_by_organ("organ_1")
    assert s.get_images_by_type(IntensityImage) == [img_1, img_1, img_2]
    assert s.get_images_by_type(SegmentationImage) == [seg_2]


def test_remove_image_by_annotator():
    s = Subject("subject")
    s.add_images([img_1, img_1, img_2, seg_1, seg_1, seg_2], force=True)
    s.remove_image_by_annotator("annotator_1")
    assert s.get_images_by_type(IntensityImage) == [img_1, img_1, img_2]
    assert s.get_images_by_type(SegmentationImage) == [seg_2]


def test_remove_image_by_organ_and_annotator():
    s = Subject("subject")
    s.add_images([img_1, img_1, img_2, seg_1, seg_1, seg_2], force=True)
    s.remove_image_by_organ_and_annotator("organ_1", "annotator_1")
    assert s.get_images_by_type(IntensityImage) == [img_1, img_1, img_2]
    assert s.get_images_by_type(SegmentationImage) == [seg_2]


def test_remove_image():
    s = Subject("subject")
    s.add_images([img_1, img_2, seg_1, seg_2], force=True)
    assert isinstance(s.remove_image(img_1), bool)
    assert s.get_images_by_type(IntensityImage) == [img_2]
    assert s.get_images_by_type(SegmentationImage) == [seg_1, seg_2]
    s.remove_image(seg_1)
    assert s.get_images_by_type(IntensityImage) == [img_2]
    assert s.get_images_by_type(SegmentationImage) == [seg_2]


def test_add_data():
    s = Subject("subject")
    data = {"a": 1, "b": 2}
    s.add_data(data)
    assert s.data == data


def test_add_data_by_key():
    s = Subject("subject")
    data = {"a": 1, "b": 2}
    s.add_data_by_key("a", 1)
    assert s.data == {"a": 1}
    s.add_data_by_key("b", 2)
    assert s.data == data


def test_get_data():
    s = Subject("subject")
    data = {"a": 1, "b": 2}
    s.add_data(data)
    assert s.get_data() == data


def test_get_data_by_key():
    s = Subject("subject")
    data = {"a": 1, "b": 2}
    s.add_data(data)
    assert s.get_data_by_key("a") == 1
    assert s.get_data_by_key("b") == 2
    assert s.get_data_by_key("c") is None


def test_replace_data():
    s = Subject("subject")
    data = {"a": 1, "b": 2}
    s.add_data(data)
    s.replace_data("a", 3)
    assert s.get_data_by_key("a") == 3
    assert s.get_data_by_key("b") == 2
    s.replace_data("c", 4, add_if_missing=True)
    assert s.get_data_by_key("c") == 4
    assert s.get_data_by_key("b") == 2
    assert s.get_data_by_key("a") == 3


def test_remove_additional_data():
    s = Subject("subject")
    data = {"a": 1, "b": 2}
    s.add_data(data)
    s.remove_additional_data()
    assert s.get_data() == {}


def test_remove_additional_data_by_key():
    s = Subject("subject")
    data = {"a": 1, "b": 2}
    s.add_data(data)
    s.remove_additional_data_by_key("a")
    assert s.get_data() == {"b": 2}
    s.remove_additional_data_by_key("b")
    assert s.get_data() == {}


def test_playback_transform_tapes():
    pass


def test__str__():
    s = Subject("subject")
    s.add_images([img_1, img_2, seg_1, seg_2], force=True)
    assert str(s) == "subject (Intensity Images: 2 / Segmentation Images: 2)"
