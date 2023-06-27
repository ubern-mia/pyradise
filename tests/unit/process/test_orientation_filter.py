import SimpleITK as sitk

from pyradise.process.orientation import OrientationFilter, OrientationFilterParams
from pyradise.data import Subject, IntensityImage, TransformInfo, ImageProperties


def test_is_invertible():
    filter = OrientationFilter()
    assert filter.is_invertible() is True


def test_execute(img_file_nii):
    filter = OrientationFilter()
    filter_params = OrientationFilterParams("LPS")
    s = Subject("test_name", IntensityImage(sitk.ReadImage(img_file_nii), "modality"))
    new_s = filter.execute(s, filter_params)
    assert new_s.intensity_images[0].get_orientation() == "LPS"


def test_execute_inverse(img_file_nii):
    filter = OrientationFilter()
    filter_params_ras = OrientationFilterParams("RAS")
    s_prop = ImageProperties(sitk.ReadImage(img_file_nii))

    s = Subject("test_name", IntensityImage(sitk.ReadImage(img_file_nii), "modality"))
    new_s = filter.execute(s, filter_params_ras)
    new_s_prop = ImageProperties(new_s.intensity_images[0].get_image_data())

    trans_info = TransformInfo("first", filter_params_ras, s_prop, new_s_prop)

    # old_s = filter.execute_inverse(new_s, trans_info)
    # print(old_s.intensity_images[0].get_orientation())
    # new_s = filter.execute(new_s, filter_params.inverse())
    # assert new_s.intensity_images[0].get_orientation() == "RAS"