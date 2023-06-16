import SimpleITK as sitk
import itk
import numpy as np

dummy_meta_nifti = {'ITK_FileNotes': '', 'ITK_original_direction': '[UNKNOWN_PRINT_CHARACTERISTICS]\n',
              'ITK_original_spacing': '[UNKNOWN_PRINT_CHARACTERISTICS]\n', 'aux_file': '', 'bitpix': '32',
              'cal_max': '0', 'cal_min': '0', 'datatype': '16', 'descrip': '', 'dim[0]': '3', 'dim[1]': '182',
              'dim[2]': '218', 'dim[3]': '182', 'dim[4]': '1', 'dim[5]': '1', 'dim[6]': '1', 'dim[7]': '1',
              'dim_info': '0', 'intent_code': '0', 'intent_name': '', 'intent_p1': '0', 'intent_p2': '0',
              'intent_p3': '0', 'nifti_type': '1', 'pixdim[0]': '0', 'pixdim[1]': '1', 'pixdim[2]': '1',
              'pixdim[3]': '1', 'pixdim[4]': '0', 'pixdim[5]': '0', 'pixdim[6]': '0', 'pixdim[7]': '0',
              'qfac': '[UNKNOWN_PRINT_CHARACTERISTICS]\n', 'qform_code': '1',
              'qform_code_name': 'NIFTI_XFORM_SCANNER_ANAT', 'qoffset_x': '90', 'qoffset_y': '-126', 'qoffset_z': '-72',
              'qto_xyz': '[UNKNOWN_PRINT_CHARACTERISTICS]\n', 'quatern_b': '-0', 'quatern_c': '1', 'quatern_d': '0',
              'scl_inter': '0', 'scl_slope': '1', 'sform_code': '0', 'sform_code_name': 'NIFTI_XFORM_UNKNOWN',
              'slice_code': '0', 'slice_duration': '0', 'slice_end': '0', 'slice_start': '0', 'srow_x': '0 0 0 0',
              'srow_y': '0 0 0 0', 'srow_z': '0 0 0 0', 'toffset': '0', 'vox_offset': '352', 'xyzt_units': '2'}


def get_sitk_intensity_image(seed) -> sitk.Image:
    """Returns a SimpleITK image with random intensities and the dummy meta data."""
    np.random.seed(seed)
    image = sitk.GetImageFromArray(np.random.randint(low=0, high=101, size=(182, 218, 182), dtype=np.uint8))
    for key, value in dummy_meta_nifti.items():
        image.SetMetaData(key, value)
    return image


def get_sitk_segmentation_image(seed) -> sitk.Image:
    """Returns a SimpleITK image with random intensities and the dummy meta data."""
    np.random.seed(seed)
    image = sitk.GetImageFromArray(np.random.randint(low=0, high=2, size=(182, 218, 182), dtype=np.uint8))
    for key, value in dummy_meta_nifti.items():
        image.SetMetaData(key, value)
    return image


def get_itk_intensity_image(seed) -> itk.Image:
    np.random.seed(seed)
    sitk_image = get_sitk_intensity_image(seed)
    is_vector_image = sitk_image.GetNumberOfComponentsPerPixel() > 1
    itk_image = itk.GetImageFromArray(sitk.GetArrayFromImage(sitk_image), is_vector=is_vector_image)
    itk_image.SetOrigin(sitk_image.GetOrigin())
    itk_image.SetSpacing(sitk_image.GetSpacing())
    itk_image.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(sitk_image.GetDirection()), [3] * 2)))
    return itk_image

