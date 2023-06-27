import os
import shutil
import itk
import numpy as np
import SimpleITK as sitk

from PIL import Image
import pytest
import tempfile


dummy_meta_nifti = {
    "ITK_FileNotes": "",
    "ITK_original_direction": "[UNKNOWN_PRINT_CHARACTERISTICS]\n",
    "ITK_original_spacing": "[UNKNOWN_PRINT_CHARACTERISTICS]\n",
    "aux_file": "",
    "bitpix": "32",
    "cal_max": "0",
    "cal_min": "0",
    "datatype": "16",
    "descrip": "",
    "dim[0]": "3",
    "dim[1]": "182",
    "dim[2]": "218",
    "dim[3]": "182",
    "dim[4]": "1",
    "dim[5]": "1",
    "dim[6]": "1",
    "dim[7]": "1",
    "dim_info": "0",
    "intent_code": "0",
    "intent_name": "",
    "intent_p1": "0",
    "intent_p2": "0",
    "intent_p3": "0",
    "nifti_type": "1",
    "pixdim[0]": "0",
    "pixdim[1]": "1",
    "pixdim[2]": "1",
    "pixdim[3]": "1",
    "pixdim[4]": "0",
    "pixdim[5]": "0",
    "pixdim[6]": "0",
    "pixdim[7]": "0",
    "qfac": "[UNKNOWN_PRINT_CHARACTERISTICS]\n",
    "qform_code": "1",
    "qform_code_name": "NIFTI_XFORM_SCANNER_ANAT",
    "qoffset_x": "90",
    "qoffset_y": "-126",
    "qoffset_z": "-72",
    "qto_xyz": "[UNKNOWN_PRINT_CHARACTERISTICS]\n",
    "quatern_b": "-0",
    "quatern_c": "1",
    "quatern_d": "0",
    "scl_inter": "0",
    "scl_slope": "1",
    "sform_code": "0",
    "sform_code_name": "NIFTI_XFORM_UNKNOWN",
    "slice_code": "0",
    "slice_duration": "0",
    "slice_end": "0",
    "slice_start": "0",
    "srow_x": "0 0 0 0",
    "srow_y": "0 0 0 0",
    "srow_z": "0 0 0 0",
    "toffset": "0",
    "vox_offset": "352",
    "xyzt_units": "2",
}

dummy_meta_dicom = {
    "0008|0005": "ISO_IR 100",
    "0008|0008": "DERIVED\\PRIMARY\\DIFFUSION\\TRACEW\\ND ",
    "0008|0012": "20130327",
    "0008|0013": "111401.570000 ",
    "0008|0016": "1.2.840.10008.5.1.4.1.1.4",
    "0008|0018": "1.3.12.2.1107.5.2.18.41437.2013032711140092932821838",
    "0008|0020": "20130327",
    "0008|0021": "20130327",
    "0008|0022": "20130327",
    "0008|0023": "20130327",
    "0008|0030": "111054",
    "0008|0031": "111401.509000 ",
    "0008|0032": "111316.990000 ",
    "0008|0033": "111401.570000 ",
    "0008|0050": "3113357 ",
    "0008|0060": "MR",
    "0008|0070": "SIEMENS ",
    "0008|0080": "",
    "0008|0081": "",
    "0008|0090": "",
    "0008|1010": "",
    "0008|1030": "",
    "0008|103e": "ep2d_diff_3scan_p3_m128_TRACEW",
    "0008|1040": "",
    "0008|1048": "",
    "0008|1050": "",
    "0008|1070": "",
    "0008|1090": "Aera",
    "0010|0010": "WAS_084_1953",
    "0010|0020": "1234567890",
    "0010|0030": "19000101",
    "0010|0040": "F ",
    "0010|1010": "060Y",
    "0010|1020": "1.56",
    "0010|1030": "77",
    "0010|1040": "",
    "0010|21c0": "",
    "0018|0015": "MR",
    "0018|0020": "EP",
    "0018|0021": "SK\\SP ",
    "0018|0022": "PFP\\FS",
    "0018|0023": "2D",
    "0018|0024": "*ep_b0",
    "0018|0025": "N ",
    "0018|0050": "5 ",
    "0018|0080": "4500",
    "0018|0081": "70",
    "0018|0083": "1 ",
    "0018|0084": "63.571835 ",
    "0018|0085": "1H",
    "0018|0086": "1 ",
    "0018|0087": "1.5 ",
    "0018|0088": "6.5 ",
    "0018|0089": "96",
    "0018|0091": "32",
    "0018|0093": "100 ",
    "0018|0094": "100 ",
    "0018|0095": "1115",
    "0018|1000": "41437 ",
    "0018|1020": "syngo MR D13",
    "0018|1030": "ep2d_diff_3scan_p3_m128 ",
    "0018|1251": "Body",
    "0018|1310": "128\\0\\0\\128",
    "0018|1312": "COL ",
    "0018|1314": "90",
    "0018|1315": "N ",
    "0018|1316": "0.04887582886102",
    "0018|1318": "0 ",
    "0018|5100": "HFS ",
    "0020|000d": "1.2.752.24.7.1834641122.2366502",
    "0020|000e": "1.3.12.2.1107.5.2.18.41437.2013032711125140077020963.0.0.0",
    "0020|0010": "0 ",
    "0020|0011": "2 ",
    "0020|0012": "1 ",
    "0020|0013": "1 ",
    "0020|0032": "-127.2450464557\\-96.916864538982\\-61.173939013721 ",
    "0020|0037": "0.99990252362573\\-0.0139586140877\\-0.0003167643497\\0.01396220780907\\0.99964515801537\\0.02268512308825 ",
    "0020|0052": "1.3.12.2.1107.5.2.18.41437.1.20130327111054412.0.0.0",
    "0020|1040": "",
    "0020|1041": "-58.959407962629",
    "0020|4000": "",
    "0028|0002": "1",
    "0028|0004": "MONOCHROME2 ",
    "0028|0010": "256",
    "0028|0011": "256",
    "0028|0030": "0.8984375\\0.8984375 ",
    "0028|0100": "16",
    "0028|0101": "12",
    "0028|0102": "11",
    "0028|0103": "0",
    "0028|0106": "0",
    "0028|0107": "922",
    "0028|1050": "363 ",
    "0028|1051": "790 ",
    "0028|1055": "Algo1 ",
    "0032|1032": "",
    "0032|1033": "",
    "0032|1060": "MR Protonenspektroskopie",
    "0040|0244": "20130327",
    "0040|0245": "111054",
    "0040|0253": "0 ",
    "0040|0254": "MR",
    "0040|0280": "",
    "0008|1115": "ReferencedSeriesSequence",
}


def get_sitk_image(seed=0, low=0, high=101, meta="dcm") -> sitk.Image:
    """Returns a SimpleITK image with random intensities and the dummy meta data."""
    np.random.seed(seed)
    image = sitk.GetImageFromArray(
        np.random.randint(low=low, high=high, size=(182, 218, 182), dtype=np.uint8)
    )

    if meta == "dcm":
        meta_data = dummy_meta_dicom
    elif meta == "dcm_no_meta":
        meta_data = {}
    elif meta == "nii":
        meta_data = dummy_meta_nifti
    else:
        raise ValueError("meta must be either dcm or nii")

    for key, value in meta_data.items():
        image.SetMetaData(key, value)

    return image


def get_itk_image(seed=0, low=0, high=101, meta="dcm") -> itk.Image:
    np.random.seed(seed)
    sitk_image = get_sitk_image(seed, low, high, meta)
    is_vector_image = sitk_image.GetNumberOfComponentsPerPixel() > 1
    itk_image = itk.GetImageFromArray(
        sitk.GetArrayFromImage(sitk_image), is_vector=is_vector_image
    )
    itk_image.SetOrigin(sitk_image.GetOrigin())
    itk_image.SetSpacing(sitk_image.GetSpacing())
    itk_image.SetDirection(
        itk.GetMatrixFromArray(np.reshape(np.array(sitk_image.GetDirection()), [3] * 2))
    )
    return itk_image


@pytest.fixture
def png_file(tmp_path):
    tmp_dir = tempfile.mkdtemp(dir=tmp_path)
    image = Image.new("RGB", (100, 100), color="red")
    tmp_file = os.path.join(tmp_dir, "image.png")
    image.save(tmp_file)
    yield str(tmp_file)
    shutil.rmtree(tmp_dir)


@pytest.fixture
def img_file_nii(tmp_path):
    tmp_dir = tempfile.mkdtemp(dir=tmp_path)
    tmp_file = os.path.join(tmp_dir, "img.nii")
    sitk.WriteImage(get_sitk_image(0, 0, 101, "nii"), tmp_file)
    yield str(tmp_file)
    shutil.rmtree(tmp_dir)


@pytest.fixture
def seg_file_nii(tmp_path) -> str:
    tmp_dir = tempfile.mkdtemp(dir=tmp_path)
    tmp_file = os.path.join(tmp_dir, "seg.nii")
    sitk.WriteImage(get_sitk_image(0, 0, 2, "nii"), tmp_file)
    yield str(tmp_file)
    shutil.rmtree(tmp_dir)


@pytest.fixture
def img_file_dcm(tmp_path) -> str:
    tmp_dir = tempfile.mkdtemp(dir=tmp_path)
    tmp_file = os.path.join(tmp_dir, "img.dcm")
    sitk.WriteImage(get_sitk_image(0, 0, 101, "dcm"), tmp_file)
    yield str(tmp_file)
    shutil.rmtree(tmp_dir)


@pytest.fixture
def img_file_dcm_no_meta(tmp_path) -> str:
    tmp_dir = tempfile.mkdtemp(dir=tmp_path)
    tmp_file = os.path.join(tmp_dir, "img.dcm")
    sitk.WriteImage(get_sitk_image(0, 0, 101, "dcm_no_meta"), tmp_file)
    yield str(tmp_file)
    shutil.rmtree(tmp_dir)


@pytest.fixture
def seg_file_dcm(tmp_path) -> str:
    tmp_dir = tempfile.mkdtemp(dir=tmp_path)
    tmp_file = os.path.join(tmp_dir, "seg.dcm")
    sitk.WriteImage(get_sitk_image(0, 0, 101, "dcm"), tmp_file)
    yield str(tmp_file)
    shutil.rmtree(tmp_dir)


@pytest.fixture
def img_series_dcm(tmp_path_factory) -> str:
    tmp_dir = tmp_path_factory.mktemp("data")
    sitk_image = get_sitk_image(0, 0, 101, "dcm")
    for z in range(sitk_image.GetDepth()):
        tmp_file = os.path.join(tmp_dir, f"img_serie_{z}.dcm")
        slice_image = sitk_image[:, :, z]
        for key in sitk_image.GetMetaDataKeys():
            slice_image.SetMetaData(key, sitk_image.GetMetaData(key))
        sitk.WriteImage(slice_image, tmp_file)
    yield str(tmp_dir)
    shutil.rmtree(tmp_dir)


@pytest.fixture
def seg_series_dcm(tmp_path_factory) -> str:
    tmp_dir = tmp_path_factory.mktemp("data")
    sitk_image = get_sitk_image(0, 0, 2, "dcm")
    for z in range(sitk_image.GetDepth()):
        tmp_file = os.path.join(tmp_dir, f"seg_serie_{z}.dcm")
        slice_image = sitk_image[:, :, z]
        for key in sitk_image.GetMetaDataKeys():
            slice_image.SetMetaData(key, sitk_image.GetMetaData(key))
        sitk.WriteImage(slice_image, tmp_file)
    yield str(tmp_dir)
    shutil.rmtree(tmp_dir)
