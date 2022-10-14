from typing import Tuple, Optional, List, Any, Union, Dict
import os
from datetime import datetime

import itk
import numpy as np
import SimpleITK as sitk
import vtkmodules.vtkFiltersCore as vtk_fcore
import vtkmodules.vtkImagingCore as vtk_icore
import vtkmodules.vtkImagingGeneral as vtk_igen
import vtkmodules.vtkCommonDataModel as vtk_dm
import vtkmodules.vtkFiltersModeling as vtk_fmodel
# import vtkmodules.vtkCommonCore as vtk_core
import vtkmodules.vtkRenderingCore as vtk_rcore
import vtkmodules.vtkCommonColor as vtk_ccolor
import vtkmodules.vtkCommonCore as vtk_ccore
import vtkmodules.vtkFiltersGeneral as vtk_fgeneral
import vtkmodules.vtkIOGeometry as vtk_iog
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    generate_uid,
    ImplicitVRLittleEndian,
    PYDICOM_IMPLEMENTATION_UID)
from pydicom import dcmwrite



from pyradise.utils import (
    convert_to_itk_image,
    load_datasets,
    get_slice_position,
    get_spacing_between_slices,
    get_slice_direction)
import pyradise.fileio as ps_io
import pyradise.data as ps_data


class ExampleModalityExtractor(ps_io.ModalityExtractor):

    def extract_from_dicom(self, path: str) -> Optional[ps_data.Modality]:
        # Extract the necessary attributes from the DICOM file
        tags = (ps_io.Tag((0x0008, 0x0060)),    # Modality
                ps_io.Tag((0x0008, 0x103e)))  # Series Description
        dataset_dict = self._load_dicom_attributes(tags, path)

        # Identify the modality rule-based
        modality = dataset_dict.get('Modality', {}).get('value', None)
        series_desc = dataset_dict.get('Series Description', {}).get('value', '')
        if modality == 'MR':
            if 't1' in series_desc.lower():
                return ps_data.Modality('T1')
            elif 't2' in series_desc.lower():
                return ps_data.Modality('T2')
            else:
                return None
        else:
            return None

    def extract_from_path(self, path: str) -> Optional[ps_data.Modality]:
        # Identify the discrete image file's modality rule-based
        filename = os.path.basename(path)

        # Check if the image contains an img prefix
        # (i.e., it is a intensity image)
        if not filename.startswith('img'):
            return None

        # Check if the image contains a modality search string
        if 'T1' in filename:
            return ps_data.Modality('T1')
        elif 'T2' in filename:
            return ps_data.Modality('T2')
        else:
            return None


def generate_basic_rtss(ref_image_datasets: Tuple[Dataset, ...],
                        file_name: str = 'rt_struct'
                        ) -> FileDataset:
    """Generate the basic RTSS skeleton.

    Args:
        ref_image_datasets (Tuple[Dataset, ...]): The referenced image datasets.
        file_name (str): The file name.

    Returns:
        FileDataset: The basic RT Structure Set.
    """
    # create the file meta for the dataset
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 202
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

    # create the dataset
    rtss = FileDataset(file_name, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # add the basic information
    now = datetime.now()
    rtss.SpecificCharacterSet = 'ISO_IR 100'
    rtss.InstanceCreationDate = now.strftime('%Y%m%d')
    rtss.InstanceCreationTime = now.strftime('%H%M%S.%f')
    rtss.StructureSetLabel = 'VTK Auto'
    rtss.StructureSetDate = now.strftime('%Y%m%d')
    rtss.StructureSetTime = now.strftime('%H%M%S.%f')
    rtss.Modality = 'RTSTRUCT'
    rtss.Manufacturer = 'University of Bern, Switzerland'
    rtss.ManufacturerModelName = 'PyRaDiSe'
    rtss.InstitutionName = 'University of Bern, Switzerland'
    rtss.OperatorsName = 'PyRaDiSe Converter Algorithm'
    rtss.ApprovalStatus = 'UNAPPROVED'

    rtss.is_little_endian = True
    rtss.is_implicit_VR = True

    # set values already defined in the file meta
    rtss.SOPClassUID = rtss.file_meta.MediaStorageSOPClassUID
    rtss.SOPInstanceUID = rtss.file_meta.MediaStorageSOPInstanceUID

    # add study and series information
    reference_dataset = ref_image_datasets[0]
    rtss.StudyDate = reference_dataset.StudyDate
    rtss.SeriesDate = getattr(reference_dataset, 'SeriesDate', '')
    rtss.StudyTime = reference_dataset.StudyTime
    rtss.SeriesTime = getattr(reference_dataset, 'SeriesTime', '')
    rtss.StudyDescription = getattr(reference_dataset, 'StudyDescription', '')
    # rtss.SeriesDescription = getattr(reference_dataset, 'SeriesDescription', '')
    rtss.SeriesDescription = 'VTK TEST'  # TODO change
    rtss.StudyInstanceUID = reference_dataset.StudyInstanceUID
    rtss.SeriesInstanceUID = generate_uid()
    rtss.StudyID = reference_dataset.StudyID
    rtss.SeriesNumber = "99"
    rtss.ReferringPhysicianName = "UNKNOWN"
    rtss.AccessionNumber = "0"

    # add the patient information
    rtss.PatientName = getattr(reference_dataset, 'PatientName', '')
    rtss.PatientID = getattr(reference_dataset, 'PatientID', '')
    rtss.PatientBirthDate = getattr(reference_dataset, 'PatientBirthDate', '')
    rtss.PatientSex = getattr(reference_dataset, 'PatientSex', '')
    rtss.PatientAge = getattr(reference_dataset, 'PatientAge', '')
    rtss.PatientSize = getattr(reference_dataset, 'PatientSize', '')
    rtss.PatientWeight = getattr(reference_dataset, 'PatientWeight', '')

    # ImageToRTSSConverter._add_referenced_frame_of_ref_sequence(rtss, reference_image_datasets)

    # construct the ContourImageSequence
    contour_image_sequence = Sequence()
    for image_dataset in ref_image_datasets:
        contour_image_entry = Dataset()
        contour_image_entry.ReferencedSOPClassUID = image_dataset.file_meta.MediaStorageSOPClassUID
        contour_image_entry.ReferencedSOPInstanceUID = image_dataset.file_meta.MediaStorageSOPInstanceUID
        contour_image_sequence.append(contour_image_entry)

    # construct the RTReferencedSeriesSequence
    rt_referenced_series_sequence = Sequence()
    rt_referenced_series_entry = Dataset()
    rt_referenced_series_entry.SeriesInstanceUID = reference_dataset.SeriesInstanceUID
    rt_referenced_series_entry.ContourImageSequence = contour_image_sequence
    rt_referenced_series_sequence.append(rt_referenced_series_entry)

    # construct the RTReferencedStudySequence
    rt_referenced_study_sequence = Sequence()
    rt_referenced_study_entry = Dataset()
    rt_referenced_study_entry.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.1'  # RT Structure Set Storage
    rt_referenced_study_entry.ReferencedSOPInstanceUID = reference_dataset.StudyInstanceUID
    rt_referenced_study_entry.RTReferencedSeriesSequence = rt_referenced_series_sequence
    rt_referenced_study_sequence.append(rt_referenced_study_entry)

    # construct the ReferencedFrameOfReferenceSequence
    rtss.ReferencedFrameOfReferenceSequence = Sequence()
    referenced_frame_of_ref_entry = Dataset()
    referenced_frame_of_ref_entry.FrameOfReferenceUID = reference_dataset.FrameOfReferenceUID
    referenced_frame_of_ref_entry.RTReferencedStudySequence = rt_referenced_study_sequence
    rtss.ReferencedFrameOfReferenceSequence.append(referenced_frame_of_ref_entry)

    # construct the ROIContourSequence, StructureSetROISequence and RTROIObservationsSequence
    rtss.ROIContourSequence = Sequence()
    rtss.StructureSetROISequence = Sequence()
    rtss.RTROIObservationsSequence = Sequence()

    return rtss


def create_contour_sequence(coords: List[List[List[List[float]]]],
                            image_datasets: Tuple[Dataset, ...]
                            ) -> Sequence:
    contour_sequence = Sequence()

    for slice_dataset, slice_coords in zip(image_datasets, coords):
        for contour_data in slice_coords:
            contour_image = Dataset()
            contour_image.ReferencedSOPClassUID = slice_dataset.file_meta.MediaStorageSOPClassUID
            contour_image.ReferencedSOPInstanceUID = slice_dataset.file_meta.MediaStorageSOPInstanceUID

            contour_image_sequence = Sequence()
            contour_image_sequence.append(contour_image)

            contour = Dataset()
            contour.ContourImageSequence = contour_image_sequence
            contour.ContourGeometricType = 'CLOSED_PLANAR'
            contour.NumberOfContourPoints = len(contour_data)
            contour.ContourData = [coord for point in contour_data for coord in point]
            contour_sequence.append(contour)

    return contour_sequence


def add_roi_contour_sequence(rtss: Dataset,
                             datasets: Tuple[Dataset, ...],
                             contours: List[List[List[List[float]]]],
                             color: Tuple[int, int, int],
                             roi_number: int,
                             ) -> None:
    """Add a ROIContourSequence to the RTSS dataset."""
    roi_contour = Dataset()
    roi_contour.ROIDisplayColor = list(color)
    roi_contour.ReferencedROINumber = str(roi_number)
    roi_contour.ContourSequence = create_contour_sequence(contours, datasets)

    rtss.ROIContourSequence.append(roi_contour)
    #
    #
    #
    #
    #
    # roi_contour = Dataset()
    # roi_contour.ROIDisplayColor = list(color)
    # roi_contour.ReferencedROINumber = roi_number
    # roi_contour.ContourSequence = Sequence()
    #
    # for dataset, contour in zip(datasets, contours):
    #     contour_item = Dataset()
    #     contour_item.ContourGeometricType = 'CLOSED_PLANAR'
    #     contour_item.NumberOfContourPoints = len(contour) // 3
    #     contour_item.ContourData = [entry for entry in contour]  # TODO maybe reduce precision
    #     contour_item.ContourImageSequence = Sequence()
    #     contour_image_item = Dataset()
    #     contour_image_item.ReferencedSOPClassUID = dataset.file_meta.MediaStorageSOPClassUID
    #     contour_image_item.ReferencedSOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID
    #     contour_item.ContourImageSequence.append(contour_image_item)
    #     roi_contour.ContourSequence.append(contour_item)
    #
    # rtss.ROIContourSequence.append(roi_contour)


def add_structure_set_roi_sequence(rtss: Dataset,
                                   image_dataset: Dataset,
                                   roi_name: str,
                                   roi_number: int,
                                   roi_generation_algorithm: str = 'AUTOMATIC'
                                   ) -> None:
    structure_set_roi = Dataset()
    structure_set_roi.ROINumber = roi_number
    structure_set_roi.ReferencedFrameOfReferenceUID = image_dataset.get('FrameOfReferenceUID')
    structure_set_roi.ROIName = roi_name
    structure_set_roi.ROIDescription = ''
    structure_set_roi.ROIGenerationAlgorithm = roi_generation_algorithm
    rtss.StructureSetROISequence.append(structure_set_roi)


def add_rt_roi_observations_sequence(rtss: Dataset,
                                     roi_number: int,
                                     ) -> None:
    rt_roi_observations = Dataset()
    rt_roi_observations.ObservationNumber = roi_number
    rt_roi_observations.ReferencedROINumber = roi_number
    rt_roi_observations.RTROIInterpretedType = ''
    rt_roi_observations.ROIInterpreter = ''
    rt_roi_observations.ROIObservationDescription = 'Type:Soft,Range:*/*,Fill:0,Opacity:0.0,Thickness:1,' \
                                                    'LineThickness:2,read-only:false'
    rt_roi_observations.private_creators = 'University of Bern, Switzerland'
    rtss.RTROIObservationsSequence.append(rt_roi_observations)


def load_dicom_image(dir_path: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dir_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


def load_dicom_datasets(dir_path: str) -> Tuple[Dataset, ...]:
    crawler = ps_io.SubjectDicomCrawler(dir_path,
                                        modality_extractor=ExampleModalityExtractor())
    series_info = crawler.execute()

    selection = ps_io.NoRTSSInfoSelector()
    series_info = selection.execute(series_info)

    dicom_dataset_paths = series_info[0].get_path()
    dicom_datasets = load_datasets(dicom_dataset_paths)
    return dicom_datasets


def load_nifti_image(image_path: str) -> sitk.Image:
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_path)
    image = reader.Execute()
    return image


def preprocess_image(nifti_image: sitk.Image,
                     dcm_image: sitk.Image,
                     dcm_datasets: Tuple[Dataset, ...]
                     ) -> sitk.Image:
    nifti_image = sitk.DICOMOrient(nifti_image, 'LPS')

    # Generate the resampling via the dicom datasets
    # TODO Check with non isotropic data
    origin = dcm_datasets[0].ImagePositionPatient
    slice_spacing = get_spacing_between_slices(dcm_datasets)
    spacing = (dcm_datasets[0].PixelSpacing[0], dcm_datasets[0].PixelSpacing[1], slice_spacing)
    direction = np.array(get_slice_direction(dcm_datasets[0])).T.flatten()
    size = (dcm_datasets[0].Rows,
            dcm_datasets[0].Columns,
            len(dcm_datasets))

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputDirection(direction)
    resampler.SetSize(size)
    resampler.SetTransform(sitk.Transform())
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    nifti_image_2 = resampler.Execute(nifti_image)


    # Resample to match the DICOM image
    nifti_image = sitk.Resample(nifti_image, dcm_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0,
                                nifti_image.GetPixelID())

    # Convert to binary
    nifti_image = sitk.BinaryThreshold(nifti_image, 1, 255, 255, 0)

    return nifti_image


def get_3d_reconstruction(nifti_image_path: str,
                          dicom_dir_path: str
                          ) -> None:
    # Load the data
    nifti_image = load_nifti_image(nifti_image_path)
    dicom_datasets = load_dicom_datasets(dicom_dir_path)
    dicom_image = load_dicom_image(dicom_dir_path)

    # Check the image
    image_np = sitk.GetArrayFromImage(nifti_image)

    # Preprocess the data
    nifti_image = preprocess_image(nifti_image, dicom_image, dicom_datasets)

    image_np = sitk.GetArrayFromImage(nifti_image)

    # Cast the image to vtkImageData
    itk_image = convert_to_itk_image(nifti_image)
    vtk_image = itk.vtk_image_from_image(itk_image)

    # MAYBE: set the direction matrix
    vtk_image.SetDirectionMatrix(nifti_image.GetDirection())

    # Apply gaussian smoothing
    gaussian = vtk_igen.vtkImageGaussianSmooth()
    gaussian.SetInputDataObject(0, vtk_image)
    gaussian.SetStandardDeviation(2.0)
    gaussian.SetRadiusFactor(1.0)

    # Apply flying edges
    flying_edges = vtk_fcore.vtkFlyingEdges3D()
    flying_edges.SetInputConnection(0, gaussian.GetOutputPort(0))
    flying_edges.SetValue(0, 127.5)
    flying_edges.ComputeGradientsOff()
    flying_edges.ComputeNormalsOff()

    # Apply decimation
    decimate = vtk_fcore.vtkDecimatePro()
    decimate.SetInputConnection(0, flying_edges.GetOutputPort(0))
    decimate.SetTargetReduction(0.5)
    decimate.PreserveTopologyOn()
    decimate.SetMaximumError(0.02)
    decimate.SplittingOff()
    decimate.SetFeatureAngle(60.)

    # Smooth via the sinc filter
    smoother = vtk_fcore.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(0, decimate.GetOutputPort(0))
    smoother.SetNumberOfIterations(15)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(60.0)
    smoother.SetPassBand(0.25)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOff()

    # Get normals
    normals = vtk_fcore.vtkPolyDataNormals()
    normals.SetInputConnection(0, smoother.GetOutputPort(0))
    normals.SetFeatureAngle(60.0)

    # Strip the polydata
    stripper = vtk_fcore.vtkStripper()
    stripper.SetInputConnection(0, normals.GetOutputPort(0))
    stripper.JoinContiguousSegmentsOn()
    stripper.Update(0)

    # Write the polydata to a file
    writer = vtk_iog.vtkSTLWriter()
    writer.SetFileName('D:/test_ii.stl')
    writer.SetInputConnection(stripper.GetOutputPort())
    writer.Write()

    # Create a mapper and actor
    mapper_left = vtk_rcore.vtkPolyDataMapper()
    mapper_left.SetInputConnection(0, stripper.GetOutputPort(0))

    actor_left = vtk_rcore.vtkActor()
    actor_left.SetMapper(mapper_left)

    # Create the slicing of the model
    # -------------------------------
    origin = dicom_datasets[0].get('ImagePositionPatient')
    origin = [float(val) for val in origin]
    first_pos = dicom_datasets[0].get('ImagePositionPatient')
    last_pos = dicom_datasets[-1].get('ImagePositionPatient')
    length = np.abs(np.linalg.norm(np.array(last_pos) - np.array(first_pos)))
    normal = tuple(np.cross(np.array(dicom_datasets[0].get('ImageOrientationPatient')[0:3]),
                            np.array(dicom_datasets[0].get('ImageOrientationPatient')[3:6])))

    # Create the plane
    plane = vtk_dm.vtkPlane()
    plane.SetOrigin(*origin)
    plane.SetNormal(*normal)

    # Create the cutter
    cutter = vtk_fcore.vtkCutter()
    cutter.SetInputConnection(0, stripper.GetOutputPort(0))
    cutter.SetCutFunction(plane)
    cutter.GenerateTrianglesOn()
    cutter.GenerateValues(len(dicom_datasets), 0, length)

    # Create the cleaner
    cleaner = vtk_fcore.vtkCleanPolyData()
    cleaner.SetInputConnection(0, cutter.GetOutputPort(0))
    cleaner.SetAbsoluteTolerance(0.1)
    cleaner.SetPointMerging(True)
    cleaner.Update(0)

    # Get the polylines
    loop = vtk_fmodel.vtkContourLoopExtraction()
    loop.SetInputConnection(0, cleaner.GetOutputPort(0))
    loop.SetOutputModeToPolygons()
    loop.SetNormal(*normal)
    loop.Update(0)

    # Get the polylines
    cells = loop.GetOutput().GetPolys()
    points = loop.GetOutput().GetPoints()

    indices = vtk_ccore.vtkIdList()
    contours_points = []
    poly_count = 0
    while cells.GetNextCell(indices):
        contour_points = []
        for i in range(indices.GetNumberOfIds()):
            point = list(points.GetPoint(indices.GetId(i)))
            # print(point)
            contour_points.append(point)

        contours_points.append(contour_points)
        print(f'Contour {poly_count}: {len(contour_points)} points')
        poly_count += 1


    # Prepare the RTSS
    rtss = generate_basic_rtss(tuple(dicom_datasets))
    add_structure_set_roi_sequence(rtss, dicom_datasets[0], 'test', 1)
    add_roi_contour_sequence(rtss, dicom_datasets, [contours_points,], (255, 0, 0), 1)
    add_rt_roi_observations_sequence(rtss, 1)
    rtss_path = os.path.join(dicom_dir_path, 'test.dcm')
    dcmwrite(rtss_path, rtss)

    # Create the right mapper
    mapper_right = vtk_rcore.vtkPolyDataMapper()
    mapper_right.SetInputConnection(0, cleaner.GetOutputPort(0))
    mapper_right.ScalarVisibilityOff()

    # Create the right actor
    colors = vtk_ccolor.vtkNamedColors()
    actor_right = vtk_rcore.vtkActor()
    actor_right.SetMapper(mapper_right)
    actor_right.GetProperty().SetColor(colors.GetColor3d('Orange'))
    actor_right.GetProperty().SetLineWidth(2)

    # Create a renderer, render window, and interactor
    renderer_left = vtk_rcore.vtkRenderer()
    renderer_left.SetViewport(0, 0, 0.5, 1)
    renderer_right = vtk_rcore.vtkRenderer()
    renderer_right.SetViewport(0.5, 0, 1, 1)
    render_window = vtk_rcore.vtkRenderWindow()
    render_window.SetSize(640, 480)
    render_window.AddRenderer(renderer_left)
    render_window.AddRenderer(renderer_right)
    render_window_interactor = vtk_rcore.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    renderer_left.AddActor(actor_left)
    renderer_right.AddActor(actor_right)

    render_window.Render()
    render_window_interactor.Start()


class SegmentToRTSSConverter3D(ps_io.Converter):

    def __init__(self,
                 label_images: Union[Tuple[str, ...], Tuple[sitk.Image, ...]],
                 ref_image_datasets: Union[Tuple[str, ...], Tuple[Dataset, ...]],
                 roi_names: Union[Tuple[str, ...], Dict[int, str], None],
                 colors: Optional[Tuple[Tuple[int, int, int], ...]],
                 smoothing: Union[bool, Tuple[bool, ...]] = True,
                 smoothing_sigma: Union[float, Tuple[float, ...]] = 2.0,
                 smoothing_kernel_size: Union[int, Tuple[int, ...]] = 32,
                 meta_data: ps_io.RTSSMetaData = ps_io.RTSSMetaData()
                 ):
        super().__init__()

        if isinstance(label_images[0], str):
            self.label_images: Tuple[sitk.Image, ...] = tuple([sitk.ReadImage(path, sitk.sitkUInt8)
                                                               for path in label_images])
        else:
            self.label_images: Tuple[sitk.Image, ...] = label_images

        if isinstance(ref_image_datasets[0], str):
            self.image_datasets: Tuple[Dataset, ...] = load_datasets(ref_image_datasets)
        else:
            self.image_datasets: Tuple[Dataset, ...] = ref_image_datasets

        if isinstance(roi_names, dict):
            sorted_keys = sorted(roi_names.keys())
            self.roi_names = tuple([str(roi_names.get(key)) for key in sorted_keys])
        elif not roi_names:
            self.roi_names = tuple([f'Structure_{i}' for i in range(len(self.label_images))])
        else:
            self.roi_names = roi_names

        if not colors:
            self.colors = ps_io.dicom_conversion.COLOR_PALETTE
        else:
            self.colors = colors

        assert len(self.roi_names) >= len(self.label_images), 'The number of ROI names must be equal or larger ' \
                                                              'than the number of label images!'
        assert len(self.colors) >= len(self.label_images), 'The number of colors must be equal or larger ' \
                                                           'than the number of label images!'

        if isinstance(smoothing, bool):
            self.smoothing = [smoothing] * len(self.label_images)
        else:
            assert len(smoothing) == len(self.label_images), 'The number of smoothing indicators must be equal ' \
                                                             'to the number of label images!'
            self.smoothing = smoothing

        if isinstance(smoothing_sigma, float) or isinstance(smoothing_sigma, int):
            self.smoothing_sigma = [float(smoothing_sigma)] * len(self.label_images)
        else:
            assert len(smoothing_sigma) == len(self.label_images), 'The number of smoothing sigmas must be equal ' \
                                                                   'to the number of label images!'
            self.smoothing_sigma = smoothing_sigma

        assert all([sigma > 0 for sigma in self.smoothing_sigma]), 'All smoothing sigma must be larger than 0!'

        if isinstance(smoothing_kernel_size, int):
            self.smoothing_kernel_size = [smoothing_kernel_size] * len(self.label_images)
        else:
            assert len(smoothing_kernel_size) == len(self.label_images), 'The number of smoothing kernel sizes must ' \
                                                                         'be equal to the number of label images!'
            self.smoothing_kernel_size = smoothing_kernel_size

        assert all([kernel > 0 for kernel in self.smoothing_kernel_size]), 'All smoothing kernel must be larger than 0!'

        self.meta_data = meta_data

    def _generate_basic_rtss(self,
                             file_name: str = 'rt_struct'
                             ) -> FileDataset:
        """Generate the basic RTSS skeleton.

        Args:
            file_name (str): The file name.

        Returns:
            FileDataset: The basic RT Structure Set.
        """
        # create the file meta for the dataset
        file_meta = FileMetaDataset()
        file_meta.FileMetaInformationGroupLength = 202
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

        # create the dataset
        rtss = FileDataset(file_name, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # add the basic information
        now = datetime.now()
        rtss.SpecificCharacterSet = 'ISO_IR 100'
        rtss.InstanceCreationDate = now.strftime('%Y%m%d')
        rtss.InstanceCreationTime = now.strftime('%H%M%S.%f')
        rtss.StructureSetLabel = self.meta_data.structure_set_label
        rtss.StructureSetDate = now.strftime('%Y%m%d')
        rtss.StructureSetTime = now.strftime('%H%M%S.%f')
        rtss.Modality = 'RTSTRUCT'
        rtss.Manufacturer = self.meta_data.manufacturer
        rtss.ManufacturerModelName = self.meta_data.manufacturer_model_name
        rtss.InstitutionName = self.meta_data.institution_name
        rtss.OperatorsName = self.meta_data.operators_name
        rtss.ApprovalStatus = self.meta_data.approval_status

        rtss.is_little_endian = True
        rtss.is_implicit_VR = True

        # set values already defined in the file meta
        rtss.SOPClassUID = rtss.file_meta.MediaStorageSOPClassUID
        rtss.SOPInstanceUID = rtss.file_meta.MediaStorageSOPInstanceUID

        # add study and series information
        reference_dataset = self.image_datasets[0]
        rtss.StudyDate = reference_dataset.StudyDate
        rtss.SeriesDate = getattr(reference_dataset, 'SeriesDate', '')
        rtss.StudyTime = reference_dataset.StudyTime
        rtss.SeriesTime = getattr(reference_dataset, 'SeriesTime', '')

        if self.meta_data.study_description is None:
            rtss.StudyDescription = getattr(reference_dataset, 'StudyDescription', '')
        else:
            rtss.StudyDescription = self.meta_data.study_description

        if self.meta_data.series_description is None:
            rtss.SeriesDescription = getattr(reference_dataset, 'SeriesDescription', '')
        else:
            rtss.SeriesDescription = self.meta_data.series_description

        rtss.StudyInstanceUID = reference_dataset.StudyInstanceUID
        rtss.SeriesInstanceUID = generate_uid()
        rtss.StudyID = reference_dataset.StudyID
        rtss.SeriesNumber = self.meta_data.series_number
        rtss.ReferringPhysicianName = self.meta_data.referring_physician_name
        rtss.AccessionNumber = "0"

        # add the patient information
        if self.meta_data.patient_name is None:
            rtss.PatientName = getattr(reference_dataset, 'PatientName', '')
        else:
            rtss.PatientName = self.meta_data.patient_name

        if self.meta_data.patient_id is None:
            rtss.PatientID = getattr(reference_dataset, 'PatientID', '')
        else:
            rtss.PatientID = self.meta_data.patient_id

        if self.meta_data.patient_birth_date is None:
            rtss.PatientBirthDate = getattr(reference_dataset, 'PatientBirthDate', '')
        else:
            rtss.PatientBirthDate = self.meta_data.patient_birth_date

        if self.meta_data.patient_sex is None:
            rtss.PatientSex = getattr(reference_dataset, 'PatientSex', '')
        else:
            rtss.PatientSex = self.meta_data.patient_sex

        if self.meta_data.patient_age is None:
            rtss.PatientAge = getattr(reference_dataset, 'PatientAge', '')
        else:
            rtss.PatientAge = self.meta_data.patient_age

        if self.meta_data.patient_weight is None:
            rtss.PatientWeight = getattr(reference_dataset, 'PatientWeight', '')
        else:
            rtss.PatientWeight = self.meta_data.patient_weight

        if self.meta_data.patient_size is None:
            rtss.PatientSize = getattr(reference_dataset, 'PatientSize', '')
        else:
            rtss.PatientSize = self.meta_data.patient_size

        # construct the ContourImageSequence
        contour_image_sequence = Sequence()
        for image_dataset in self.image_datasets:
            contour_image_entry = Dataset()
            contour_image_entry.ReferencedSOPClassUID = image_dataset.file_meta.MediaStorageSOPClassUID
            contour_image_entry.ReferencedSOPInstanceUID = image_dataset.file_meta.MediaStorageSOPInstanceUID
            contour_image_sequence.append(contour_image_entry)

        # construct the RTReferencedSeriesSequence
        rt_referenced_series_sequence = Sequence()
        rt_referenced_series_entry = Dataset()
        rt_referenced_series_entry.SeriesInstanceUID = reference_dataset.SeriesInstanceUID
        rt_referenced_series_entry.ContourImageSequence = contour_image_sequence
        rt_referenced_series_sequence.append(rt_referenced_series_entry)

        # construct the RTReferencedStudySequence
        rt_referenced_study_sequence = Sequence()
        rt_referenced_study_entry = Dataset()
        rt_referenced_study_entry.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.1'  # RT Structure Set Storage
        rt_referenced_study_entry.ReferencedSOPInstanceUID = reference_dataset.StudyInstanceUID
        rt_referenced_study_entry.RTReferencedSeriesSequence = rt_referenced_series_sequence
        rt_referenced_study_sequence.append(rt_referenced_study_entry)

        # construct the ReferencedFrameOfReferenceSequence
        rtss.ReferencedFrameOfReferenceSequence = Sequence()
        referenced_frame_of_ref_entry = Dataset()
        referenced_frame_of_ref_entry.FrameOfReferenceUID = reference_dataset.FrameOfReferenceUID
        referenced_frame_of_ref_entry.RTReferencedStudySequence = rt_referenced_study_sequence
        rtss.ReferencedFrameOfReferenceSequence.append(referenced_frame_of_ref_entry)

        # construct the ROIContourSequence, StructureSetROISequence and RTROIObservationsSequence
        rtss.ROIContourSequence = Sequence()
        rtss.StructureSetROISequence = Sequence()
        rtss.RTROIObservationsSequence = Sequence()

        return rtss

    def preprocess_image(self, nifti_image: sitk.Image) -> sitk.Image:
        nifti_image = sitk.DICOMOrient(nifti_image, 'LPS')

        # Resample to match the DICOM image
        origin = self.image_datasets[0].ImagePositionPatient
        slice_spacing = get_spacing_between_slices(self.image_datasets)
        spacing = (self.image_datasets[0].PixelSpacing[0],
                   self.image_datasets[0].PixelSpacing[1],
                   slice_spacing)
        direction = np.array(get_slice_direction(self.image_datasets[0])).T.flatten()
        size = (self.image_datasets[0].Rows,
                self.image_datasets[0].Columns,
                len(self.image_datasets))

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputOrigin(origin)
        resampler.SetOutputSpacing(spacing)
        resampler.SetOutputDirection(direction)
        resampler.SetSize(size)
        resampler.SetTransform(sitk.Transform())
        resampler.SetOutputPixelType(sitk.sitkUInt8)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        nifti_image = resampler.Execute(nifti_image)

        # Convert to binary
        nifti_image = sitk.BinaryThreshold(nifti_image, 1, 255, 1, 0)

        return nifti_image

    def _get_3d_model(self, nifti_image: sitk.Image) -> vtk_dm.vtkPolyData:
        # Cast the image to vtkImageData
        itk_image = convert_to_itk_image(nifti_image)
        vtk_image = itk.vtk_image_from_image(itk_image)

        # MAYBE: set the direction matrix
        vtk_image.SetDirectionMatrix(nifti_image.GetDirection())

        # Apply gaussian smoothing
        gaussian = vtk_igen.vtkImageGaussianSmooth()
        gaussian.SetInputDataObject(0, vtk_image)
        gaussian.SetStandardDeviation(2.0)
        gaussian.SetRadiusFactor(1.0)

        # Apply flying edges
        flying_edges = vtk_fcore.vtkFlyingEdges3D()
        flying_edges.SetInputConnection(0, gaussian.GetOutputPort(0))
        flying_edges.SetValue(0, 127.5)
        flying_edges.ComputeGradientsOff()
        flying_edges.ComputeNormalsOff()

        # Apply decimation
        decimate = vtk_fcore.vtkDecimatePro()
        decimate.SetInputConnection(0, flying_edges.GetOutputPort(0))
        decimate.SetTargetReduction(0.5)
        decimate.PreserveTopologyOn()
        decimate.SetMaximumError(0.02)
        decimate.SplittingOff()
        decimate.SetFeatureAngle(60.)

        # Smooth via the sinc filter
        smoother = vtk_fcore.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(0, decimate.GetOutputPort(0))
        smoother.SetNumberOfIterations(15)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetFeatureAngle(60.0)
        smoother.SetPassBand(0.25)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOff()

        # Get normals
        normals = vtk_fcore.vtkPolyDataNormals()
        normals.SetInputConnection(0, smoother.GetOutputPort(0))
        normals.SetFeatureAngle(60.0)

        # Strip the polydata
        stripper = vtk_fcore.vtkStripper()
        stripper.SetInputConnection(0, normals.GetOutputPort(0))
        stripper.JoinContiguousSegmentsOn()
        stripper.Update(0)

        return stripper.GetOutput()

    def _get_2d_contours(self, polydata: vtk_dm.vtkPolyData) -> List[List[List[List[float]]]]:
        origin = self.image_datasets[0].ImagePositionPatient
        origin = [float(val) for val in origin]
        first_pos = self.image_datasets[0].ImagePositionPatient
        last_pos = self.image_datasets[-1].ImagePositionPatient
        length = np.abs(np.linalg.norm(np.array(last_pos) - np.array(first_pos)))
        normal = tuple(np.cross(np.array(self.image_datasets[0].get('ImageOrientationPatient')[0:3]),
                                np.array(self.image_datasets[0].get('ImageOrientationPatient')[3:6])))

        # Create the plane
        plane = vtk_dm.vtkPlane()
        plane.SetOrigin(*origin)
        plane.SetNormal(*normal)

        # Create the cutter
        cutter = vtk_fcore.vtkCutter()
        cutter.SetInputDataObject(0, polydata)
        cutter.SetCutFunction(plane)
        cutter.GenerateTrianglesOn()
        cutter.GenerateValues(len(self.image_datasets), 0, length)

        # Create the cleaner
        cleaner = vtk_fcore.vtkCleanPolyData()
        cleaner.SetInputConnection(0, cutter.GetOutputPort(0))
        cleaner.SetAbsoluteTolerance(0.1)
        cleaner.SetPointMerging(True)
        cleaner.Update(0)

        # Get the polylines
        loop = vtk_fmodel.vtkContourLoopExtraction()
        loop.SetInputConnection(0, cleaner.GetOutputPort(0))
        loop.SetOutputModeToPolygons()
        loop.SetNormal(*normal)
        loop.Update(0)

        # Get the polylines
        cells = loop.GetOutput().GetPolys()
        points = loop.GetOutput().GetPoints()

        indices = vtk_ccore.vtkIdList()
        contours_points = []
        poly_count = 0
        while cells.GetNextCell(indices):
            contour_points = []
            for i in range(indices.GetNumberOfIds()):
                point = list(points.GetPoint(indices.GetId(i)))
                contour_points.append(point)

            contours_points.append(contour_points)
            print(f'Contour {poly_count}: {len(contour_points)} points')
            poly_count += 1

        return [contours_points,]

    def create_contour_sequence(self,
                                coords: List[List[List[List[float]]]],
                                ) -> Sequence:
        contour_sequence = Sequence()

        for slice_dataset, slice_coords in zip(self.image_datasets, coords):
            for contour_data in slice_coords:
                contour_image = Dataset()
                contour_image.ReferencedSOPClassUID = slice_dataset.file_meta.MediaStorageSOPClassUID
                contour_image.ReferencedSOPInstanceUID = slice_dataset.file_meta.MediaStorageSOPInstanceUID

                contour_image_sequence = Sequence()
                contour_image_sequence.append(contour_image)

                contour = Dataset()
                contour.ContourImageSequence = contour_image_sequence
                contour.ContourGeometricType = 'CLOSED_PLANAR'
                contour.NumberOfContourPoints = len(contour_data)
                contour.ContourData = [coord for point in contour_data for coord in point]
                contour_sequence.append(contour)

        return contour_sequence

    def add_roi_contour_sequence(self,
                                 rtss: Dataset,
                                 contours: List[List[List[List[float]]]],
                                 color: Tuple[int, int, int],
                                 roi_number: int,
                                 ) -> None:
        """Add a ROIContourSequence to the RTSS dataset."""
        roi_contour = Dataset()
        roi_contour.ROIDisplayColor = list(color)
        roi_contour.ReferencedROINumber = str(roi_number)
        roi_contour.ContourSequence = self.create_contour_sequence(contours)

        rtss.ROIContourSequence.append(roi_contour)

    def add_structure_set_roi_sequence(self,
                                       rtss: Dataset,
                                       roi_name: str,
                                       roi_number: int,
                                       roi_generation_algorithm: str = 'AUTOMATIC'
                                       ) -> None:
        structure_set_roi = Dataset()
        structure_set_roi.ROINumber = roi_number
        structure_set_roi.ReferencedFrameOfReferenceUID = self.image_datasets[0].get('FrameOfReferenceUID')
        structure_set_roi.ROIName = roi_name
        structure_set_roi.ROIDescription = ''
        structure_set_roi.ROIGenerationAlgorithm = roi_generation_algorithm
        rtss.StructureSetROISequence.append(structure_set_roi)

    @staticmethod
    def add_rt_roi_observations_sequence(rtss: Dataset,
                                         roi_number: int,
                                         ) -> None:
        rt_roi_observations = Dataset()
        rt_roi_observations.ObservationNumber = roi_number
        rt_roi_observations.ReferencedROINumber = roi_number
        rt_roi_observations.RTROIInterpretedType = ''
        rt_roi_observations.ROIInterpreter = ''
        rt_roi_observations.ROIObservationDescription = 'Type:Soft,Range:*/*,Fill:0,Opacity:0.0,Thickness:1,' \
                                                        'LineThickness:2,read-only:false'
        rt_roi_observations.private_creators = 'University of Bern, Switzerland'
        rtss.RTROIObservationsSequence.append(rt_roi_observations)


    def convert(self) -> Any:
        rtss = self._generate_basic_rtss()

        for idx, (label_image, label_name, color) in enumerate(zip(self.label_images, self.roi_names, self.colors)):

            # preprocess the image
            label_image = self.preprocess_image(label_image)

            # Get polydata
            polydata = self._get_3d_model(label_image)

            # Get the contour data
            contours = self._get_2d_contours(polydata)

            # enhance the rtss with the data
            self.add_structure_set_roi_sequence(rtss, label_name, idx + 1)
            self.add_roi_contour_sequence(rtss, contours, color, idx + 1)
            self.add_rt_roi_observations_sequence(rtss, idx + 1)

        return rtss


if __name__ == '__main__':
    dicom_data_dir = 'D:/experiment_data_9/dicom0/VS-SEG-001'
    nifti_file_path = 'D:/experiment_data_9/nifti0/VS-SEG-001/seg_VS-SEG-001_NA_circle50.nii.gz'
    get_3d_reconstruction(nifti_file_path, dicom_data_dir)
