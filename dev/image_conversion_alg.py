import os
import shutil
import typing as t

import cv2
import cv2 as cv
import itk
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import vtkmodules.all as vtk2
import vtkmodules.vtkCommonCore as vtk_ccore
import vtkmodules.vtkCommonDataModel as vtk_dm
import vtkmodules.vtkFiltersCore as vtk_fcore
import vtkmodules.vtkFiltersGeometry as vtk_fgeom
import vtkmodules.vtkFiltersModeling as vtk_fmodel
import vtkmodules.vtkImagingCore as vtk_icore
import vtkmodules.vtkImagingGeneral as vtk_igen
from pydicom import Dataset

import pyradise.data as dat
import pyradise.fileio as fio
import pyradise.process as proc
from pyradise.utils import convert_to_itk_image, load_datasets


def get_datasets(path: str) -> t.Tuple[Dataset]:
    files = [file.path for file in os.scandir(path) if file.is_file() and file.name.endswith(".dcm")]
    datasets = load_datasets(files)
    return datasets


def get_sitk_image(path: str) -> sitk.Image:
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    image = reader.Execute()

    image = sitk.DICOMOrient(image, "LPS")
    image_np = sitk.GetArrayFromImage(image)
    image_np[image_np != 0] = 255
    image_2 = sitk.GetImageFromArray(image_np)
    image_2.CopyInformation(image)
    return image_2


def convert_to_sitk(input_path: str, output_path: str) -> None:
    series_info = fio.SubjectDicomCrawler(input_path).execute()

    selection = fio.ModalityInfoSelector(keep=("T1c",))
    series_info = selection.execute(series_info)

    loader = fio.SubjectLoader()
    subject = loader.load(series_info)

    nii_path = os.path.join(output_path, "nii")
    if not os.path.exists(nii_path):
        os.makedirs(nii_path)

    fio.SubjectWriter().write(nii_path, subject, False)

    dcm_path = os.path.join(output_path, "dcm")
    if not os.path.exists(dcm_path):
        os.makedirs(dcm_path)
    for info in series_info:
        if isinstance(info, fio.DicomSeriesImageInfo):
            for path in info.get_path():
                new_dcm_path = os.path.join(dcm_path, os.path.basename(path))
                shutil.copy(path, new_dcm_path)


def get_vtk_image(sitk_image: sitk.Image) -> vtk_dm.vtkImageData:
    # cast the image to vtkImageData
    itk_image = convert_to_itk_image(sitk_image)
    vtk_image = itk.vtk_image_from_image(itk_image)

    # set the direction matrix
    vtk_image.SetDirectionMatrix(sitk_image.GetDirection())
    return vtk_image


def get_smooth_image(image_vtk: vtk_dm.vtkImageData, sigma: float = 1.0, radius: int = 1) -> vtk_dm.vtkImageData:
    # Smooth the image
    smooth = vtk_igen.vtkImageGaussianSmooth()
    smooth.SetInputData(image_vtk)
    smooth.SetDimensionality(3)
    smooth.SetRadiusFactors(radius, radius, radius)
    smooth.SetStandardDeviations(sigma, sigma, sigma)
    smooth.Update()

    threshold = vtk2.vtkImageThreshold()
    threshold.SetInputData(smooth.GetOutput())
    threshold.ThresholdBetween(1, 255)
    threshold.SetInValue(255)
    threshold.SetOutValue(0)
    threshold.Update()

    return threshold.GetOutput()


def get_cube_model(image_vtk: vtk_dm.vtkImageData):
    padder = vtk2.vtkImageWrapPad()
    padder.SetInputDataObject(0, image_vtk)
    extent = image_vtk.GetExtent()
    padder.SetOutputWholeExtent(extent[0], extent[1] + 1, extent[2], extent[3] + 1, extent[4], extent[5] + 1)
    padder.Update(0)
    padder.GetOutput().GetCellData().SetScalars(image_vtk.GetPointData().GetScalars())

    # threshold the image
    threshold = vtk2.vtkThreshold()
    threshold.SetInputArrayToProcess(
        0, 0, 0, vtk2.vtkDataObject.FIELD_ASSOCIATION_CELLS, vtk2.vtkDataSetAttributes.SCALARS
    )
    threshold.SetInputConnection(0, padder.GetOutputPort(0))
    threshold.SetLowerThreshold(1)
    threshold.SetUpperThreshold(255)
    threshold.Update(0)
    threshold_object = threshold.GetOutput()

    # get the geometry using the vtkGeometryFilter
    geometry = vtk_fgeom.vtkGeometryFilter()
    geometry.SetInputData(threshold_object)
    geometry.Update(0)

    model = geometry.GetOutput()

    show_polydata(model, show_edges=True, line_width=1)

    return None


def full_pipeline(
    sitk_image: sitk.Image,
    image_smoothing: bool,
    smooth_sigma: float,
    smooth_radius: float,
    smooth_threshold: float,
    decimate_reduction: float,
    decimate_threshold: float,
    model_smooth_iter: int,
    model_smooth_pass_band: float,
) -> vtk_dm.vtkPolyData:
    # cast the image to vtkImageData
    itk_image = convert_to_itk_image(sitk_image)
    vtk_image = itk.vtk_image_from_image(itk_image)

    # set the direction matrix
    vtk_image.SetDirectionMatrix(sitk_image.GetDirection())

    # pad the image to avoid boundary effects
    extent = vtk_image.GetExtent()
    new_extent = (extent[0] - 5, extent[1] + 5, extent[2] - 5, extent[3] + 5, extent[4] - 5, extent[5] + 5)
    padder = vtk_icore.vtkImageConstantPad()
    padder.SetInputDataObject(0, vtk_image)
    padder.SetOutputWholeExtent(new_extent)
    padder.Update(0)
    vtk_image = padder.GetOutput()

    # apply gaussian smoothing
    foreground_amount = sitk.GetArrayFromImage(sitk_image).sum() / 255
    if foreground_amount > smooth_threshold and image_smoothing:
        gaussian = vtk_igen.vtkImageGaussianSmooth()
        gaussian.SetInputDataObject(0, vtk_image)
        gaussian.SetStandardDeviation(smooth_sigma)
        gaussian.SetRadiusFactor(smooth_radius)
        gaussian.Update(0)
        vtk_image = gaussian.GetOutputDataObject(0)

    # apply flying edges
    flying_edges = vtk_fcore.vtkFlyingEdges3D()
    flying_edges.SetInputDataObject(0, vtk_image)
    flying_edges.SetValue(0, 127.5)
    flying_edges.ComputeGradientsOff()
    flying_edges.ComputeNormalsOff()
    flying_edges.Update(0)
    model = flying_edges.GetOutputDataObject(0)

    show_polydata(model, show_edges=True, line_width=1)

    # apply decimation
    if foreground_amount > decimate_threshold:
        decimate = vtk_fcore.vtkDecimatePro()
        decimate.SetInputConnection(0, flying_edges.GetOutputPort(0))
        decimate.SetTargetReduction(decimate_reduction)
        decimate.PreserveTopologyOn()
        decimate.SetMaximumError(1)
        decimate.SplittingOff()
        decimate.SetFeatureAngle(60.0)
        decimate.Update(0)
        model = decimate.GetOutputDataObject(0)

        show_polydata(model, show_edges=True, line_width=1)

    # smooth via the sinc filter
    if model_smooth_iter > 0:
        smoother = vtk_fcore.vtkWindowedSincPolyDataFilter()
        smoother.SetInputDataObject(0, model)
        smoother.SetNumberOfIterations(model_smooth_iter)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetFeatureAngle(60.0)
        smoother.SetPassBand(model_smooth_pass_band)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOff()
        smoother.Update(0)
        model = smoother.GetOutputDataObject(0)

        show_polydata(model, show_edges=True, line_width=1)

    # get normals
    normals = vtk_fcore.vtkPolyDataNormals()
    normals.SetInputDataObject(0, model)
    normals.SetFeatureAngle(60.0)

    # strip the polydata
    stripper = vtk_fcore.vtkStripper()
    stripper.SetInputConnection(0, normals.GetOutputPort(0))
    stripper.JoinContiguousSegmentsOn()
    stripper.Update(0)
    stripped = stripper.GetOutput()

    show_polydata(stripped, show_edges=True, line_width=1)

    return stripped


def slicing_2d(
    polydata: vtk_dm.vtkPolyData,
    image_datasets: t.Tuple[Dataset],
):
    origin = image_datasets[0].ImagePositionPatient
    origin = [float(val) for val in origin]
    first_pos = image_datasets[0].ImagePositionPatient
    last_pos = image_datasets[-1].ImagePositionPatient
    length = np.abs(np.linalg.norm(np.array(last_pos) - np.array(first_pos)))
    slice_spacing = length / (len(image_datasets) - 1)
    normal = tuple(
        np.cross(
            np.array(image_datasets[0].get("ImageOrientationPatient")[0:3]),
            np.array(image_datasets[0].get("ImageOrientationPatient")[3:6]),
        )
    )

    # create the initial cutting plane
    plane = vtk_dm.vtkPlane()
    plane.SetOrigin(*origin)
    plane.SetNormal(*normal)

    # plane_source = vtk2.vtkPlaneSource()
    # plane_source.SetOrigin(*origin)
    # plane_source.SetNormal(*normal)
    # plane_source.Update()
    #
    # show_polydata(plane_source.GetOutput(), show_edges=True, line_width=1)

    # create the cutter
    cutter = vtk_fcore.vtkCutter()
    cutter.SetInputDataObject(0, polydata)
    cutter.SetCutFunction(plane)
    cutter.GenerateTrianglesOn()
    cutter.GenerateValues(len(image_datasets), 0, length)
    cutter.Update(0)

    # show_polydata(cutter.GetOutput(), show_edges=True, line_width=1)

    # create the cleaner
    cleaner = vtk_fcore.vtkCleanPolyData()
    cleaner.SetInputConnection(0, cutter.GetOutputPort(0))
    cleaner.SetAbsoluteTolerance(0.01)
    cleaner.PointMergingOn()
    cleaner.Update(0)

    # show_polydata(cleaner.GetOutput(), show_edges=True, line_width=2)

    # get the polylines
    loop = vtk_fmodel.vtkContourLoopExtraction()
    loop.SetInputConnection(0, cleaner.GetOutputPort(0))
    loop.SetOutputModeToPolylines()
    loop.SetNormal(*normal)
    loop.Update(0)
    looped = loop.GetOutput()

    show_polydata(looped, show_edges=True, line_width=2)

    # get the polylines for each slice if there are any
    cells = looped.GetLines()
    points = looped.GetPoints()
    contours_points = []

    indices = vtk_ccore.vtkIdList()
    cell_indicator = cells.GetNextCell(indices)

    for slice_idx, dataset in enumerate(image_datasets):
        slice_plane = vtk_dm.vtkPlane()
        slice_plane.SetOrigin(*dataset.ImagePositionPatient)
        slice_plane.SetNormal(*normal)

        if cell_indicator:
            point = points.GetPoint(indices.GetId(0))
            distance = slice_plane.DistanceToPlane(point)
        else:
            distance = 2 * slice_spacing

        if distance <= slice_spacing and cell_indicator == 1:
            contour_points = []
            for i in range(indices.GetNumberOfIds()):
                point = list(points.GetPoint(indices.GetId(i)))
                contour_points.append(point)
            contours_points.append(contour_points)
            cell_indicator = cells.GetNextCell(indices)

        else:
            contours_points.append(None)

    return contours_points


def show_polydata(
    polydata: vtk_dm.vtkPolyData, camera_origin: t.Tuple = (5, -110, 80), show_edges: bool = False, line_width: int = 4
) -> None:
    from vtkmodules.vtkCommonColor import vtkNamedColors
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkPolyDataMapper,
        vtkRenderer,
        vtkRenderWindow,
        vtkRenderWindowInteractor,
    )

    # Visualize
    colors = vtkNamedColors()

    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtkActor()
    actor.SetMapper(mapper)
    if show_edges:
        actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetLineWidth(line_width)
    actor.GetProperty().SetColor(colors.GetColor3d("Orange"))

    renderer = vtkRenderer()
    renderWindow = vtkRenderWindow()
    renderWindow.SetWindowName("Line")
    renderWindow.SetSize(800, 800)
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    renderer.SetBackground(*colors.GetColor3d("White"))
    renderer.AddActor(actor)

    renderer.GetActiveCamera().SetPosition(*camera_origin)
    renderer.GetActiveCamera().SetFocalPoint(1.7, 37, -5.2)
    renderer.GetActiveCamera().SetViewUp(0, 1, 0)
    renderer.GetActiveCamera()
    renderer.GetActiveCamera().SetDistance(15.0)

    renderWindow.Render()
    renderWindowInteractor.Start()


def main(dcm_dir_path: str, nii_dir_path: str) -> None:
    # Load the datasets
    datasets = get_datasets(dcm_dir_path)

    # Load the image
    image_path = os.path.join(nii_dir_path, "seg_ISAS_GBM_005_Robert_Poel_Brainstem.nii.gz")
    image = get_sitk_image(image_path)

    # Get the vtk image
    vtk_image = get_vtk_image(image)

    # extract the geometry
    vtk_image2 = vtk2.vtkImageData()
    vtk_image2.DeepCopy(vtk_image)
    get_cube_model(vtk_image2)

    # smooth the image
    vtk_image2 = vtk2.vtkImageData()
    vtk_image2.DeepCopy(vtk_image)
    smoothed_image = get_smooth_image(vtk_image2)
    get_cube_model(smoothed_image)

    # Get the 3D model
    polydata = full_pipeline(image, True, 2.0, 2.0, 100, 0.6, 100, 15, 0.002)

    # Get the 2D contours
    contours = slicing_2d(polydata, datasets)


def main2d(dcm_dir_path: str, nii_dir_path: str) -> None:
    # Load the image
    image_path = os.path.join(nii_dir_path, "seg_ISAS_GBM_005_Robert_Poel_Brainstem.nii.gz")
    image = get_sitk_image(image_path)

    # # plot the original image
    path_img = os.path.join(dcm_dir_path, "img_0.png")
    img_np = sitk.GetArrayFromImage(image)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_off()
    ax.imshow(img_np[70 : (70 + 112), 103 : (103 + 112), 90], cmap="gray_r")
    fig.savefig(path_img, bbox_inches="tight", pad_inches=0)
    # plt.show()
    plt.close(fig)

    # plot the smoothed image
    path_img = os.path.join(dcm_dir_path, "img_1.png")
    img2 = sitk.DiscreteGaussian(image, 1.5)
    img_np2 = sitk.GetArrayFromImage(img2)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img_np2[70 : (70 + 112), 103 : (103 + 112), 90], cmap="gray_r")
    fig.tight_layout()
    ax.set_axis_off()
    # plt.show()
    fig.savefig(path_img, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # plot the contour image
    img_np = sitk.GetArrayFromImage(image)
    img_np3 = img_np[:, :, 90]
    empty = np.zeros_like(img_np3)
    contour, _ = cv2.findContours(img_np3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(empty, contour, color=1)

    erode = sitk.BinaryErodeImageFilter()
    erode.SetKernelRadius(1)
    erode.SetKernelType(sitk.sitkBall)
    erode.SetBackgroundValue(0)
    erode.SetForegroundValue(1)
    erode.SetNumberOfThreads(1)
    erode.SetForegroundValue(255)
    img_err = erode.Execute(image)

    # img_err = sitk.BinaryErode(image, (1, 1, 1))
    img_np4 = sitk.GetArrayFromImage(img_err)
    img_np4 = img_np4[:, :, 90]
    empty[img_np4 != 0] = 0

    path_img = os.path.join(dcm_dir_path, "img_3.png")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(empty[70 : (70 + 112), 103 : (103 + 112)], cmap="gray_r")
    ax.set_axis_off()
    # plt.show()
    fig.savefig(path_img, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    #
    #
    # img3 = sitk.SimpleContourExtractor(img2)
    #
    # contour = sitk.BinaryContourImageFilter()
    # contour.SetFullyConnected(True)
    # contour.SetBackgroundValue(0)
    # contour.SetForegroundValue(1)
    # img3 = contour.Execute(img2)
    #
    # path_img = os.path.join(dcm_dir_path, 'img_2.png')
    # img_np3 = sitk.GetArrayFromImage(img3)
    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.imshow(img_np3[:, :, 90], cmap='gray')
    # fig.tight_layout()
    # ax.set_axis_off()
    # plt.show()
    # plt.close(fig)


if __name__ == "__main__":
    original_dcm_dir = "D:/image/original/ISAS_GBM_005"
    output_convert_dir = "D:/image/curated/"

    conv_dcm_dir = "D:/image/curated/dcm"
    conv_nii_dir = "D:/image/curated/nii"

    img_out = "D:/image/"

    # Convert the images and generate NIfTI and DICOM files
    # convert_to_sitk(original_dcm_dir, output_convert_dir)

    # main
    # main(conv_dcm_dir, conv_nii_dir)
    main2d(img_out, conv_nii_dir)
