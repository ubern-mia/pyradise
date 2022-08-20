import vtkmodules

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import (
    VTK_VERSION_NUMBER,
    vtkLookupTable,
    vtkVersion
)

from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonColor import vtkNamedColors

from vtkmodules.vtkCommonDataModel import (vtkPolyData, vtkPlane)

from vtkmodules.vtkFiltersCore import (
    vtkFlyingEdges3D,
    vtkPolyDataNormals,
    vtkStripper,
    vtkCutter,
    vtkWindowedSincPolyDataFilter,
    vtkAppendPolyData,
    vtkCleanPolyData
)
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkImagingCore import vtkImageThreshold
from vtkmodules.vtkImagingGeneral import vtkImageGaussianSmooth
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)

from pydicom import dcmread, Dataset
from pydicom.sequence import Sequence
from pydicom.tag import Tag


def main():
    # file_name = 'D:/temp/oar_auto.nii.gz'
    # file_name = 'D:/DataBackupsConversion/20210105_ISAS_OAR_conversion_small/ISAS_GBM_009/' \
    #             'seg_ISAS_GBM_009_RP_Hippocampus_R.nii.gz'
    file_name = 'D:/DataBackupsConversion/20210105_ISAS_OAR_conversion_small/ISAS_GBM_009/' \
                'seg_ISAS_GBM_009_RP_Brainstem.nii.gz'
    reader = vtkNIFTIImageReader()
    reader.SetFileName(str(file_name))
    reader.Update(0)

    select_tissue = vtkImageThreshold()
    select_tissue.ThresholdBetween(1, 1)
    select_tissue.SetInValue(255)
    select_tissue.SetOutValue(0)
    select_tissue.SetInputConnection(reader.GetOutputPort())

    gaussian_radius = 1
    gaussian_standard_deviation = 2.0
    gaussian = vtkImageGaussianSmooth()
    gaussian.SetStandardDeviations(gaussian_standard_deviation, gaussian_standard_deviation,
                                   gaussian_standard_deviation)
    gaussian.SetRadiusFactors(gaussian_radius, gaussian_radius, gaussian_radius)
    gaussian.SetInputConnection(select_tissue.GetOutputPort())

    iso_surface = vtkFlyingEdges3D()
    iso_surface.SetInputConnection(gaussian.GetOutputPort())
    iso_surface.ComputeScalarsOff()
    iso_surface.ComputeGradientsOff()
    iso_surface.ComputeNormalsOff()
    iso_value = 127.5
    iso_surface.SetValue(0, iso_value)

    smoothing_iterations = 5
    pass_band = 0.001
    feature_angle = 60.0
    smoother = vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(iso_surface.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOff()
    smoother.Update()

    normals = vtkPolyDataNormals()
    normals.SetInputConnection(smoother.GetOutputPort())
    normals.SetFeatureAngle(feature_angle)

    stripper = vtkStripper()
    stripper.SetInputConnection(normals.GetOutputPort())
    stripper.Update()

    contour_actor = get_contours(stripper.GetOutput())

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(stripper.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)

    renderer_left = vtkRenderer()
    renderer_left.SetViewport(0, 0, 0.5, 1)
    renderer_right = vtkRenderer()
    renderer_right.SetViewport(0.5, 0, 1, 1)
    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer_left)
    render_window.AddRenderer(renderer_right)
    render_window_interactor = vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    renderer_left.AddActor(actor)
    renderer_right.AddActor(contour_actor)

    render_window.SetSize(640, 480)
    render_window.SetWindowName('FrogBrain')
    render_window.Render()

    render_window_interactor.Start()


def get_contours(polydata: vtkPolyData):
    bounds = polydata.GetBounds()
    step_size = 0.5

    minimum_value = bounds[2] - step_size
    maximum_value = bounds[3] + step_size

    # position = minimum_value

    # while position < maximum_value:
    #     plane = vtkPlane()
    #     plane.SetOrigin((bounds[0] + bounds[1]) // 2,
    #                     (bounds[2] + bounds[3]) // 2,
    #                     position)
    #     plane.SetNormal(1, 0, 0)
    #
    #     cutter = vtkCutter()
    #     cutter.SetInputData(polydata)
    #     cutter.SetCutFunction(plane)
    #     cutter.Update()
    #     all_contours.append(cutter.GetOutput())
    #
    #     position += step_size

    # append = vtkAppendPolyData()
    # [append.AddInputData(contour) for contour in all_contours]
    # append.Update()

    plane = vtkPlane()
    plane.SetOrigin(bounds[0] - 1, bounds[2] - 1, bounds[4] - 1)
    plane.SetNormal(0, 1, 0)

    cutter = vtkCutter()
    cutter.SetInputData(polydata)
    cutter.SetCutFunction(plane)
    cutter.GenerateValues(50, minimum_value, maximum_value)
    cutter.Update()

    cleaner = vtkCleanPolyData()
    cleaner.SetInputData(cutter.GetOutput())
    cleaner.Update()

    mapper = vtkPolyDataMapper()
    mapper.SetInputData(cleaner.GetOutput())
    mapper.ScalarVisibilityOff()

    colors = vtkNamedColors()

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d('Orange'))
    actor.GetProperty().SetLineWidth(2)

    return actor


# def get_dicom_series_info(path: str):



if __name__ == '__main__':
    main()
