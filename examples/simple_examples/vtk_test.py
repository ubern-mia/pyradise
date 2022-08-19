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
from vtkmodules.vtkFiltersCore import (
    vtkFlyingEdges3D,
    vtkPolyDataNormals,
    vtkStripper,
    vtkWindowedSincPolyDataFilter
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


def main():
    file_name = 'D:/temp/oar_auto.nii.gz'
    reader = vtkNIFTIImageReader()
    reader.SetFileName(str(file_name))
    reader.Update(0)

    select_tissue = vtkImageThreshold()
    select_tissue.ThresholdBetween(1, 17)
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

    smoothing_iterations = 2
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

    render_window.SetSize(640, 480)
    render_window.SetWindowName('FrogBrain')
    render_window.Render()

    render_window_interactor.Start()


if __name__ == '__main__':
    main()
