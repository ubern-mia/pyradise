import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import pyradise.data as ps_data
import pyradise.fileio as ps_fio
import pyradise.process as ps_proc
from examples.inference.network import UNet


class ExampleInferenceFilter(ps_proc.InferenceFilter):
    """An example implementation of an InferenceFilter for
    slice-wise segmentation with a PyTorch-based U-Net."""

    def __init__(self) -> None:
        super().__init__()

        # Define the device on which the model should be run
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define a class attribute for the model
        self.model: Optional[nn.Module] = None

    def _prepare_model(self, model: nn.Module, model_path: str) -> nn.Module:
        """Implementation using the PyTorch framework."""

        # Load model parameters
        model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Assign the model to the class
        self.model = model.to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        return model

    def _infer_on_batch(self, batch: Dict[str, Any], params: ps_proc.InferenceFilterParams) -> Dict[str, Any]:
        """Implementation using the PyTorch framework."""

        # Stack and adjust the numpy array such that it fits the
        # [batch, channel / images, height, width, (depth)] format
        # Note: The following statement works for slice-wise and patch-wise processing
        if (loop_axis := params.indexing_strategy.loop_axis) is None:
            adjusted_input = np.stack(batch["data"], axis=0)
        else:
            adjusted_input = np.stack(batch["data"], axis=0).squeeze(loop_axis + 2)

        # Generate a tensor from the numpy array
        input_tensor = torch.from_numpy(adjusted_input)

        # Move the batch to the same device as the model
        input_tensor = input_tensor.to(self.device, dtype=torch.float32)

        # Apply the model to the batch
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        # Retrieve the predicted classes from the output
        final_activation_fn = nn.Sigmoid()
        output_tensor = (final_activation_fn(output_tensor) > 0.5).bool()

        # Convert the output to a numpy array
        # Note: The output shape must be [batch, height, width, (depth)]
        output_array = output_tensor.cpu().numpy()

        # Construct a list of output arrays such that it fits the index expressions
        batch_output_list = [output_array[i, ...] for i in range(output_array.shape[0])]

        # Combine the output arrays into a dictionary
        output = {"data": batch_output_list, "index_expr": batch["index_expr"]}

        return output


def get_pipeline(model_path: str) -> ps_proc.FilterPipeline:
    # Construct a pipeline the processing
    pipeline = ps_proc.FilterPipeline()

    # Construct and ddd the preprocessing filters to the pipeline
    output_size = (256, 256, 256)
    output_spacing = (1.0, 1.0, 1.0)
    reference_modality = "T1"
    resample_filter_params = ps_proc.ResampleFilterParams(
        output_size, output_spacing, reference_modality=reference_modality, centering_method="reference"
    )
    resample_filter = ps_proc.ResampleFilter()
    pipeline.add_filter(resample_filter, resample_filter_params)

    norm_filter_params = ps_proc.ZScoreNormFilterParams()
    norm_filter = ps_proc.ZScoreNormFilter()
    pipeline.add_filter(norm_filter, norm_filter_params)

    # Construct and add the inference filter
    modalities_to_use = ("T1", "T2")
    inf_params = ps_proc.InferenceFilterParams(
        model=UNet(num_channels=2, num_classes=1),
        model_path=model_path,
        modalities=modalities_to_use,
        reference_modality=reference_modality,
        output_organs=(ps_data.Organ("Skull"),),
        output_annotator=ps_data.Annotator("AutoSegmentation"),
        organ_indices=(1,),
        batch_size=8,
        indexing_strategy=ps_proc.SliceIndexingStrategy(0),
    )

    inf_filter = ExampleInferenceFilter()
    pipeline.add_filter(inf_filter, inf_params)

    # Add postprocessing filters
    cc_filter_params = ps_proc.SingleConnectedComponentFilterParams()
    cc_filter = ps_proc.SingleConnectedComponentFilter()
    pipeline.add_filter(cc_filter, cc_filter_params)

    # Because the spatial properties of the subject images are
    # changed with respect to the reference T1 image a playback
    # of the TransformTape is not required. If the spatial properties
    # of the reference image would have been changed the playback can
    # be achieved using the PlaybackTransformTapeFilter.
    #
    # playback_params = PlaybackTransformTapeFilterParams()
    # playback_filter = PlaybackTransformTapeFilter()
    # pipeline.add_filter(playback_filter, playback_params)

    return pipeline


# pylint: disable=duplicate-code
def test_inference_2d(dicom_test_dataset_path: str, tmpdir: str, model_path: str) -> None:
    # Crawl the data in the input directory
    crawler = ps_fio.SubjectDicomCrawler(dicom_test_dataset_path)
    series_info = crawler.execute()

    # Select the required modalities
    used_modalities = ("T1", "T2")
    modality_selector = ps_fio.ModalityInfoSelector(used_modalities)
    series_info = modality_selector.execute(series_info)

    # Exclude the existing DICOM-RTSS files
    no_rtss_selector = ps_fio.NoRTSSInfoSelector()
    series_info = no_rtss_selector.execute(series_info)

    # Construct the loader and load the subject
    loader = ps_fio.SubjectLoader()
    subject = loader.load(series_info)

    # Construct the pipeline and execute it
    pipeline = get_pipeline(model_path)
    subject = pipeline.execute(subject)

    # Define the customizable metadata for the DICOM-RTSS
    # Note: Check the value formatting at:
    # https://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.2.html
    meta_data = ps_fio.RTSSMetaData(
        patient_name="Jack Demo",
        patient_id=subject.get_name(),
        patient_birth_date="19700101",
        patient_sex="F",
        patient_weight="80",
        patient_size="180",
        series_description="Demo Series Description",
        series_number="10",
        operators_name="Auto-Segmentation Alg.",
    )

    # Convert the segmentations to a DICOM-RTSS
    reference_modality = "T1"
    conv_conf = ps_fio.RTSSConverter2DConfiguration()
    converter = ps_fio.SubjectToRTSSConverter(subject, series_info, reference_modality, conv_conf, meta_data)
    rtss_dataset = converter.convert()

    # Save the new DICOM-RTSS
    output_dir_path = os.path.join(tmpdir, "2d_inference")
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    named_rtss = (("rtss.dcm", rtss_dataset),)
    writer = ps_fio.DicomSeriesSubjectWriter()
    writer.write(named_rtss, output_dir_path, subject.get_name(), series_info)


# pylint: disable=duplicate-code
def test_inference_3d(dicom_test_dataset_path: str, tmpdir: str, model_path: str) -> None:
    # Crawl the data in the input directory
    crawler = ps_fio.SubjectDicomCrawler(dicom_test_dataset_path)
    series_info = crawler.execute()

    # Select the required modalities
    used_modalities = ("T1", "T2")
    modality_selector = ps_fio.ModalityInfoSelector(used_modalities)
    series_info = modality_selector.execute(series_info)

    # Exclude the existing DICOM-RTSS files
    no_rtss_selector = ps_fio.NoRTSSInfoSelector()
    series_info = no_rtss_selector.execute(series_info)

    # Construct the loader and load the subject
    loader = ps_fio.SubjectLoader()
    subject = loader.load(series_info)

    # Construct the pipeline and execute it
    pipeline = get_pipeline(model_path)
    subject = pipeline.execute(subject)

    # Define the customizable metadata for the DICOM-RTSS
    # Note: Check the value formatting at:
    # https://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.2.html
    meta_data = ps_fio.RTSSMetaData(
        patient_name="Jack Demo",
        patient_id=subject.get_name(),
        patient_birth_date="19700101",
        patient_sex="F",
        patient_weight="80",
        patient_size="180",
        series_description="Demo Series Description",
        series_number="10",
        operators_name="Auto-Segmentation Alg.",
    )

    # Convert the segmentations to a DICOM-RTSS
    reference_modality = "T1"
    conv_conf = ps_fio.RTSSConverter3DConfiguration()
    converter = ps_fio.SubjectToRTSSConverter(subject, series_info, reference_modality, conv_conf, meta_data)
    rtss_dataset = converter.convert()

    # Save the new DICOM-RTSS
    output_dir_path = os.path.join(tmpdir, "3d_inference")
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    named_rtss = (("rtss.dcm", rtss_dataset),)
    writer = ps_fio.DicomSeriesSubjectWriter()
    writer.write(named_rtss, output_dir_path, subject.get_name(), series_info)
