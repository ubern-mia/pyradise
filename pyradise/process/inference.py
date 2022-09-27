import os.path
from typing import (
    Optional,
    Tuple)

import numpy as np

from pyradise.data import (
    Subject,
    Modality,
    TransformInfo)
from .base import (
    LoopEntryFilter,
    LoopEntryFilterParams)

__all__ = ['InferenceFilterParams', 'InferenceFilter']


class InferenceFilterParams(LoopEntryFilterParams):
    """A filter parameter class for the prototype :class:`~pyradise.process.inference.InferenceFilter` class.

    Args:
        model (object): The model to apply.
        model_path (Optional[str]): The path to the model parameters.
        modalities (Tuple[Modality, ...]): The :class:`~pyradise.data.modality.Modality` s of the
         :class:`~pyradise.data.image.IntensityImage` instances to use for inference.
        loop_axis (Optional[int]): The axis to loop over. If None, the filter will be applied to the whole image at
         once (default: None).
    """

    def __init__(self,
                 model: object,
                 model_path: Optional[str],
                 modalities: Tuple[Modality, ...],
                 loop_axis: Optional[int] = None,
                 ) -> None:
        super().__init__(loop_axis)

        # the model to apply
        self.model = model

        # the path to the model parameters
        if model_path is not None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path '{model_path}' is invalid.")
            self.model_path: Optional[str] = model_path
        else:
            self.model_path: Optional[str] = None

        # the modalities to use for inference
        # -> the modalities must be in the correct order to construct the model input
        self.modalities = modalities


class InferenceFilter(LoopEntryFilter):
    """A prototype filter class for applying a DL-model to a :class:`~pyradise.data.subject.Subject` instance.

    This class is a prototype for applying a DL-model to a :class:`~pyradise.data.subject.Subject` instance. PyRaDiSe
    provides just a prototype for this filter because we want to stay framework-agnostic. Therefore, the actual
    implementation of the DL-framework specific methods must be implemented in a subclass.
    """

    @staticmethod
    def is_invertible() -> bool:
        """Returns whether the filter is invertible or not.

        Note:
            If your DL model is invertible, you should override this method and return ``True``.

        Returns:
            bool: False because the inference filter is typically not invertible.
        """
        return False

    @staticmethod
    def get_input_array(subject: Subject,
                        params: InferenceFilterParams
                        ) -> np.ndarray:
        """Return the input array for the DL-model.

        Note:
            This function returns the concatenated data in C (channels) x H (height) x W (width) x D (depth) order.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance.
            params (InferenceFilterParams): The filter parameters.

        Returns:
            np.ndarray: The input array for the DL-model.
        """
        # get the images for the modalities
        images = [subject.get_image_by_modality(modality) for modality in params.modalities]

        # get the image arrays and stack them
        input_array = np.stack([image.get_image_data_as_np() for image in images], axis=0)

        return input_array

    def load_model_parameters(self,
                              model: object,
                              model_path: str
                              ) -> object:
        """Load the model parameters from the provided path.

        Args:
            model (object): The model instance.
            model_path (str): The path to the model parameters.
        """
        raise NotImplementedError("This method must be implemented for the specific DL-framework.")

    def apply_model(self,
                    model: object,
                    input_array: np.ndarray,
                    params: InferenceFilterParams
                    ) -> np.ndarray:
        """Apply the model to the input array to predict the segmentation.

        Args:
            model (object): The model instance.
            input_array (np.ndarray): The input array for the DL-model.
            params (InferenceFilterParams): The filter parameters.

        Returns:
            np.ndarray: The output array of the DL-model.
        """
        # create an empty output array
        # -> the first dimension is the channel dimension
        # predicted_classes = np.zeros(input_array.shape[1:], dtype=np.uint8)

        # construct the model input finally
        # -> DL-framework specific

        # apply the model
        # -> DL-framework specific
        # for slice-wise inference you can use the loop_entries method of the LoopEntryFilter class

        # convert the output to the predicted classes
        # -> DL-framework specific

        # return the predicted classes
        # -> DL-framework specific

        raise NotImplementedError("This method must be implemented for the specific DL-framework.")

    def array_to_subject(self,
                         output_array: np.ndarray,
                         subject: Subject,
                         params: InferenceFilterParams
                         ) -> Subject:
        """Convert the output array of the DL-model to one or multiple :class:`~pyradise.data.image.SegmentationImage`
        instances and add them to the provided :class:`~pyradise.data.subject.Subject` instance.

        Args:
            output_array (np.ndarray): The output array of the DL-model.
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to which the new
             :class:`~pyradise.data.image.SegmentationImage` instances will be added.
            params (InferenceFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with the new
            :class:`~pyradise.data.image.SegmentationImage` instances added.
        """
        raise NotImplementedError("This method must be implemented for the specific DL-framework.")

    def execute(self,
                subject: Subject,
                params: InferenceFilterParams
                ) -> Subject:
        """Execute the filter on the provided :class:`~pyradise.data.subject.Subject` instance.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance.
            params (InferenceFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with the newly added
            :class:`~pyradise.data.image.SegmentationImage` instances.
        """

        # get the input array for the DL-model
        input_array = self.get_input_array(subject, params)

        # load the model parameters
        if params.model_path is not None:
            model = self.load_model_parameters(params.model, params.model_path)
        else:
            model = params.model

        # apply the model
        output_array = self.apply_model(model, input_array, params)

        # construct the output image
        subject = self.array_to_subject(output_array, subject, params)

        return subject

    def execute_inverse(self,
                        subject: Subject,
                        transform_info: TransformInfo
                        ) -> Subject:
        """Return the provided :class:`~pyradise.data.subject.Subject` instance without any processing because
        the inference of a DL-model is typically not invertible.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be returned.
            transform_info (TransformInfo): The transform information.

        Returns:
            Subject: The provided :class:`~pyradise.data.subject.Subject` instance.
        """
        return subject
