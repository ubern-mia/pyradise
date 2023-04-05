import os.path
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import product
from math import ceil
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk

from pyradise.data import (Annotator, IntensityImage, Modality, Organ,
                           SegmentationImage, Subject, TransformInfo,
                           seq_to_modalities, seq_to_organs, str_to_annotator,
                           str_to_modality)

from .base import Filter, FilterParams

__all__ = [
    "InferenceFilterParams",
    "InferenceFilter",
    "IndexingStrategy",
    "SliceIndexingStrategy",
    "PatchIndexingStrategy",
]


class IndexingStrategy(ABC):
    """An abstract class that defines the strategy for indexing / looping over the image data content during model
    inference with an :class:`~pyradise.process.inference.InferenceFilter`. The :class:`IndexingStrategy` is typically
    assigned to the :attr:`~pyradise.process.inference.InferenceFilterParams` for getting used by the
    :class:`~pyradise.process.inference.InferenceFilter`.

    Args:
        loop_axis (Optional[int]): The axis along which the image data should be processed. If None, the image data
         will be processed as a whole.

    .. automethod:: __call__
    """

    def __init__(self, loop_axis: Optional[int]) -> None:
        self.loop_axis: Optional[int] = loop_axis

    @abstractmethod
    def __call__(self, shape: Tuple[int, ...]) -> Tuple[Tuple[slice, ...], ...]:
        """Compute the indexing expressions based on the given shape of the image data and the ``loop_axis`` attribute.

        Args:
            shape (Tuple[int, ...]): The shape of the image data for which the indexing expressions should be computed.

        Returns:
            Tuple[Tuple[slice, ...], ...]: The indexing expressions.
        """
        raise NotImplementedError()


class SliceIndexingStrategy(IndexingStrategy):
    """An indexing strategy class that computes the indexing expressions for slice-wise looping over the image data
    content.

    Args:
        slice_axis (int): The axis along which the image data should be sliced.

    .. automethod:: __call__
    """

    def __init__(self, slice_axis: int) -> None:
        super().__init__(slice_axis)

    def __call__(self, shape: Tuple[int, ...]) -> Tuple[Tuple[slice, ...], ...]:
        """Compute the indexing expressions for each slice based on the given shape of the image data and the
        ``loop_axis`` attribute.

        Args:
            shape (Tuple[int, ...]): The shape of the image data for which the indexing expressions should be computed.

        Returns:
            Tuple[Tuple[slice, ...], ...]: The indexing expressions.
        """
        # get the number of slices
        num_slices = shape[self.loop_axis]

        # create the indexing expressions
        index_expressions = []
        for i in range(num_slices):
            # loop through the axis to build the indexing expression of a single slice
            index_expression = []
            for axis_idx in range(len(shape)):
                # for the slice axis, we want to have just the current slice
                if axis_idx == self.loop_axis:
                    index_expression.append(slice(i, i + 1))

                # for all other axes, we want to have the full image extent
                else:
                    index_expression.append(slice(None))

            index_expressions.append(tuple(index_expression))

        return tuple(index_expressions)


class PatchIndexingStrategy(IndexingStrategy):
    """An indexing strategy class that computes the indexing expressions for patch-wise looping over the image data
    content.

    Args:
        patch_shape (Tuple[int, ...]): The shape of the patches.
        stride (Optional[Tuple[int, ...]]): The stride of the patches. If None, the patches will be extracted with the
         same stride as the patch shape.

    .. automethod:: __call__
    """

    def __init__(self, patch_shape: Tuple[int, ...], stride: Optional[Tuple[int, ...]] = None) -> None:
        super().__init__(None)

        if stride is None:
            stride_ = patch_shape
        else:
            stride_ = stride

        if len(patch_shape) != len(stride_):
            raise ValueError(
                f"Invalid patch shape {patch_shape} and stride {stride_}. "
                "Patch shape and stride must have the same length."
            )

        if any([patch_size <= 0 for patch_size in patch_shape]):
            raise ValueError(f"Invalid patch shape {patch_shape}. Patch shape must be positive.")
        self.patch_shape = patch_shape

        if any([stride_size <= 0 for stride_size in stride_]):
            raise ValueError(f"Invalid stride {stride_}. Stride must be positive.")
        self.stride = stride_

    def __call__(self, shape: Tuple[int, ...]) -> Tuple[Tuple[slice, ...], ...]:
        """Compute the indexing expressions for each patch based on the given shape of the image data and the
        ``patch_shape`` and ``stride`` attributes.

        Args:
            shape (Tuple[int, ...]): The shape of the image data for which the indexing expressions should be computed.

        Returns:
            Tuple[Tuple[slice, ...], ...]: The indexing expressions.
        """
        # get the number of patches in each dimension
        num_patches = tuple(
            range(ceil((shape[i] - self.patch_shape[i]) / self.stride[i]) + 1) for i in range(len(self.patch_shape))
        )

        # get the combinations of patch indices
        patch_indexes = tuple(product(*num_patches))

        # create the indexing expressions
        index_expressions = []
        for patch_index in patch_indexes:
            index_expression = []

            for axis_idx in range(len(self.patch_shape)):
                patch_start = patch_index[axis_idx] * self.stride[axis_idx]
                patch_end = min(patch_start + self.patch_shape[axis_idx], shape[axis_idx])

                # correct the patch start such that each patch is of equal size
                if patch_end - patch_start < self.patch_shape[axis_idx]:
                    patch_start = patch_end - self.patch_shape[axis_idx]

                index_expression.append(slice(patch_start, patch_end))

            index_expressions.append(tuple(index_expression))

        return tuple(index_expressions)


class InferenceFilterParams(FilterParams):
    """A filter parameter class for the prototype :class:`~pyradise.process.inference.InferenceFilter` class.

    Args:
        model (Any): The model to apply.
        model_path (Optional[str]): The path to the model parameters.
        modalities (Tuple[Union[str, Modality], ...]): The :class:`~pyradise.data.modality.Modality` s of the
         :class:`~pyradise.data.image.IntensityImage` instances to use for inference.
        reference_modality (Union[str, Modality]): The :class:`~pyradise.data.modality.Modality` that is used as the
         reference to define the output properties of the created :class:`~pyradise.data.image.SegmentationImage`
         instances.
        output_organs (Tuple[Union[str, Organ], ...]): The organs that get assigned to the created
         :class:`~pyradise.data.image.SegmentationImage` instances.
        output_annotator (Union[str, Annotator]): The annotator that get assigned to the created
         :class:`~pyradise.data.image.SegmentationImage` instances.
        organ_indices (Tuple[int, ...]): The indices of the organs on the output mask of the `model`
         (must match `output_organs` and `output_annotators`).
        batch_size (int): The batch size to use for inference.
        indexing_strategy (IndexingStrategy): The :class:`~pyradise.process.inference.IndexingStrategy` defining how
         the data is fed to the `model`.
    """

    def __init__(
        self,
        model: Any,
        model_path: Optional[str],
        modalities: Tuple[Union[str, Modality], ...],
        reference_modality: Union[str, Modality],
        output_organs: Tuple[Union[str, Organ], ...],
        output_annotator: Union[str, Annotator],
        organ_indices: Tuple[int, ...],
        batch_size: int,
        indexing_strategy: IndexingStrategy,
    ) -> None:
        # adjust the loop_axis because the first axis will be the channel axis
        super().__init__()

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
        self.modalities = seq_to_modalities(modalities)

        # the modality that is used as the reference to define the output properties of the created
        # segmentation images
        self.reference_modality: Modality = str_to_modality(reference_modality)

        # the organs that get assigned to the created segmentation images
        self.output_organs: Tuple[Organ, ...] = seq_to_organs(output_organs)

        # the annotator that get assigned to the created segmentation images
        self.output_annotator: Annotator = str_to_annotator(output_annotator)

        # the indexes of the organs on the output mask of the model (must match output_organs)
        if len(output_organs) != len(organ_indices):
            raise ValueError("Invalid number of organ indices. Must match the number of output_organs.")
        self.organ_indices = organ_indices

        # the batch size
        self.batch_size = batch_size

        # the indexing strategy
        self.indexing_strategy = indexing_strategy


class InferenceFilter(Filter):
    """A prototype filter class for applying a DL-model to a :class:`~pyradise.data.subject.Subject` instance.

    This class is a prototype for applying a DL-model to a :class:`~pyradise.data.subject.Subject` instance. PyRaDiSe
    provides just a prototype for this filter such that it stays DL framework-agnostic. Therefore, the actual
    implementation of the DL-framework specific methods must be implemented in a subclass.

    For implementing a DL-framework specific :class:`InferenceFilter`, the following methods must be implemented:

    * :meth:`~pyradise.process.inference.InferenceFilter._prepare_model`: Prepare the model for inference (e.g. load
      the parameters from a model file).

    * :meth:`~pyradise.process.inference.InferenceFilter._infer_on_batch`: Apply the model to a batch of data such
      that the output shape can be inserted into the new image via the indexing expressions provided by the
      chosen :class:`~pyradise.process.inference.IndexingStrategy`.


    Example:

           Implementation example of a PyTorch-based :class:`InferenceFilter` subclass:

           >>> import torch
           >>> import torch.nn as nn
           >>>
           >>> class ExampleInferenceFilter(InferenceFilter):
           >>>
           >>>  def __init__(self):
           >>>      super().__init__()
           >>>
           >>>      # Define the device on which the model should be run
           >>>      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
           >>>
           >>>      # Define a class attribute for the model
           >>>      self.model: Optional[nn.Module] = None
           >>>
           >>>  def _prepare_model(self, model: nn.Module, model_path: str) -> nn.Module:
           >>>
           >>>      # Load model parameters
           >>>      model.load_state_dict(torch.load(model_path, map_location=self.device))
           >>>
           >>>      # Assign the model to the class
           >>>      self.model = model.to(self.device)
           >>>
           >>>      # Set model to evaluation mode
           >>>      self.model.eval()
           >>>
           >>>      return model
           >>>
           >>>  def _infer_on_batch(self,
           >>>                      batch: Dict[str, Any],
           >>>                      params: InferenceFilterParams
           >>>                      ) -> Dict[str, Any]:
           >>>      # Stack and adjust the numpy array such that it fits the
           >>>      # [batch, channel / images, height, width, (depth)] format
           >>>      # Note: The following statement works for slice-wise and patch-wise processing
           >>>      if (loop_axis := params.indexing_strategy.loop_axis) is None:
           >>>          adjusted_input = np.stack(batch['data'], axis=0)
           >>>      else:
           >>>          adjusted_input = np.stack(batch['data'], axis=0).squeeze(loop_axis + 2)
           >>>
           >>>      # Generate a tensor from the numpy array
           >>>      input_tensor = torch.from_numpy(adjusted_input)
           >>>
           >>>      # Move the batch to the same device as the model
           >>>      input_tensor = input_tensor.to(self.device, dtype=torch.float32)
           >>>
           >>>      # Apply the model to the batch
           >>>      with torch.no_grad():
           >>>          output_tensor = self.model(input_tensor)
           >>>
           >>>      # Retrieve the predicted classes from the output
           >>>      if type(params.indexing_strategy) is SliceIndexingStrategy:
           >>>          # Slice-wise processing
           >>>
           >>>          if len(params.output_organs) > 1:
           >>>              # For multi-class segmentation
           >>>              final_activation_fn = nn.Softmax2d()
           >>>              output_tensor = torch.argmax(final_activation_fn(output_tensor), dim=1)
           >>>
           >>>          else:
           >>>              # For binary segmentation
           >>>              final_activation_fn = nn.Sigmoid()
           >>>              output_tensor = (final_activation_fn(output_tensor) > 0.5).bool()
           >>>
           >>>      elif type(params.indexing_strategy) is PatchIndexingStrategy:
           >>>
           >>>          if len(params.output_organs) > 1:
           >>>              # For multi-class segmentation
           >>>              final_activation_fn = nn.Softmax(dim=1)
           >>>              output_tensor = torch.argmax(final_activation_fn(output_tensor), dim=1)
           >>>
           >>>          else:
           >>>              # For binary segmentation
           >>>              final_activation_fn = nn.Sigmoid()
           >>>              output_tensor = (final_activation_fn(output_tensor) > 0.5).bool()
           >>>
           >>>      else:
           >>>          raise NotImplementedError(f'Indexing strategy {type(params.indexing_strategy).__name__} not'
           >>>                                    'implemented.')
           >>>
           >>>      # Convert the output to a numpy array
           >>>      # Note: The output shape must be [batch, height, width, (depth)]
           >>>      output_array = output_tensor.cpu().numpy()
           >>>
           >>>      # Construct a list of output arrays such that it fits the index expressions
           >>>      batch_output_list = []
           >>>      for i in range(output_array.shape[0]):
           >>>          batch_output_list.append(output_array[i, ...])
           >>>
           >>>      # Combine the output arrays into a dictionary
           >>>      output = {'data': batch_output_list,
           >>>                'index_expr': batch['index_expr']}
           >>>
           >>>      return output

    .. automethod:: _get_input_array
    .. automethod:: _prepare_model
    .. automethod:: _infer_on_batch
    .. automethod:: _apply_model
    .. automethod:: _array_to_subject
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
    def _get_input_array(subject: Subject, params: InferenceFilterParams) -> np.ndarray:
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

        # check if images have the same shape
        if not all(image.get_size() == images[0].get_size() for image in images):
            raise ValueError(
                "All images selected for inference must have the same shape. " "Please resample accordingly."
            )

        # get the image arrays and stack them
        input_array = np.stack([image.get_image_data_as_np() for image in images], axis=0)

        return input_array

    @abstractmethod
    def _prepare_model(self, model: Any, model_path: str) -> None:
        """Prepare the model for inference (e.g. loading the model parameters). The loaded model must be added to
        a class attribute such that it can be accessed by all methods.

        This method must be implemented for the specific DL-framework.

        Args:
            model (Any): The model instance.
            model_path (str): The path to the model parameters.

        Returns:
            Any: The model prepared for inference.
        """
        raise NotImplementedError("This method must be implemented for the specific DL-framework.")

    @abstractmethod
    def _infer_on_batch(self, batch: Dict[str, Any], params: InferenceFilterParams) -> Dict[str, Any]:
        """Apply the model to a batch of data.

        This method must be implemented for the specific DL-framework and is called with a batch of data. The batch
        is a dictionary with the following keys:

        * ``data``: A list of numpy arrays with the input data.

        * ``index_expr``: A list of index expressions for the input data.


        Note:
            The output data in the dictionary must be a list of numpy arrays with the same length as the input data.
            Each ``data`` entry must be a numpy array with the shape [C (channels) x H (height) x W (width) x
            (D (depth))].


        Args:
            batch (Dict[str, Any]): The batch of data.
            params (InferenceFilterParams): The filter parameters.

        Returns:
            Dict[str, Any]: The output of the model.
        """
        raise NotImplementedError("This method must be implemented for the specific DL-framework.")

    def _apply_model(self, input_array: np.ndarray, params: InferenceFilterParams) -> np.ndarray:
        """Apply the model to the input array to predict the segmentation.

        Args:
            input_array (np.ndarray): The input array for the DL-model.
            params (InferenceFilterParams): The filter parameters.

        Returns:
            np.ndarray: The output array of the DL-model.
        """
        # Get the indexes of the data subset for processing
        content_shape = input_array.shape[1:]
        index_expressions = list(params.indexing_strategy(content_shape))

        # Generate an empty array to store the output
        output_array = np.zeros(content_shape, dtype=np.uint8)

        # Iterate over the indexes in batches
        while index_expressions:
            # Construct the batch
            batch = {"data": list(), "index_expr": list()}
            for i in range(params.batch_size):
                if len(index_expressions) <= 0:
                    break

                # Extend the index expression to include the image / channel axis
                index_expr = index_expressions.pop(0)
                ext_index_expr = (slice(None), *index_expr)

                # Add the data to the batch
                batch["data"].append(input_array[ext_index_expr])
                batch["index_expr"].append(index_expr)

            # Apply the model to the batch
            processed_batch = self._infer_on_batch(batch, params)

            # Insert the output batch into the output array
            for i in range(len(processed_batch["data"])):
                index_expr = processed_batch["index_expr"][i]
                output_data = processed_batch["data"][i]
                output_array[index_expr] = output_data

        return output_array

    @staticmethod
    def _array_to_subject(output_array: np.ndarray, subject: Subject, params: InferenceFilterParams) -> Subject:
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
        # Get the reference image
        reference_image = subject.get_image_by_modality(params.reference_modality)

        if reference_image is None:
            raise ValueError("The reference image is not available.")

        reference_image_sitk = reference_image.get_image_data()

        # Separate the output array according to the provided organ indices
        # and create images from the arrays
        for idx in range(len(params.organ_indices)):
            organ_array = np.zeros_like(output_array, dtype=np.uint8)
            organ_array[output_array == params.organ_indices[idx]] = 1

            # Create an image from the array
            image = sitk.GetImageFromArray(organ_array)
            image.CopyInformation(reference_image_sitk)

            # Create a segmentation image from the image
            segmentation_image = SegmentationImage(image, params.output_organs[idx], params.output_annotator)

            # Copy the transform tape from the reference image such that the segmentation image
            # can be transformed in the same way as the reference image
            segmentation_image.transform_tape = deepcopy(reference_image.transform_tape)

            # Add the segmentation image to the subject
            subject.add_image(segmentation_image, force=True)

        return subject

    def execute(self, subject: Subject, params: InferenceFilterParams) -> Subject:
        """Execute the filter on the provided :class:`~pyradise.data.subject.Subject` instance.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance.
            params (InferenceFilterParams): The filter parameters.

        Returns:
            Subject: The :class:`~pyradise.data.subject.Subject` instance with the newly added
            :class:`~pyradise.data.image.SegmentationImage` instances.
        """

        # get the input array for the DL-model
        input_array = self._get_input_array(subject, params)

        # prepare the model for inference (e.g. load the model parameters)
        self._prepare_model(params.model, params.model_path)

        # apply the model
        output_array = self._apply_model(input_array, params)

        # construct the output image
        subject = self._array_to_subject(output_array, subject, params)

        return subject

    def execute_inverse(
        self,
        subject: Subject,
        transform_info: TransformInfo,
        target_image: Optional[Union[SegmentationImage, IntensityImage]] = None,
    ) -> Subject:
        """Return the provided :class:`~pyradise.data.subject.Subject` instance without any processing because
        the inference of a DL-model is typically not invertible.

        Args:
            subject (Subject): The :class:`~pyradise.data.subject.Subject` instance to be returned.
            transform_info (TransformInfo): The transform information.
            target_image (Optional[Union[SegmentationImage, IntensityImage]]): The target image to which the inverse
             transformation should be applied. If None, the inverse transformation is applied to all images (default:
             None).

        Returns:
            Subject: The provided :class:`~pyradise.data.subject.Subject` instance.
        """

        # potentially warn the user that the operation is not invertible
        if self.warn_on_non_invertible:
            warnings.warn(
                f"The {self.__class__.__name__} filter is called but is not invertible. "
                "The provided subject is returned without modification."
            )

        return subject
