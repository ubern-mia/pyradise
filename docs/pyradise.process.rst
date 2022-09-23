.. module:: pyradise.process

Overview Process Package
========================

The :mod:`~pyradise.process` package provides functionality for pre-processing, DL-model inference, and post-processing
in combination with the provided data model. The main building block of this package is the
:class:`~pyradise.process.base.Filter` and its associated :class:`~pyradise.process.base.FilterParams` which process
:class:`~pyradise.data.subject.Subject` instances. Due to the standardized interface, the chaining of multiple filters
in a :class:`~pyradise.process.base.FilterPipeline` is feasible, improving clarity and reproducibility. Furthermore,
this package provides an invertibility mechanism for filters that implement invertible process steps. This feature
renders feasibility to restore the original physical orientation of the processed :class:`~pyradise.data.image.Image`,
which may be crucial when processing medical imaging data. However, subsequent data processing with multiple filters
limits the invertibility because the data experiences information loss.

This package provides a basic set of extensible filter implementations. Currently, the process package includes
exclusively filters often applied in auto-segmentation development. However, we want to encourage the community to
implement and share their filters (e.g., via pull requests to the PyRaDiSe GitHub repository). The recommended workflow
for implementing new filters is documented in the documentation of the :class:`~pyradise.process.base.Filter` class.







