"""Conformance test: PyRaDiSe vs RTMaskConformanceTest analytic ground truth.

Generates the RTMaskConformanceTest fixture (synthetic CT + RTSTRUCT +
analytic per-ROI NIfTI ground truth), drives PyRaDiSe through its
``SubjectDicomCrawler`` -> ``IterableSubjectLoader`` API to produce per-ROI
masks, and asserts each ROI passes the published conformance thresholds
(Dice, Surface DSC @ 1 mm, HD95, MSD, relative volume error).

The ground-truth NIfTIs are computed analytically (sub-voxel quadrature
against the closed-form shape definitions) -- independent of any
rasterizer -- so a Dice failure here is a real accuracy regression in
PyRaDiSe's segmentation extraction path, not a discretization artifact.

This module is opt-in: it imports the third-party ``rtmask_conformance``
package, installed via the ``conformance`` extra::

    pip install -e .[conformance]

If the package is not installed the entire module is skipped via
``pytest.importorskip``, so the default ``pytest`` run is unaffected.
``rtmask_conformance`` itself requires Python >= 3.10; on older
interpreters the extra cannot be installed and the test is skipped.

Threshold overrides go in ``tests/conformance.yaml`` (set
``RTMASK_CONFORMANCE_CONFIG`` to use a different file). See
https://github.com/brianmanderson/RTMaskConformanceTest for the schema.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

rtmask_conformance = pytest.importorskip(
    "rtmask_conformance",
    reason="install the `conformance` extra: pip install -e .[conformance]",
)

import SimpleITK as sitk  # noqa: E402

from rtmask_conformance import CONFORMANCE_ROIS, generate_fixture, load_config  # noqa: E402
from rtmask_conformance.generate import GenerateOptions  # noqa: E402
from rtmask_conformance.verify import Status, evaluate_one  # noqa: E402

# n_quadrature=2 (8 sub-voxel samples) makes the GT masks stable to ~1
# voxel of partial-volume disagreement on the boundary -- well below the
# pass thresholds and an order of magnitude faster than n=8.
_FIXTURE_QUADRATURE = 2

_CONFIG_YAML = Path(__file__).with_name("conformance.yaml")


def _stage_dicom_inputs(rtstruct: Path, image_folder: Path, stage: Path) -> None:
    """Hard-link (or copy) the CT slices + RTSTRUCT into a single directory.

    PyRaDiSe's ``SubjectDicomCrawler`` walks a directory tree and groups by
    study/series; it can't accept the CT folder and RTSTRUCT path as
    separate arguments. Mirrors the staging pattern from the upstream
    benchmark adapter at PythonCode/subenvs/pyradise/run_forward.py.
    """
    stage.mkdir(parents=True, exist_ok=True)
    for src in image_folder.glob("*.dcm"):
        dst = stage / src.name
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    rt_dst = stage / rtstruct.name
    try:
        os.link(rtstruct, rt_dst)
    except OSError:
        shutil.copy2(rtstruct, rt_dst)


def _extract_sitk_image(seg) -> "sitk.Image | None":
    """Pull the underlying SimpleITK image out of a PyRaDiSe segmentation
    container. Different point releases expose different attribute names;
    try the documented ones in order.
    """
    for attr in ("get_image_data", "get_image"):
        if hasattr(seg, attr):
            try:
                v = getattr(seg, attr)()
                if v is not None:
                    return v
            except Exception:
                continue
    return None


@pytest.fixture(scope="session")
def conformance_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic CT + RTSTRUCT + analytic GT NIfTIs."""
    out = tmp_path_factory.mktemp("rtmask_conformance_fixture")
    generate_fixture(out, options=GenerateOptions(n_quadrature=_FIXTURE_QUADRATURE))
    return out


@pytest.fixture(scope="session")
def pyradise_predictions(
    conformance_fixture: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Run PyRaDiSe against the fixture; emit one binary <roi>.nii.gz per ROI.

    The verifier expects ``<predictions>/<roi>.nii.gz`` per ROI, so we save
    each subject->organ segmentation under its ROI name from PyRaDiSe.
    """
    pred_dir = tmp_path_factory.mktemp("pyradise_preds")

    # Stage CT + RTSTRUCT in one directory for the crawler.
    stage = Path(tempfile.mkdtemp(prefix="pyradise_stage_"))
    try:
        _stage_dicom_inputs(
            rtstruct=conformance_fixture / "rtstruct" / "primitives_planar.dcm",
            image_folder=conformance_fixture / "refct",
            stage=stage,
        )

        # Lazy imports keep import-time errors out of pytest collection.
        from pyradise.fileio import (
            IterableSubjectLoader,
            SimpleModalityExtractor,
            SubjectDicomCrawler,
        )

        modality_extractor = SimpleModalityExtractor(modalities=("CT",))
        crawler = SubjectDicomCrawler(
            path=str(stage),
            modality_extractor=modality_extractor,
        )
        series_infos = crawler.execute()
        if not series_infos:
            pytest.fail("PyRaDiSe crawler discovered no series in the staged fixture.")

        # IterableSubjectLoader expects Tuple[Tuple[SeriesInfo, ...], ...]
        # (subjects -> series-within-subject). The crawler returns a flat
        # tuple for a single-subject input -- wrap it once more.
        if not isinstance(series_infos[0], tuple):
            series_infos = (tuple(series_infos),)

        loader = IterableSubjectLoader(info=series_infos)
        n_written = 0
        for subject in loader:
            organs = subject.get_organs() if hasattr(subject, "get_organs") else []
            for organ in organs:
                roi_name = organ.get_name() if hasattr(organ, "get_name") else str(organ)
                try:
                    seg = subject.get_image_by_organ(organ)
                except Exception:
                    continue
                if seg is None:
                    continue
                image = _extract_sitk_image(seg)
                if image is None:
                    continue
                sitk.WriteImage(image, str(pred_dir / f"{roi_name}.nii.gz"))
                n_written += 1

        if n_written == 0:
            pytest.fail("PyRaDiSe yielded no per-ROI masks for any subject.")
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    return pred_dir


@pytest.fixture(scope="session")
def conformance_config():
    """Resolution: env var > tests/conformance.yaml > package defaults."""
    config_path = os.environ.get("RTMASK_CONFORMANCE_CONFIG")
    if config_path is None and _CONFIG_YAML.is_file():
        config_path = str(_CONFIG_YAML)
    return load_config(config_path)


@pytest.mark.parametrize("roi", CONFORMANCE_ROIS)
def test_pyradise_conformance(
    roi: str,
    conformance_fixture: Path,
    pyradise_predictions: Path,
    conformance_config,
):
    """Each ROI: PyRaDiSe's mask must match analytic ground truth within
    the published thresholds (Dice, Surface DSC, HD95, MSD, volume error).
    """
    pred_path = pyradise_predictions / f"{roi}.nii.gz"
    gt_path = conformance_fixture / "groundtruth" / f"{roi}.nii.gz"
    assert gt_path.is_file(), f"fixture incomplete: {gt_path}"
    if not pred_path.is_file():
        pytest.fail(
            f"PyRaDiSe produced no mask for {roi!r}. "
            f"Predictions dir: {sorted(p.name for p in pyradise_predictions.iterdir())}"
        )

    result = evaluate_one(roi, pred_path, gt_path, conformance_config)

    if result.status == Status.GEOMETRY_MISMATCH:
        pytest.fail(
            f"{roi}: geometry mismatch between PyRaDiSe output and ground truth: "
            f"{result.geometry_diagnostic}"
        )
    if result.status != Status.PASS:
        pytest.fail(
            f"{roi}: {result.status.value}\n"
            f"  violations: {result.violations}\n"
            f"  metrics:    {result.metrics}\n"
            f"  thresholds: {result.thresholds}"
        )
