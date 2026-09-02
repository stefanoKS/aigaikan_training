"""Tests for direct-quality comparison eligibility."""

from app.core.run_comparison import compare_training_runs
from app.models.training_run import TrainingRun


def _run(
    name: str,
    *,
    dataset_manifest: str,
    calibration_manifest: str,
    final_test_manifest: str,
    inspection_region_hash: str = "r" * 64,
    preprocessing_hash: str = "p" * 64,
) -> TrainingRun:
    return TrainingRun(
        run_name=name,
        run_dir=name,
        model_name="PatchCore",
        device="cpu",
        dataset_manifest_sha256=dataset_manifest,
        calibration_manifest_sha256=calibration_manifest,
        final_test_manifest_sha256=final_test_manifest,
        inspection_region_hash=inspection_region_hash,
        preprocessing_hash=preprocessing_hash,
        metrics={"NG Detection Rate": 0.9},
    )


def test_direct_quality_comparison_requires_identical_complete_split_manifests() -> None:
    first = _run("first", dataset_manifest="a" * 64, calibration_manifest="b" * 64, final_test_manifest="c" * 64)
    second = _run("second", dataset_manifest="a" * 64, calibration_manifest="b" * 64, final_test_manifest="c" * 64)

    report = compare_training_runs((first, second))

    assert report.direct_quality_comparison_allowed
    assert report.metric_rows["NG Detection Rate"] == (0.9, 0.9)


def test_different_final_test_manifest_is_not_a_direct_quality_comparison() -> None:
    first = _run("first", dataset_manifest="a" * 64, calibration_manifest="b" * 64, final_test_manifest="c" * 64)
    second = _run("second", dataset_manifest="a" * 64, calibration_manifest="b" * 64, final_test_manifest="d" * 64)

    report = compare_training_runs((first, second))

    assert not report.direct_quality_comparison_allowed
    assert "NOT ALLOWED" in report.reason


def test_different_calibration_manifest_is_not_a_direct_quality_comparison() -> None:
    first = _run("first", dataset_manifest="a" * 64, calibration_manifest="b" * 64, final_test_manifest="c" * 64)
    second = _run("second", dataset_manifest="a" * 64, calibration_manifest="d" * 64, final_test_manifest="c" * 64)

    report = compare_training_runs((first, second))

    assert not report.direct_quality_comparison_allowed
    assert "NOT ALLOWED" in report.reason


def test_different_inspection_roi_hash_is_not_a_direct_quality_comparison() -> None:
    first = _run("first", dataset_manifest="a" * 64, calibration_manifest="b" * 64, final_test_manifest="c" * 64)
    second = _run(
        "second",
        dataset_manifest="a" * 64,
        calibration_manifest="b" * 64,
        final_test_manifest="c" * 64,
        inspection_region_hash="s" * 64,
    )

    report = compare_training_runs((first, second))

    assert not report.direct_quality_comparison_allowed
    assert "inspection ROI" in report.reason


def test_different_preprocessing_policy_is_not_a_direct_quality_comparison() -> None:
    first = _run("first", dataset_manifest="a" * 64, calibration_manifest="b" * 64, final_test_manifest="c" * 64)
    second = _run(
        "second",
        dataset_manifest="a" * 64,
        calibration_manifest="b" * 64,
        final_test_manifest="c" * 64,
        preprocessing_hash="q" * 64,
    )

    report = compare_training_runs((first, second))

    assert not report.direct_quality_comparison_allowed
    assert "preprocessing policy" in report.reason