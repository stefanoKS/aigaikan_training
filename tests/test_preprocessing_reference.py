"""Tests for the standalone-reference preprocessing verification surface."""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

from app.core.preprocessing_reference import prepare_model_inputs, prepare_torch_model_inputs, preprocess_rgb, verify_golden_vectors
from app.models.image_preprocessing import ColorMode, ImagePreprocessingConfig
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import PreprocessingConfig


def test_checked_in_preprocessing_golden_vectors_match_the_shared_runner() -> None:
    vectors = Path(__file__).parents[1] / "app" / "resources" / "preprocessing_golden_vectors.json"

    verify_golden_vectors(vectors)


def test_reference_runner_matches_the_core_profile_output() -> None:
    source = np.array([[[20, 40, 60]]], dtype=np.uint8)
    profile = ImagePreprocessingConfig(profile_id="gray-v1", color_mode=ColorMode.GRAYSCALE_REPLICATED_RGB)

    output = preprocess_rgb(source, profile)

    assert output.shape == (1, 1, 3)
    assert output[0, 0, 0] == output[0, 0, 1] == output[0, 0, 2]


def test_torch_reference_preprocessing_matches_model_ready_trainer_pixels(tmp_path: Path) -> None:
    profile = ImagePreprocessingConfig(profile_id="gray-v1", color_mode=ColorMode.GRAYSCALE_REPLICATED_RGB)
    plan = PreprocessingConfig(image_preprocessing=profile).resolve("patchcore", (3, 2))
    source = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.uint8)

    numpy_inputs = prepare_model_inputs(source, InspectionRegionConfig(), plan)
    torch_inputs = prepare_torch_model_inputs(source, InspectionRegionConfig(), plan)

    assert len(torch_inputs) == len(numpy_inputs) == 1
    assert torch_inputs[0].dtype is torch.uint8
    assert np.array_equal(torch_inputs[0].numpy(), np.moveaxis(numpy_inputs[0], -1, 0))


def test_reference_runner_cli_reproduces_saved_full_pipeline_input(tmp_path: Path) -> None:
    profile = ImagePreprocessingConfig(profile_id="gray-v1", color_mode=ColorMode.GRAYSCALE_REPLICATED_RGB)
    roi = InspectionRegionConfig()
    plan = PreprocessingConfig(image_preprocessing=profile).resolve("patchcore", (3, 2))
    source = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.uint8)
    source_path = tmp_path / "source.png"
    roi_path = tmp_path / "inspection_region.json"
    plan_path = tmp_path / "preprocessing_plan.json"
    output_directory = tmp_path / "prepared"
    Image.fromarray(source, "RGB").save(source_path)
    roi_path.write_text(json.dumps(roi.to_dict()), encoding="utf-8")
    plan_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
    runner = Path(__file__).parents[1] / "scripts" / "preprocessing_reference_runner.py"

    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--input",
            str(source_path),
            "--output",
            str(output_directory),
            "--inspection-region",
            str(roi_path),
            "--resolved-plan",
            str(plan_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert np.array_equal(np.asarray(Image.open(output_directory / "tile-000.png").convert("RGB")), prepare_model_inputs(source, roi, plan)[0])