"""Run the trainer's canonical preprocessing implementation or verify fixed golden vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

runner_directory = Path(__file__).resolve().parent
package_root = runner_directory if (runner_directory / "app").is_dir() else runner_directory.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

import numpy as np
from PIL import Image

from app.core.preprocessing_reference import verify_golden_vectors
from app.core.image_preprocessor import ImagePreprocessor
from app.models.image_preprocessing import ImagePreprocessingConfig
from app.models.inspection_region import InspectionRegionConfig
from app.models.preprocessing_config import ResolvedPreprocessingPlan
from app.core.preprocessing_reference import prepare_model_inputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, help="Verify a fixed preprocessing golden-vector JSON file.")
    parser.add_argument("--profile", type=Path, help="Standalone preprocessing profile JSON.")
    parser.add_argument("--inspection-region", type=Path, help="Saved inspection_region.json for full raw-image processing.")
    parser.add_argument("--resolved-plan", type=Path, help="Saved preprocessing_plan.json for full raw-image processing.")
    parser.add_argument("--input", type=Path, help="Rectified RGB image input.")
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    args = parser.parse_args()
    if args.golden is not None:
        verify_golden_vectors(args.golden)
        return 0
    if args.input is None or args.output is None:
        parser.error("provide --golden, or both --input and --output")
    with Image.open(args.input) as image:
        source_rgb = np.asarray(image.convert("RGB"))
    if args.inspection_region is not None or args.resolved_plan is not None:
        if args.inspection_region is None or args.resolved_plan is None:
            parser.error("full raw-image processing requires both --inspection-region and --resolved-plan")
        inspection_region = InspectionRegionConfig.from_dict(json.loads(args.inspection_region.read_text(encoding="utf-8")))
        plan = ResolvedPreprocessingPlan.from_dict(json.loads(args.resolved_plan.read_text(encoding="utf-8")))
        args.output.mkdir(parents=True, exist_ok=True)
        for index, prepared in enumerate(prepare_model_inputs(source_rgb, inspection_region, plan)):
            Image.fromarray(prepared, "RGB").save(args.output / f"tile-{index:03d}.png")
        return 0
    if args.profile is None:
        parser.error("rectified-image processing requires --profile")
    profile = ImagePreprocessingConfig.from_dict(json.loads(args.profile.read_text(encoding="utf-8")))
    processed = ImagePreprocessor(profile).apply(source_rgb)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(processed, "RGB").save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())