"""Run verified in-memory Torch deployment reference inference on one RGB image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

script_directory = Path(__file__).resolve().parent
package_root = script_directory if (script_directory / "app").is_dir() else script_directory.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

import numpy as np
from PIL import Image

from app.core.deployment_reference import TorchDeploymentReferenceInferencer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    with Image.open(args.input) as image:
        source_rgb = np.asarray(image.convert("RGB"))
    reference = TorchDeploymentReferenceInferencer.load(args.package, device=args.device)
    result = reference.infer_rgb(source_rgb)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "score": result.score,
                "score_semantic": result.score_semantic,
                "score_source": result.score_source,
                "predicted_label": result.predicted_label,
                "timing": result.timing.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())