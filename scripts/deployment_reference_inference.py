"""Run two-file Torch deployment reference inference without a training directory."""

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

from app.core.deployment_package import DeploymentPackage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    with Image.open(args.input) as image:
        raw_uint8 = np.asarray(image)
    deployment = DeploymentPackage.load(args.package, device=args.device)
    result = deployment.predict(raw_uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "decision_score": result.decision_score,
                "score_semantic": result.score_semantic,
                "threshold": result.threshold,
                "is_ng": result.is_ng,
                "predicted_label": "NG" if result.is_ng else "OK",
                "anomaly_map_shape": list(result.anomaly_map.shape),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())