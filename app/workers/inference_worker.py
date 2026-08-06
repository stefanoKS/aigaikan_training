"""Inference worker entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def emit(message: dict[str, object]) -> None:
    """Emit a JSON line."""
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    """Minimal inference worker stub."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    emit({"type": "log", "level": "info", "message": f"Inference requested for {Path(args.image)}"})
    emit({"type": "completed", "result_dir": ""})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
