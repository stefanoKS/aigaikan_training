"""Download required PatchCore backbone weights."""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    root = Path("weights")
    root.mkdir(exist_ok=True)
    try:
        from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
    except Exception as exc:
        raise SystemExit(f"torchvision is required to download weights: {exc}") from exc

    weights = Wide_ResNet50_2_Weights.DEFAULT
    model = wide_resnet50_2(weights=weights)
    state_dict = model.state_dict()
    destination = root / "wide_resnet50_2-default.pth"
    import torch

    torch.save(state_dict, destination)
    print(f"Saved PatchCore backbone weights to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
