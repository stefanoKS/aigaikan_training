"""Versioned, aspect-preserving preprocessing contracts for inspection images."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

PREPROCESSING_CONTRACT_VERSION = 2
REFERENCE_RECTIFIED_SIZE = (639, 177)


class PaddingMode(StrEnum):
    """Supported deterministic padding policies."""

    CONSTANT = "constant"


class ScoreAggregation(StrEnum):
    """Application-owned aggregation of valid anomaly-map pixels."""

    MAX = "max"
    TOP_K_MEAN = "top_k_mean"


@dataclass(frozen=True, slots=True)
class TilingConfig:
    """Optional deterministic horizontal tile geometry in rectified pixels."""

    enabled: bool = False
    tile_width: int = 320
    tile_height: int = 177
    overlap_x: int = 160
    final_tile_alignment: str = "end"

    def validate(self) -> None:
        """Reject ambiguous or unsupported tile layouts."""
        if self.tile_width < 2 or self.tile_height < 2:
            raise ValueError("Tile dimensions must both be at least two pixels.")
        if not 0 <= self.overlap_x < self.tile_width:
            raise ValueError("Tile horizontal overlap must be non-negative and smaller than tile width.")
        if self.final_tile_alignment != "end":
            raise ValueError("Only end-aligned final horizontal tiles are supported.")

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete tile-layout policy."""
        return {
            "enabled": self.enabled,
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
            "overlap_x": self.overlap_x,
            "final_tile_alignment": self.final_tile_alignment,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "TilingConfig":
        """Deserialize a tile-layout policy without accepting unknown shapes."""
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("Preprocessing tiling must be an object.")
        result = cls(
            enabled=bool(payload.get("enabled", False)),
            tile_width=int(payload.get("tile_width", 320)),
            tile_height=int(payload.get("tile_height", 177)),
            overlap_x=int(payload.get("overlap_x", 160)),
            final_tile_alignment=str(payload.get("final_tile_alignment", "end")),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Project-level preprocessing policy resolved against one trained model and ROI."""

    preprocessing_contract_version: int = PREPROCESSING_CONTRACT_VERSION
    padding_mode: PaddingMode = PaddingMode.CONSTANT
    padding_value_rgb: tuple[int, int, int] = (0, 0, 0)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    score_aggregation: ScoreAggregation = ScoreAggregation.MAX
    top_k_fraction: float = 0.01
    aspect_ratio_tolerance: float = 0.005

    def validate(self) -> None:
        """Validate the project-owned behavior before it becomes a run contract."""
        if self.preprocessing_contract_version != PREPROCESSING_CONTRACT_VERSION:
            raise ValueError(f"Unsupported preprocessing contract version: {self.preprocessing_contract_version}")
        if self.padding_mode is not PaddingMode.CONSTANT:
            raise ValueError(f"Unsupported padding mode: {self.padding_mode}")
        if len(self.padding_value_rgb) != 3 or any(not 0 <= value <= 255 for value in self.padding_value_rgb):
            raise ValueError("Padding RGB values must be integers between zero and 255.")
        self.tiling.validate()
        if not 0 < self.top_k_fraction <= 1:
            raise ValueError("Top-k score fraction must be greater than zero and at most one.")
        if not 0 <= self.aspect_ratio_tolerance < 1:
            raise ValueError("Aspect-ratio tolerance must be non-negative and smaller than one.")

    def to_dict(self) -> dict[str, object]:
        """Serialize the stable project-level v2 preprocessing policy."""
        self.validate()
        return {
            "preprocessing_contract_version": self.preprocessing_contract_version,
            "padding_mode": self.padding_mode.value,
            "padding_value_rgb": list(self.padding_value_rgb),
            "tiling": self.tiling.to_dict(),
            "score_aggregation": self.score_aggregation.value,
            "top_k_fraction": self.top_k_fraction,
            "aspect_ratio_tolerance": self.aspect_ratio_tolerance,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "PreprocessingConfig":
        """Deserialize a v2 project policy; missing sidecars are handled by callers as legacy."""
        if not isinstance(payload, dict):
            raise ValueError("Preprocessing configuration must be a JSON object.")
        raw_padding = payload.get("padding_value_rgb", (0, 0, 0))
        if not isinstance(raw_padding, (list, tuple)) or len(raw_padding) != 3:
            raise ValueError("Padding RGB value must contain exactly three components.")
        result = cls(
            preprocessing_contract_version=int(
                payload.get("preprocessing_contract_version", PREPROCESSING_CONTRACT_VERSION)
            ),
            padding_mode=PaddingMode(payload.get("padding_mode", PaddingMode.CONSTANT.value)),
            padding_value_rgb=tuple(int(value) for value in raw_padding),
            tiling=TilingConfig.from_dict(payload.get("tiling")),
            score_aggregation=ScoreAggregation(payload.get("score_aggregation", ScoreAggregation.MAX.value)),
            top_k_fraction=float(payload.get("top_k_fraction", 0.01)),
            aspect_ratio_tolerance=float(payload.get("aspect_ratio_tolerance", 0.005)),
        )
        result.validate()
        return result

    def resolve(self, model_id: str, rectified_size: tuple[int, int]) -> "ResolvedPreprocessingPlan":
        """Resolve model input and valid-mask geometry for one fixed rectified ROI size."""
        self.validate()
        width, height = rectified_size
        if width < 2 or height < 2:
            raise ValueError("Rectified ROI dimensions must both be at least two pixels.")
        profile = _model_profile(model_id)
        if self.tiling.enabled:
            return _resolve_tiled_plan(self, profile, width, height)
        return _resolve_full_roi_plan(self, profile, width, height)


@dataclass(frozen=True, slots=True)
class PreprocessingTile:
    """One input sent to a model, with its source extent and valid output rectangle."""

    index: int
    rectified_box: tuple[int, int, int, int]
    padded_size: tuple[int, int]
    model_input_size: tuple[int, int]
    valid_box: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, object]:
        """Serialize exact geometry instead of relying on implicit model transforms."""
        return {
            "index": self.index,
            "rectified_box": list(self.rectified_box),
            "padded_size": list(self.padded_size),
            "model_input_size": list(self.model_input_size),
            "valid_pixel_mask": {
                "encoding": "rectangular_valid_region",
                "size": list(self.model_input_size),
                "box": list(self.valid_box),
            },
        }

    @classmethod
    def from_dict(cls, payload: object) -> "PreprocessingTile":
        """Deserialize one exact model-input tile geometry."""
        if not isinstance(payload, dict):
            raise ValueError("Preprocessing tile must be an object.")
        mask = payload.get("valid_pixel_mask")
        if not isinstance(mask, dict):
            raise ValueError("Preprocessing tile must contain a valid-pixel mask descriptor.")
        if mask.get("encoding") != "rectangular_valid_region":
            raise ValueError("Unsupported preprocessing valid-pixel mask encoding.")
        model_input_size = _int_tuple(payload.get("model_input_size"), 2, "model_input_size")
        if _int_tuple(mask.get("size"), 2, "valid_pixel_mask.size") != model_input_size:
            raise ValueError("Preprocessing valid-pixel mask size does not match model input size.")
        return cls(
            index=int(payload.get("index", -1)),
            rectified_box=_int_tuple(payload.get("rectified_box"), 4, "rectified_box"),
            padded_size=_int_tuple(payload.get("padded_size"), 2, "padded_size"),
            model_input_size=model_input_size,
            valid_box=_int_tuple(mask.get("box"), 4, "valid_pixel_mask.box"),
        )


@dataclass(frozen=True, slots=True)
class ResolvedPreprocessingPlan:
    """The model- and ROI-specific v2 preprocessing contract persisted with a run."""

    preprocessing_contract_version: int
    model_id: str
    patch_size: int
    rectified_size: tuple[int, int]
    padding_mode: PaddingMode
    padding_value_rgb: tuple[int, int, int]
    score_aggregation: ScoreAggregation
    top_k_fraction: float
    aspect_ratio_tolerance: float
    tiles: tuple[PreprocessingTile, ...]

    @property
    def tiled(self) -> bool:
        """Return whether one source image produces multiple model inputs."""
        return len(self.tiles) > 1

    @property
    def model_input_size(self) -> tuple[int, int]:
        """Return the uniform model input dimensions for this resolved plan."""
        input_sizes = {tile.model_input_size for tile in self.tiles}
        if len(input_sizes) != 1:
            raise ValueError("A preprocessing plan must resolve every tile to one model input size.")
        return next(iter(input_sizes))

    def validate(self) -> None:
        """Validate reconstructed run metadata before it controls image geometry."""
        if self.preprocessing_contract_version != PREPROCESSING_CONTRACT_VERSION:
            raise ValueError(f"Unsupported preprocessing contract version: {self.preprocessing_contract_version}")
        if self.patch_size < 1:
            raise ValueError("Patch size must be positive.")
        width, height = self.rectified_size
        if width < 2 or height < 2:
            raise ValueError("Rectified ROI dimensions must both be at least two pixels.")
        if not self.tiles:
            raise ValueError("A preprocessing plan must contain at least one tile.")
        expected_indices = tuple(range(len(self.tiles)))
        if tuple(tile.index for tile in self.tiles) != expected_indices:
            raise ValueError("Preprocessing tile indexes must be contiguous and start at zero.")
        input_size = self.model_input_size
        if input_size[0] % self.patch_size or input_size[1] % self.patch_size:
            raise ValueError("Model input dimensions must be divisible by the selected patch size.")
        for tile in self.tiles:
            x, y, tile_width, tile_height = tile.rectified_box
            valid_x, valid_y, valid_width, valid_height = tile.valid_box
            if tile_width < 1 or tile_height < 1 or x < 0 or y < 0 or x + tile_width > width or y + tile_height > height:
                raise ValueError("Preprocessing tile rectified bounds are invalid.")
            if tile.padded_size[0] < tile_width or tile.padded_size[1] < tile_height:
                raise ValueError("Preprocessing tile padding cannot remove valid source pixels.")
            if valid_x < 0 or valid_y < 0 or valid_width < 1 or valid_height < 1:
                raise ValueError("Preprocessing valid-pixel region is invalid.")
            if valid_x + valid_width > input_size[0] or valid_y + valid_height > input_size[1]:
                raise ValueError("Preprocessing valid-pixel region exceeds the model input.")

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete model-ready processing plan."""
        self.validate()
        return {
            "preprocessing_contract_version": self.preprocessing_contract_version,
            "model_id": self.model_id,
            "patch_size": self.patch_size,
            "rectified_size": list(self.rectified_size),
            "padding_mode": self.padding_mode.value,
            "padding_value_rgb": list(self.padding_value_rgb),
            "score_aggregation": self.score_aggregation.value,
            "top_k_fraction": self.top_k_fraction,
            "aspect_ratio_tolerance": self.aspect_ratio_tolerance,
            "tiles": [tile.to_dict() for tile in self.tiles],
        }

    @classmethod
    def from_dict(cls, payload: object) -> "ResolvedPreprocessingPlan":
        """Deserialize a complete run-level v2 processing plan."""
        if not isinstance(payload, dict):
            raise ValueError("Resolved preprocessing plan must be a JSON object.")
        raw_padding = payload.get("padding_value_rgb")
        raw_tiles = payload.get("tiles")
        if not isinstance(raw_padding, (list, tuple)) or not isinstance(raw_tiles, list):
            raise ValueError("Resolved preprocessing plan has invalid padding or tile data.")
        result = cls(
            preprocessing_contract_version=int(payload.get("preprocessing_contract_version", -1)),
            model_id=str(payload.get("model_id", "")),
            patch_size=int(payload.get("patch_size", 0)),
            rectified_size=_int_tuple(payload.get("rectified_size"), 2, "rectified_size"),
            padding_mode=PaddingMode(payload.get("padding_mode", PaddingMode.CONSTANT.value)),
            padding_value_rgb=_int_tuple(raw_padding, 3, "padding_value_rgb"),
            score_aggregation=ScoreAggregation(payload.get("score_aggregation", ScoreAggregation.MAX.value)),
            top_k_fraction=float(payload.get("top_k_fraction", 0.01)),
            aspect_ratio_tolerance=float(payload.get("aspect_ratio_tolerance", 0.005)),
            tiles=tuple(PreprocessingTile.from_dict(tile) for tile in raw_tiles),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class _ModelPreprocessingProfile:
    model_id: str
    patch_size: int
    minimum_full_size: tuple[int, int]
    tiled_model_input_size: tuple[int, int] | None


def _model_profile(model_id: str) -> _ModelPreprocessingProfile:
    normalized = "".join(character for character in model_id.casefold() if character.isalnum())
    if normalized == "dinomalydinov3":
        return _ModelPreprocessingProfile("dinomaly_dinov3", 16, (640, 192), (448, 256))
    if normalized in {"dinomalydinov2", "anomalydino"}:
        canonical = "anomaly_dino" if normalized == "anomalydino" else "dinomaly_dinov2"
        return _ModelPreprocessingProfile(canonical, 14, (644, 182), (448, 252))
    if normalized == "superadd":
        return _ModelPreprocessingProfile("super_add", 16, (640, 448), None)
    if normalized in {"efficientad", "supersimplenet"}:
        canonical = "efficient_ad" if normalized == "efficientad" else "supersimplenet"
        return _ModelPreprocessingProfile(canonical, 1, (640, 192), None)
    if normalized in {"patchcore", "padim"}:
        return _ModelPreprocessingProfile(normalized, 1, (640, 192), None)
    raise ValueError(f"No preprocessing v2 profile is registered for model: {model_id}")


def _resolve_full_roi_plan(
    config: PreprocessingConfig,
    profile: _ModelPreprocessingProfile,
    width: int,
    height: int,
) -> ResolvedPreprocessingPlan:
    padded_size = _padded_size(width, height, profile.minimum_full_size, profile.patch_size)
    tile = PreprocessingTile(
        index=0,
        rectified_box=(0, 0, width, height),
        padded_size=padded_size,
        model_input_size=padded_size,
        valid_box=(0, 0, width, height),
    )
    return _plan(config, profile, (width, height), (tile,))


def _resolve_tiled_plan(
    config: PreprocessingConfig,
    profile: _ModelPreprocessingProfile,
    width: int,
    height: int,
) -> ResolvedPreprocessingPlan:
    tiling = config.tiling
    if height != tiling.tile_height:
        raise ValueError(
            "Tiled preprocessing requires the configured rectified tile height "
            f"of {tiling.tile_height}, received {height}."
        )
    if width < tiling.tile_width:
        raise ValueError(f"Tiled preprocessing requires a rectified width of at least {tiling.tile_width} pixels.")
    if profile.tiled_model_input_size is None:
        raise ValueError(f"Tiled preprocessing is not registered for {profile.model_id}.")
    padded_size = _padded_size(tiling.tile_width, tiling.tile_height, (0, 0), profile.patch_size)
    output_size = profile.tiled_model_input_size
    valid_width = _resized_valid_extent(tiling.tile_width, padded_size[0], output_size[0])
    valid_height = _resized_valid_extent(tiling.tile_height, padded_size[1], output_size[1])
    starts = _tile_starts(width, tiling.tile_width, tiling.overlap_x)
    tiles = tuple(
        PreprocessingTile(
            index=index,
            rectified_box=(start, 0, tiling.tile_width, tiling.tile_height),
            padded_size=padded_size,
            model_input_size=output_size,
            valid_box=(0, 0, valid_width, valid_height),
        )
        for index, start in enumerate(starts)
    )
    return _plan(config, profile, (width, height), tiles)


def _plan(
    config: PreprocessingConfig,
    profile: _ModelPreprocessingProfile,
    rectified_size: tuple[int, int],
    tiles: tuple[PreprocessingTile, ...],
) -> ResolvedPreprocessingPlan:
    result = ResolvedPreprocessingPlan(
        preprocessing_contract_version=config.preprocessing_contract_version,
        model_id=profile.model_id,
        patch_size=profile.patch_size,
        rectified_size=rectified_size,
        padding_mode=config.padding_mode,
        padding_value_rgb=config.padding_value_rgb,
        score_aggregation=config.score_aggregation,
        top_k_fraction=config.top_k_fraction,
        aspect_ratio_tolerance=config.aspect_ratio_tolerance,
        tiles=tiles,
    )
    result.validate()
    return result


def _padded_size(width: int, height: int, minimum_size: tuple[int, int], patch_size: int) -> tuple[int, int]:
    minimum_width, minimum_height = minimum_size
    return _round_up(max(width, minimum_width), patch_size), _round_up(max(height, minimum_height), patch_size)


def _round_up(value: int, divisor: int) -> int:
    return ((value + divisor - 1) // divisor) * divisor


def _tile_starts(image_width: int, tile_width: int, overlap_x: int) -> tuple[int, ...]:
    stride = tile_width - overlap_x
    starts = list(range(0, image_width - tile_width + 1, stride))
    last_start = image_width - tile_width
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return tuple(starts)


def _resized_valid_extent(valid_extent: int, padded_extent: int, output_extent: int) -> int:
    return (valid_extent * output_extent + padded_extent - 1) // padded_extent


def _int_tuple(value: object, length: int, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"Preprocessing {name} must contain exactly {length} integer values.")
    if any(isinstance(item, bool) or int(item) != item for item in value):
        raise ValueError(f"Preprocessing {name} must contain integer values.")
    return tuple(int(item) for item in value)