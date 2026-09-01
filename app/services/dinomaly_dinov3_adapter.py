"""Experimental, application-owned Dinomaly reconstruction adapter for DINOv3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Iterable


class DinoV3AdapterError(ValueError):
    """Raised when an encoder cannot satisfy the DINOv3 adapter contract."""


@dataclass(frozen=True, slots=True)
class DinoV3EncoderMetadata:
    """Runtime-derived DINOv3 encoder details persisted with an experimental run."""

    encoder_name: str
    patch_size: int
    embedding_dim: int
    depth: int
    num_heads: int
    feature_layers: tuple[int, ...]
    input_size: tuple[int, int]
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe runtime metadata."""
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, tuple):
                payload[key] = list(value)
        return payload


def inspect_dinov3_encoder(
    encoder: Any,
    encoder_name: str,
    requested_feature_layers: Iterable[int] = (),
) -> DinoV3EncoderMetadata:
    """Inspect a loaded DINOv3 encoder without inferring architecture from its name."""
    if "dinov3" not in encoder_name.casefold():
        raise DinoV3AdapterError("Dinomaly DINOv3 requires an explicitly named DINOv3 encoder.")

    patch_size = _square_patch_size(getattr(getattr(encoder, "patch_embed", None), "patch_size", None))
    embedding_dim = _positive_integer(getattr(encoder, "embed_dim", getattr(encoder, "num_features", None)), "embedding dimension")
    blocks = getattr(encoder, "blocks", None)
    try:
        depth = len(blocks)
        first_block = blocks[0]
    except (TypeError, IndexError, KeyError) as exc:
        raise DinoV3AdapterError("The DINOv3 encoder must expose a non-empty transformer block sequence.") from exc
    if depth <= 0:
        raise DinoV3AdapterError("The DINOv3 encoder must expose at least one transformer block.")
    num_heads = _positive_integer(getattr(getattr(first_block, "attn", None), "num_heads", None), "attention head count")
    if embedding_dim % num_heads:
        raise DinoV3AdapterError("DINOv3 embedding dimension must be divisible by its runtime attention head count.")
    if not callable(getattr(encoder, "forward_intermediates", None)):
        raise DinoV3AdapterError("The selected DINOv3 encoder does not support forward_intermediates.")

    feature_layers = tuple(int(layer) for layer in requested_feature_layers) or _default_feature_layers(depth)
    if len(set(feature_layers)) != len(feature_layers) or any(layer < 0 or layer >= depth for layer in feature_layers):
        raise DinoV3AdapterError(
            f"DINOv3 feature layers must be distinct indices between 0 and {depth - 1}; got {list(feature_layers)}."
        )

    input_size, mean, std = _preprocessing_metadata(encoder)
    return DinoV3EncoderMetadata(
        encoder_name=encoder_name,
        patch_size=patch_size,
        embedding_dim=embedding_dim,
        depth=depth,
        num_heads=num_heads,
        feature_layers=feature_layers,
        input_size=input_size,
        normalization_mean=mean,
        normalization_std=std,
    )


def verify_dinov3_feature_contract(
    encoder: Any,
    metadata: DinoV3EncoderMetadata,
    image_size: tuple[int, int],
) -> None:
    """Prove that the runtime feature API returns exactly one patch token per patch."""
    image_height, image_width = image_size
    if image_height % metadata.patch_size or image_width % metadata.patch_size:
        raise DinoV3AdapterError(
            f"Dinomaly DINOv3 input dimensions must be divisible by the runtime patch size {metadata.patch_size}."
        )
    try:
        import torch
    except Exception as exc:
        raise DinoV3AdapterError("PyTorch is required to validate the DINOv3 encoder contract.") from exc

    parameter = next(iter(encoder.parameters()), None)
    device = getattr(parameter, "device", torch.device("cpu"))
    dtype = getattr(parameter, "dtype", torch.float32)
    was_training = bool(getattr(encoder, "training", False))
    encoder.eval()
    try:
        with torch.inference_mode():
            features = encoder.forward_intermediates(
                torch.zeros((1, 3, image_height, image_width), device=device, dtype=dtype),
                indices=list(metadata.feature_layers),
                norm=False,
                output_fmt="NLC",
                intermediates_only=True,
            )
    except Exception as exc:
        raise DinoV3AdapterError(
            "DINOv3 feature extraction failed while validating the selected encoder's runtime API."
        ) from exc
    finally:
        encoder.train(was_training)

    try:
        features = list(features)
    except TypeError as exc:
        raise DinoV3AdapterError("DINOv3 forward_intermediates must return one token tensor per selected layer.") from exc
    expected_tokens = (image_height // metadata.patch_size) * (image_width // metadata.patch_size)
    if len(features) != len(metadata.feature_layers):
        raise DinoV3AdapterError("DINOv3 returned a different feature-layer count than requested.")
    for feature in features:
        shape = tuple(getattr(feature, "shape", ()))
        if len(shape) != 3 or shape[0] != 1 or shape[1] != expected_tokens or shape[2] != metadata.embedding_dim:
            raise DinoV3AdapterError(
                "DINOv3 must return NLC tensors containing only one token per image patch; "
                "the selected encoder exposes an incompatible token layout."
            )


def create_dinomaly_dinov3_model(config: Any) -> Any:
    """Build the experimental adapter only after the real encoder contract is verified."""
    try:
        import timm
        import torch
        import torch.nn.functional as functional
        from torch import nn
        from anomalib import LearningType
        from anomalib.data import InferenceBatch
        from anomalib.models.components import AnomalibModule
        from anomalib.pre_processing import PreProcessor
        from torchvision.transforms.v2 import Compose, Normalize, Resize
    except Exception as exc:
        raise DinoV3AdapterError("Dinomaly DINOv3 requires working timm, PyTorch, torchvision, and Anomalib imports.") from exc

    encoder_name = str(config.dinomaly_dinov3_encoder).strip()
    encoder = timm.create_model(encoder_name, pretrained=True, num_classes=0).eval()
    metadata = inspect_dinov3_encoder(encoder, encoder_name, config.dinov3_feature_layers)
    verify_dinov3_feature_contract(encoder, metadata, config.model_input_size)
    if config.dinomaly_context_recentering:
        raise DinoV3AdapterError(
            "Dinomaly DINOv3 context recentering is unavailable because its verified token contract contains no class token."
        )

    class ReconstructionModel(nn.Module):
        """Feature reconstruction model whose dimensions come from inspected encoder metadata."""

        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.metadata = metadata
            self.bottleneck = nn.Sequential(
                nn.LayerNorm(metadata.embedding_dim),
                nn.Linear(metadata.embedding_dim, metadata.embedding_dim * 4, bias=False),
                nn.GELU(),
                nn.Dropout(config.dinomaly_bottleneck_dropout),
                nn.Linear(metadata.embedding_dim * 4, metadata.embedding_dim, bias=False),
            )
            self.decoder = nn.ModuleList(
                nn.TransformerEncoderLayer(
                    d_model=metadata.embedding_dim,
                    nhead=metadata.num_heads,
                    dim_feedforward=metadata.embedding_dim * 4,
                    dropout=config.dinomaly_bottleneck_dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.dinomaly_decoder_depth)
            )

        def _features(self, batch: Any) -> Any:
            with torch.no_grad():
                features = self.encoder.forward_intermediates(
                    batch,
                    indices=list(metadata.feature_layers),
                    norm=False,
                    output_fmt="NLC",
                    intermediates_only=True,
                )
            return torch.stack(list(features), dim=0).mean(dim=0).float()

        def forward(self, batch: Any) -> Any:
            source = self._features(batch)
            reconstructed = self.bottleneck(source)
            for decoder_block in self.decoder:
                reconstructed = decoder_block(reconstructed)
            difference = 1 - functional.cosine_similarity(source, reconstructed, dim=-1)
            if self.training:
                return difference.mean()
            batch_size = difference.shape[0]
            patch_rows = batch.shape[2] // metadata.patch_size
            patch_columns = batch.shape[3] // metadata.patch_size
            anomaly_map = difference.reshape(batch_size, 1, patch_rows, patch_columns)
            anomaly_map = functional.interpolate(
                anomaly_map,
                size=(batch.shape[2], batch.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            top_count = max(ceil(anomaly_map[0].numel() * 0.01), 1)
            scores = anomaly_map.flatten(1).topk(top_count, dim=1).values.mean(dim=1)
            return InferenceBatch(pred_score=scores, anomaly_map=anomaly_map)

    class DinomalyDinoV3Adapter(AnomalibModule):
        """Anomalib module for the verified application-side DINOv3 reconstruction model."""

        def __init__(self) -> None:
            pre_processor = PreProcessor(
                transform=Compose(
                    [
                        Resize(config.model_input_size, antialias=True),
                        Normalize(mean=list(metadata.normalization_mean), std=list(metadata.normalization_std)),
                    ]
                )
            )
            super().__init__(pre_processor=pre_processor)
            self.model = ReconstructionModel()
            self.encoder_metadata = metadata.to_dict()
            self.model_variant = "dinomaly_dinov3"

        def training_step(self, batch: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
            del args, kwargs
            loss = self.model(batch.image)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            return {"loss": loss}

        def validation_step(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            predictions = self.model(batch.image)
            return batch.update(pred_score=predictions.pred_score, anomaly_map=predictions.anomaly_map)

        def configure_optimizers(self) -> Any:
            return torch.optim.AdamW(
                (parameter for parameter in self.model.parameters() if parameter.requires_grad),
                lr=2e-3,
                weight_decay=1e-4,
            )

        @property
        def learning_type(self) -> Any:
            return LearningType.ONE_CLASS

    return DinomalyDinoV3Adapter()


def _default_feature_layers(depth: int) -> tuple[int, ...]:
    """Select four evenly distributed runtime block indices without name-based tables."""
    count = min(depth, 4)
    if count == 1:
        return (0,)
    return tuple(round(index * (depth - 1) / (count - 1)) for index in range(count))


def _square_patch_size(value: Any) -> int:
    """Read a positive square patch size from the loaded encoder."""
    if isinstance(value, int):
        return _positive_integer(value, "patch size")
    try:
        height, width = value
    except (TypeError, ValueError) as exc:
        raise DinoV3AdapterError("The DINOv3 encoder must expose a square patch_embed.patch_size.") from exc
    if height != width:
        raise DinoV3AdapterError("Dinomaly DINOv3 requires a square encoder patch size.")
    return _positive_integer(height, "patch size")


def _positive_integer(value: Any, description: str) -> int:
    """Validate and normalize a positive runtime integer."""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise DinoV3AdapterError(f"The DINOv3 encoder must expose a positive {description}.") from exc
    if number <= 0:
        raise DinoV3AdapterError(f"The DINOv3 encoder must expose a positive {description}.")
    return number


def _preprocessing_metadata(
    encoder: Any,
) -> tuple[tuple[int, int], tuple[float, float, float], tuple[float, float, float]]:
    """Read exact input normalization metadata exposed by the loaded encoder."""
    configuration = getattr(encoder, "pretrained_cfg", None) or getattr(encoder, "default_cfg", None)
    input_size = _configuration_value(configuration, "input_size")
    mean = _configuration_value(configuration, "mean")
    std = _configuration_value(configuration, "std")
    try:
        channels, height, width = tuple(input_size)
        mean = tuple(float(value) for value in mean)
        std = tuple(float(value) for value in std)
    except (TypeError, ValueError) as exc:
        raise DinoV3AdapterError("The DINOv3 encoder must expose input_size, mean, and std preprocessing metadata.") from exc
    if channels != 3 or height <= 0 or width <= 0 or len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise DinoV3AdapterError("The DINOv3 encoder preprocessing metadata is invalid for RGB image inference.")
    return (int(height), int(width)), mean, std


def _configuration_value(configuration: Any, name: str) -> Any:
    """Read data configuration from either a timm mapping or configuration object."""
    if isinstance(configuration, dict):
        return configuration.get(name)
    return getattr(configuration, name, None)