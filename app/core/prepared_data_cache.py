"""Immutable cache for deterministic model-ready image tiles."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from app.core.preprocessing_contract import resolved_preprocessing_hash
from app.core.preprocessing_pipeline import PreprocessingPipeline

PREPARED_DATA_CACHE_VERSION = 1


@dataclass(frozen=True, slots=True)
class PreparedDataCacheReport:
    """Per-run cache outcomes suitable for run provenance and operator logs."""

    hits: int
    misses: int
    rebuilt_entries: int

    def to_dict(self) -> dict[str, int]:
        return {
            "version": PREPARED_DATA_CACHE_VERSION,
            "hits": self.hits,
            "misses": self.misses,
            "rebuilt_entries": self.rebuilt_entries,
        }


class PreparedDataCache:
    """Cache deterministic RGB tile outputs by source bytes and resolved-plan identity."""

    def __init__(self, cache_directory: Path, preprocessing_pipeline: PreprocessingPipeline) -> None:
        self.cache_directory = cache_directory.expanduser().resolve()
        self.preprocessing_pipeline = preprocessing_pipeline
        self._plan_sha256 = resolved_preprocessing_hash(preprocessing_pipeline.plan)
        self._hits = 0
        self._misses = 0
        self._rebuilt_entries = 0

    def materialize(self, source_path: Path) -> tuple[Path, ...]:
        """Return verified immutable tiles, rebuilding only a missing or corrupt entry."""
        source_path = source_path.expanduser().resolve()
        source_sha256 = _sha256_file(source_path)
        entry_directory = self.cache_directory / self._entry_key(source_sha256)
        cached_tiles = self._verified_tiles(entry_directory, source_sha256)
        if cached_tiles is not None:
            self._hits += 1
            return cached_tiles
        was_corrupt = entry_directory.exists()
        if was_corrupt:
            if entry_directory.is_dir():
                shutil.rmtree(entry_directory)
            else:
                entry_directory.unlink()
            self._rebuilt_entries += 1
        self._misses += 1
        return self._build_entry(entry_directory, source_path, source_sha256)

    def report(self) -> PreparedDataCacheReport:
        """Return immutable cache outcomes for the current staging operation."""
        return PreparedDataCacheReport(self._hits, self._misses, self._rebuilt_entries)

    def clear(self) -> int:
        """Remove all cache entries owned by this cache version and return the number removed."""
        if not self.cache_directory.is_dir():
            return 0
        removed = 0
        for entry in self.cache_directory.iterdir():
            if entry.is_dir() and (entry / "manifest.json").is_file():
                shutil.rmtree(entry)
                removed += 1
        return removed

    def _entry_key(self, source_sha256: str) -> str:
        identity = f"{PREPARED_DATA_CACHE_VERSION}\0{self._plan_sha256}\0{source_sha256}"
        return hashlib.sha256(identity.encode("ascii")).hexdigest()

    def _verified_tiles(self, entry_directory: Path, source_sha256: str) -> tuple[Path, ...] | None:
        manifest_path = entry_directory / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                payload.get("version") != PREPARED_DATA_CACHE_VERSION
                or payload.get("source_sha256") != source_sha256
                or payload.get("preprocessing_plan_sha256") != self._plan_sha256
            ):
                return None
            tiles = payload.get("tiles")
            if not isinstance(tiles, list) or len(tiles) != len(self.preprocessing_pipeline.plan.tiles):
                return None
            paths: list[Path] = []
            for expected_tile, record in zip(self.preprocessing_pipeline.plan.tiles, tiles, strict=True):
                if not isinstance(record, dict) or record.get("index") != expected_tile.index:
                    return None
                expected_name = f"tile-{expected_tile.index:03d}.png"
                if record.get("file") != expected_name or not isinstance(record.get("sha256"), str):
                    return None
                tile_path = entry_directory / expected_name
                if not tile_path.is_file() or _sha256_file(tile_path) != record["sha256"]:
                    return None
                with Image.open(tile_path) as image:
                    if image.mode != "RGB" or image.size != expected_tile.model_input_size:
                        return None
                paths.append(tile_path)
            return tuple(paths)
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _build_entry(self, entry_directory: Path, source_path: Path, source_sha256: str) -> tuple[Path, ...]:
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".prepared-", dir=self.cache_directory) as temporary_directory:
            temporary_entry = Path(temporary_directory) / "entry"
            temporary_entry.mkdir()
            prepared_tiles = self.preprocessing_pipeline.prepare_path(source_path)
            if _sha256_file(source_path) != source_sha256:
                raise RuntimeError(f"Source image changed during preprocessing: {source_path}")
            records: list[dict[str, object]] = []
            for prepared in prepared_tiles:
                tile_path = temporary_entry / f"tile-{prepared.tile.index:03d}.png"
                Image.fromarray(prepared.image_rgb, "RGB").save(tile_path)
                records.append(
                    {
                        "index": prepared.tile.index,
                        "file": tile_path.name,
                        "sha256": _sha256_file(tile_path),
                    }
                )
            (temporary_entry / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": PREPARED_DATA_CACHE_VERSION,
                        "source_sha256": source_sha256,
                        "preprocessing_plan_sha256": self._plan_sha256,
                        "tiles": records,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if entry_directory.exists():
                shutil.rmtree(entry_directory)
            temporary_entry.replace(entry_directory)
        cached_tiles = self._verified_tiles(entry_directory, source_sha256)
        if cached_tiles is None:
            raise RuntimeError(f"Prepared-data cache entry could not be verified: {entry_directory}")
        return cached_tiles


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()