"""Versioned fixed inspection-region configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

ROI_CONTRACT_VERSION = 1
POINT_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")
Point = tuple[int, int]


@dataclass(frozen=True, slots=True)
class InspectionRegionConfig:
    """Immutable, source-resolution-bound perspective quadrilateral contract."""

    enabled: bool = False
    source_width: int = 0
    source_height: int = 0
    points_px: tuple[Point, ...] = ()
    roi_contract_version: int = ROI_CONTRACT_VERSION
    region_type: str = "perspective_quad"
    interpolation: str = "linear"
    transform: str = "perspective"

    @property
    def is_configured(self) -> bool:
        """Return whether an enabled ROI has all required geometric inputs."""
        return self.enabled and self.source_width > 0 and self.source_height > 0 and len(self.points_px) == 4

    def normalized_points(self) -> tuple[tuple[float, float], ...]:
        """Return coordinates normalized against the authoritative source resolution."""
        if not self.points_px:
            return ()
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("ROI source resolution must be positive before points can be normalized.")
        return tuple((x / self.source_width, y / self.source_height) for x, y in self.points_px)

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete, versioned ROI deployment contract."""
        return {
            "roi_contract_version": self.roi_contract_version,
            "enabled": self.enabled,
            "type": self.region_type,
            "source_size": {"width": self.source_width, "height": self.source_height},
            "points_px": [list(point) for point in self.points_px],
            "points_normalized": [list(point) for point in self.normalized_points()],
            "point_order": list(POINT_ORDER),
            "rectified_size": {"width": self.rectified_size()[0], "height": self.rectified_size()[1]},
            "interpolation": self.interpolation,
            "transform": self.transform,
        }

    def rectified_size(self) -> tuple[int, int]:
        """Return the natural rectified size calculated from the quadrilateral edges."""
        if not self.is_configured:
            return 0, 0
        top_left, top_right, bottom_right, bottom_left = self.points_px
        width = max(_distance(top_left, top_right), _distance(bottom_left, bottom_right))
        height = max(_distance(top_left, bottom_left), _distance(top_right, bottom_right))
        return int(round(width)), int(round(height))

    def warnings(self) -> tuple[str, ...]:
        """Return non-blocking geometry concerns that deserve an explicit operator review."""
        if not self.is_configured:
            return ()
        width, height = self.rectified_size()
        if width <= 8 or height <= 8:
            return ("Inspection ROI is extremely thin; review the selected quadrilateral.",)
        aspect_ratio = width / height
        if aspect_ratio >= 10 or aspect_ratio <= 0.1:
            return ("Inspection ROI has an extreme aspect ratio; review the selected quadrilateral.",)
        return ()

    def validate(self) -> None:
        """Reject unsupported, unordered, or geometrically invalid enabled ROI metadata."""
        if self.roi_contract_version != ROI_CONTRACT_VERSION:
            raise ValueError(f"Unsupported ROI contract version: {self.roi_contract_version}")
        if self.region_type != "perspective_quad" or self.transform != "perspective":
            raise ValueError("Only the perspective_quad inspection ROI contract is supported.")
        if self.interpolation != "linear":
            raise ValueError("Only linear interpolation is supported by the ROI contract.")
        if not self.enabled:
            return
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("ROI source resolution must be positive.")
        if len(self.points_px) != 4:
            raise ValueError("An enabled inspection ROI requires exactly four corner points.")
        for point in self.points_px:
            if not (0 <= point[0] < self.source_width and 0 <= point[1] < self.source_height):
                raise ValueError("Inspection ROI points must be inside the source image.")
        if polygon_self_intersects(self.points_px):
            raise ValueError("Inspection ROI polygon must not self-intersect.")
        if not is_convex_quad(self.points_px):
            raise ValueError("Inspection ROI polygon must be convex.")
        if tuple(order_quad_points(self.points_px)) != self.points_px:
            raise ValueError("Inspection ROI points must be ordered top-left, top-right, bottom-right, bottom-left.")
        if polygon_area(self.points_px) <= 64:
            raise ValueError("Inspection ROI area must be greater than 64 square pixels.")
        width, height = self.rectified_size()
        if width < 2 or height < 2:
            raise ValueError("Inspection ROI rectified width and height must both be at least two pixels.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InspectionRegionConfig":
        """Deserialize metadata and verify redundant normalized coordinates when present."""
        source_size = payload.get("source_size", {})
        if not isinstance(source_size, dict):
            raise ValueError("ROI source_size must be an object.")
        raw_points = payload.get("points_px", [])
        if not isinstance(raw_points, list):
            raise ValueError("ROI points_px must be an array.")
        points = tuple(_point_from_value(point) for point in raw_points)
        config = cls(
            roi_contract_version=int(payload.get("roi_contract_version", ROI_CONTRACT_VERSION)),
            enabled=bool(payload.get("enabled", False)),
            region_type=str(payload.get("type", "perspective_quad")),
            source_width=int(source_size.get("width", 0)),
            source_height=int(source_size.get("height", 0)),
            points_px=points,
            interpolation=str(payload.get("interpolation", "linear")),
            transform=str(payload.get("transform", "perspective")),
        )
        config.validate()
        _validate_normalized_points(config, payload.get("points_normalized"))
        _validate_rectified_size(config, payload.get("rectified_size"))
        point_order = payload.get("point_order", list(POINT_ORDER))
        if point_order != list(POINT_ORDER):
            raise ValueError("ROI point_order does not match the supported perspective quadrilateral contract.")
        return config


def order_quad_points(points: Iterable[Point]) -> tuple[Point, Point, Point, Point]:
    """Canonicalize four convex vertices as top-left, top-right, bottom-right, bottom-left."""
    values = tuple((int(x), int(y)) for x, y in points)
    if len(values) != 4 or len(set(values)) != 4:
        raise ValueError("Exactly four distinct inspection ROI points are required.")
    top_left = min(values, key=lambda point: (point[0] + point[1], point[1], point[0]))
    bottom_right = max(values, key=lambda point: (point[0] + point[1], point[1], point[0]))
    remaining = tuple(point for point in values if point not in {top_left, bottom_right})
    top_right = max(remaining, key=lambda point: (point[0] - point[1], point[0], -point[1]))
    bottom_left = min(remaining, key=lambda point: (point[0] - point[1], point[0], -point[1]))
    result = top_left, top_right, bottom_right, bottom_left
    if polygon_self_intersects(result) or not is_convex_quad(result):
        raise ValueError("Inspection ROI points must form one convex quadrilateral.")
    return result  # type: ignore[return-value]


def polygon_area(points: Iterable[Point]) -> float:
    """Return the absolute shoelace area of the polygon in square pixels."""
    values = tuple(points)
    if len(values) < 3:
        return 0.0
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(values, values[1:] + values[:1]))) / 2


def polygon_self_intersects(points: Iterable[Point]) -> bool:
    """Return whether opposite quadrilateral edges cross away from their shared endpoints."""
    values = tuple(points)
    if len(values) != 4:
        return True
    return _segments_intersect(values[0], values[1], values[2], values[3]) or _segments_intersect(
        values[1], values[2], values[3], values[0]
    )


def is_convex_quad(points: Iterable[Point]) -> bool:
    """Return whether four ordered points form a strictly convex quadrilateral."""
    values = tuple(points)
    if len(values) != 4:
        return False
    crosses = [_cross(values[index], values[(index + 1) % 4], values[(index + 2) % 4]) for index in range(4)]
    return all(cross > 0 for cross in crosses) or all(cross < 0 for cross in crosses)


def _distance(first: Point, second: Point) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _cross(first: Point, second: Point, third: Point) -> int:
    return (second[0] - first[0]) * (third[1] - second[1]) - (second[1] - first[1]) * (third[0] - second[0])


def _orientation(first: Point, second: Point, third: Point) -> int:
    value = _cross(first, second, third)
    return (value > 0) - (value < 0)


def _segments_intersect(first: Point, second: Point, third: Point, fourth: Point) -> bool:
    first_orientation = _orientation(first, second, third)
    second_orientation = _orientation(first, second, fourth)
    third_orientation = _orientation(third, fourth, first)
    fourth_orientation = _orientation(third, fourth, second)
    return first_orientation != second_orientation and third_orientation != fourth_orientation


def _point_from_value(value: object) -> Point:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Each ROI point must contain exactly an x and y coordinate.")
    x, y = value
    if isinstance(x, bool) or isinstance(y, bool):
        raise ValueError("ROI point coordinates must be integer pixels.")
    if int(x) != x or int(y) != y:
        raise ValueError("ROI point coordinates must be integer pixels.")
    return int(x), int(y)


def _validate_normalized_points(config: InspectionRegionConfig, value: object) -> None:
    if value is None:
        return
    if not isinstance(value, list) or len(value) != len(config.points_px):
        raise ValueError("ROI points_normalized does not match points_px.")
    for actual, expected in zip(value, config.normalized_points(), strict=True):
        if not isinstance(actual, (list, tuple)) or len(actual) != 2:
            raise ValueError("ROI points_normalized must contain x and y values.")
        if abs(float(actual[0]) - expected[0]) > 1e-9 or abs(float(actual[1]) - expected[1]) > 1e-9:
            raise ValueError("ROI points_normalized does not match authoritative points_px.")


def _validate_rectified_size(config: InspectionRegionConfig, value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("ROI rectified_size must be an object.")
    expected_width, expected_height = config.rectified_size()
    if int(value.get("width", -1)) != expected_width or int(value.get("height", -1)) != expected_height:
        raise ValueError("ROI rectified_size does not match the authoritative polygon geometry.")