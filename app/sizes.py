"""Validation of the optional "known real dimension" used to scale a scan to millimetres."""
from __future__ import annotations

from .pipeline.export import DIMENSIONS


def parse_size(raw, dimension: str | None = None) -> dict | None:
    """Accepts {"dimension": "height", "mm": 95}, or a bare number plus `dimension`.
    Returns None when no usable size is given; raises ValueError when it is invalid."""
    if raw in (None, "", {}):
        return None
    if isinstance(raw, dict):
        dimension, mm = raw.get("dimension", "longest"), raw.get("mm")
    else:
        mm = raw
    dimension = dimension or "longest"
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {', '.join(DIMENSIONS)}")
    try:
        mm = float(mm)
    except (TypeError, ValueError):
        raise ValueError("size must be a number of millimetres") from None
    if not 0 < mm < 100_000:
        raise ValueError("size must be between 0 and 100000 mm")
    return {"dimension": dimension, "mm": mm}
