"""Display mapping.

Color channels get an independent per-channel exposure/gamma mapping (no clipping
in the data itself, only for preview). Depth and unknown channels are rendered as
false-color with raw numeric readouts, preserving numeric meaning. Non-finite and
overflowing samples get explicit, distinguishable preview colors instead of being
silently clamped to black or white.
"""

from __future__ import annotations

import numpy as np

from .fim_io import STATUS_INF, STATUS_NAN, STATUS_OVERFLOW
from .model import Semantic

# Preview sRGB colors for exceptional samples.
COLOR_NAN = (255, 0, 255)      # magenta
COLOR_INF = (255, 80, 0)       # orange
COLOR_OVERFLOW = (255, 255, 0) # yellow
COLOR_MISSING = (0, 0, 0, 0)   # transparent -> checkerboard underneath
COLOR_BLACK_REAL = (0, 0, 0, 255)


def map_color_channel(arr_f32: np.ndarray, exposure: float, gamma: float) -> np.ndarray:
    """Exposure in stops, then gamma; NaN/Inf pass through for status coloring."""
    scaled = arr_f32 * (2.0 ** exposure)
    with np.errstate(invalid="ignore"):
        out = np.power(np.clip(scaled, 0.0, None), 1.0 / gamma)
    return np.clip(out, 0.0, 1.0)


def _false_color(values: np.ndarray, valid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Blue->red ramp over [lo, hi]; invalid/missing handled by caller."""
    t = np.zeros_like(values, dtype=np.float32)
    if hi > lo:
        t = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    t = t[..., None]
    rgb = np.concatenate([t, 1.0 - np.abs(2 * t - 1), 1.0 - t], axis=-1)
    return (rgb * 255).astype(np.uint8)


def render_rgba(
    channel_arrays: dict[str, np.ndarray],
    coverage: np.ndarray,
    status: np.ndarray | None,
    semantics: dict[str, Semantic],
    settings: dict,
) -> np.ndarray:
    """Render an RGBA uint8 preview of HxW, alpha 0 where samples are missing."""
    h, w = coverage.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    any_status = status if status is not None else np.zeros((h, w), np.uint8)

    color_names = [n for n, s in semantics.items() if s == Semantic.COLOR][:3]
    depth_names = [n for n, s in semantics.items() if s == Semantic.DEPTH]
    mode = settings.get("mode", "color")

    if mode == "color" and color_names:
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        for i, name in enumerate(color_names):
            exp = settings.get("exposure", {}).get(name, 0.0)
            gamma = settings.get("gamma", {}).get(name, 2.2)
            rgb[..., i] = map_color_channel(channel_arrays[name], exp, gamma)
        safe = np.where(np.isfinite(rgb), rgb, 0.0)
        rgba[..., :3] = (np.clip(safe, 0, 1) * 255).astype(np.uint8)
    else:
        # Depth or selected unknown channel: false color from finite range.
        target = mode if mode in channel_arrays else (depth_names[0] if depth_names else next(iter(channel_arrays)))
        values = channel_arrays[target].astype(np.float32)
        finite = np.isfinite(values) & coverage
        lo = float(values[finite].min()) if finite.any() else 0.0
        hi = float(values[finite].max()) if finite.any() else 1.0
        rgba[..., :3] = _false_color(values, finite, lo, hi)

    # Exceptional status overrides the mapped color so it is visible.
    nan_mask = (any_status & STATUS_NAN).astype(bool)
    inf_mask = (any_status & STATUS_INF).astype(bool)
    overflow_mask = ((any_status & STATUS_OVERFLOW)
                     & ~(any_status & (STATUS_NAN | STATUS_INF))).astype(bool)
    rgba[nan_mask, :3] = COLOR_NAN
    rgba[inf_mask, :3] = COLOR_INF
    rgba[overflow_mask, :3] = COLOR_OVERFLOW

    # Alpha encodes coverage: real black stays opaque, missing stays transparent.
    rgba[..., 3] = np.where(coverage, 255, 0).astype(np.uint8)
    return rgba


def sample_tooltip(image, world_x: int, world_y: int) -> dict:
    """Raw numeric values at a world coordinate (depth/unknown keep meaning)."""
    if not image.contains(world_x, world_y):
        return {"inside": False}
    row, col = image.pixel_index(world_x, world_y)
    covered = bool(image.coverage[row, col])
    values = {}
    for name, ch in image.channels.items():
        v = ch.data[row, col]
        values[name] = {
            "value": float(v) if np.isfinite(v) else ("nan" if np.isnan(v) else ("+inf" if v > 0 else "-inf")),
            "dtype": ch.dtype,
            "semantic": ch.semantic.value,
        }
    return {
        "inside": True,
        "world": [world_x, world_y],
        "covered": covered,
        "note": "real sample (value may be exactly zero)" if covered else "missing sample",
        "channels": values,
    }
