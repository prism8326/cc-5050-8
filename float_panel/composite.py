"""Stitch overlapping canvases using world coordinates and coverage masks.

The receiver never trusts pixel order: every sample is placed at its world
coordinate from the data window. Missing samples (coverage 0) never overwrite
real ones, which is what makes real black pixels and absent samples distinct.
"""

from __future__ import annotations

import numpy as np

from .model import UINT32, FloatImage


def union_display_window(images: list[FloatImage]) -> tuple[int, int, int, int]:
    x0 = min(img.display_window[0] for img in images)
    y0 = min(img.display_window[1] for img in images)
    x1 = max(img.display_window[2] for img in images)
    y1 = max(img.display_window[3] for img in images)
    return x0, y0, x1, y1


def union_data_window(images: list[FloatImage]) -> tuple[int, int, int, int]:
    x0 = min(img.data_window[0] for img in images)
    y0 = min(img.data_window[1] for img in images)
    x1 = max(img.data_window[2] for img in images)
    y1 = max(img.data_window[3] for img in images)
    return x0, y0, x1, y1


def stitch(images: list[FloatImage]) -> tuple[FloatImage, np.ndarray]:
    """Place images onto a shared rectangle covering every data window.

    Returns the composited image and an owner map (uint8, 0 = no owner,
    otherwise 1-based index of the image that supplied each pixel). Overlaps
    where two real samples disagree are counted and the first image wins, so
    the caller can verify overlap consistency.
    """
    x0, y0, x1, y1 = union_data_window(images)
    width, height = x1 - x0 + 1, y1 - y0 + 1

    channel_names = images[0].channel_names()
    for img in images[1:]:
        if set(img.channel_names()) != set(channel_names):
            raise ValueError("stitch inputs must expose the same channel set")

    accum: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    for name in channel_names:
        ch0 = images[0].channels[name]
        dtypes[name] = ch0.dtype
        if ch0.dtype == UINT32:
            accum[name] = np.zeros((height, width), dtype=np.uint32)
        else:
            accum[name] = np.zeros((height, width), dtype=np.float32)

    coverage = np.zeros((height, width), dtype=bool)
    owner = np.zeros((height, width), dtype=np.uint8)
    overlap_conflicts = 0
    overlap_agreements = 0

    for idx, img in enumerate(images, start=1):
        ox, oy = img.x_origin - x0, img.y_origin - y0
        h, w = img.height, img.width
        region = (slice(oy, oy + h), slice(ox, ox + w))
        for name in channel_names:
            ch = img.channels[name]
            arr = ch.data if ch.dtype == UINT32 else ch.as_float32()
            dst = accum[name][region]
            if ch.dtype == UINT32:
                take = img.coverage & ~coverage[region]
                dst[take] = arr[take]
            else:
                valid = img.coverage
                already = coverage[region]
                both = valid & already
                if both.any():
                    a, b = dst[both], arr[both]
                    eq = np.where(np.isnan(a) & np.isnan(b), True, a == b)
                    overlap_conflicts += int((~eq).sum())
                    overlap_agreements += int(eq.sum())
                take = valid & ~already
                dst[take] = arr[take]
        coverage[region] |= img.coverage
        unowned = img.coverage & (owner[region] == 0)
        owner[region] = np.where(unowned, idx, owner[region])

    channels = {
        name: type(images[0].channels[name])(
            name=name,
            dtype=dtypes[name],
            data=accum[name].astype(
                np.uint32 if dtypes[name] == UINT32
                else np.float16 if dtypes[name] == "FLOAT16"
                else np.float32
            ),
        )
        for name in channel_names
    }
    composite = FloatImage(
        name="stitch",
        display_window=union_display_window(images),
        data_window=(x0, y0, x1, y1),
        channels=channels,
        coverage=coverage,
    )
    composite.meta = {  # type: ignore[attr-defined]
        "overlap_agreements": overlap_agreements,
        "overlap_conflicts": overlap_conflicts,
    }
    return composite, owner


def coverage_signature(image: FloatImage) -> bytes:
    """Hashable coverage + world-window signature for equivalence checks."""
    return (
        np.ascontiguousarray(image.coverage).tobytes()
        + np.array(image.data_window, dtype=np.int64).tobytes()
    )
