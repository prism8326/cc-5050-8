"""Round-trip verification: world coordinates, coverage, quantization, non-finites."""

from __future__ import annotations

import numpy as np

from .fim_io import STATUS_INF, STATUS_NAN, STATUS_OVERFLOW, read_fim
from .model import UINT32, FloatImage


def compare_images(a: FloatImage, b: FloatImage, tolerance: float = 0.0) -> dict:
    """Pixel-level comparison restricted to the shared covered domain."""
    if a.data_window != b.data_window:
        raise AssertionError(f"data window mismatch: {a.data_window} != {b.data_window}")
    if a.display_window != b.display_window:
        raise AssertionError("display window mismatch")
    cov_match = np.array_equal(a.coverage, b.coverage)
    result: dict = {
        "data_window_equal": a.data_window == b.data_window,
        "display_window_equal": a.display_window == b.display_window,
        "coverage_equal": bool(cov_match),
        "channels": {},
    }
    if not cov_match:
        diff = np.argwhere(a.coverage != b.coverage)
        result["coverage_mismatch_count"] = int(len(diff))

    for name in a.channels:
        ca, cb = a.channels[name], b.channels[name]
        entry: dict = {"dtype_a": ca.dtype, "dtype_b": cb.dtype}
        if ca.dtype == UINT32 or cb.dtype == UINT32:
            entry["exact"] = bool(np.array_equal(ca.data, cb.data))
        else:
            fa, fb = ca.as_float32(), cb.as_float32()
            valid = a.coverage & np.isfinite(fa) & np.isfinite(fb)
            with np.errstate(invalid="ignore"):
                abs_err = np.abs(fa - fb)
            if valid.any():
                denom = np.maximum(np.abs(fa), 1e-30)
                entry.update(
                    max_abs_error=float(abs_err[valid].max()),
                    mean_abs_error=float(abs_err[valid].mean()),
                    max_rel_error=float((abs_err[valid] / denom[valid]).max()),
                    within_tolerance=bool(abs_err[valid].max() <= tolerance),
                )
            # Non-finite handling must preserve the *kind* of failure.
            entry["nan_preserved"] = bool(
                np.array_equal(np.isnan(fa), np.isnan(fb))
            )
            entry["inf_sign_preserved"] = bool(
                np.array_equal(np.isposinf(fa), np.isposinf(fb))
                and np.array_equal(np.isneginf(fa), np.isneginf(fb))
            )
        result["channels"][name] = entry
    return result


def verify_fim_roundtrip(fim_path: str, source: FloatImage) -> dict:
    """Reopen an exported FIM and check it against the in-memory source."""
    reopened, status, meta = read_fim(fim_path)
    report = {
        "file": fim_path,
        "representation": "full-frame" if meta["full_frame"] else "sparse-window",
        "world_origin": [reopened.x_origin, reopened.y_origin],
        "data_window": list(reopened.data_window),
        "display_window": list(reopened.display_window),
        "coverage_equal_to_source": bool(np.array_equal(reopened.coverage, source.coverage)),
        "covered_pixels": int(reopened.coverage.sum()),
        "missing_pixels": int((~reopened.coverage).sum()),
        "status_flags": {
            "overflow": int(np.count_nonzero(status & STATUS_OVERFLOW)),
            "nan": int(np.count_nonzero(status & STATUS_NAN)),
            "inf": int(np.count_nonzero(status & STATUS_INF)),
        },
    }
    report["pixel_comparison"] = compare_images(source, reopened)
    return report


def coverage_only_signature(image: FloatImage) -> dict:
    """What the receiver actually needs to place samples."""
    ys, xs = np.nonzero(image.coverage)
    return {
        "origin": [image.x_origin, image.y_origin],
        "covered_world_points": int(len(xs)),
        "first_points": [[int(x + image.x_origin), int(y + image.y_origin)] for y, x in zip(ys[:5], xs[:5])],
    }
