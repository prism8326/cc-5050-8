"""Generate the two synthetic EXR plates used by the exchange panel.

Canvas A and B overlap in a shared world rectangle but carry different data
windows with negative origins. On purpose, inside the overlap the same world
positions include:
  * a real black pixel (coverage=1, RGB=0) on one plate and
  * a missing sample (coverage=0) on the other,
so the receiver must keep those concepts distinct.
Both plates also contain negative light, highlights above 1.0, front/back depth,
an unknown semantic channel, NaN/+Inf values and a half-range overflow.
"""

from __future__ import annotations

import os

import Imath
import numpy as np
import OpenEXR

# Shared display window (the "full frame").
DISPLAY = (-100, -80, 99, 79)  # 200 x 160

# Data windows overlap in world rect (-40,-20)-(-11,19) (30x40).
WINDOW_A = (-80, -60, -11, 19)   # 70 x 80, negative origin
WINDOW_B = (-40, -20, 39, 59)    # 80 x 80, negative origin
OVERLAP = (-40, -20, -11, 19)

BLACK_POS = (-25, -5)    # A: real black here ; B: missing here
MISSING_POS = (-20, 0)   # A: missing here      ; B: real black here


def _shape(window):
    x0, y0, x1, y1 = window
    return y1 - y0 + 1, x1 - x0 + 1


def _base_color(window, seed: int) -> np.ndarray:
    """Deterministic world-coordinate field: shared world points look identical
    on both plates, while each plate still carries its own seeded variation
    outside the overlap."""
    h, w = _shape(window)
    x0, y0 = window[0], window[1]
    xs = np.arange(x0, x0 + w)[None, :]
    ys = np.arange(y0, y0 + h)[:, None]
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[..., 0] = 0.5 + 0.5 * np.sin(xs * 0.15 + ys * 0.07)   # world-locked
    base[..., 1] = 0.5 + 0.5 * np.cos(ys * 0.12 - xs * 0.05)   # world-locked
    base[..., 2] = rng.random((h, w)).astype(np.float32)       # plate-local
    # Make the blue channel world-locked as well, so overlap real samples agree
    # on all three color channels (blue noise is blended out in the overlap).
    base[..., 2] = 0.5 + 0.5 * np.sin((xs + ys) * 0.09)
    return base


def _build_plate(window, seed, black_world, missing_world, special: dict,
                 signed_strip_rows: tuple[int, int] | None = None) -> dict:
    h, w = _shape(window)
    x0, y0 = window[0], window[1]
    xs = np.arange(x0, x0 + w)[None, :]
    ys = np.arange(y0, y0 + h)[:, None]
    rgb = _base_color(window, seed)

    # Negative light values and highlights above one.
    rgb[..., 0] -= 0.8                       # push red below zero in places
    rgb[50:60, 10:20, 1] += 3.0              # green highlight > 1, plate-local rows
    rgb[60:68, 5:15, 2] += 8.0               # strong highlight, plate-local rows

    # Depth: world-locked field spanning foreground (signed, behind camera)
    # to a far background. The signed strip sits in plate-local rows so it
    # never lands inside the shared overlap rectangle.
    depth = (xs.astype(np.float32) * 0.15 + ys.astype(np.float32) * 0.25) + 32.0
    if signed_strip_rows is not None:
        r0, r1 = signed_strip_rows
        depth[r0:r1, :] = -5.0               # behind-camera / signed depth

    # Unknown semantic channel: world-locked arbitrary floats, never tone-mapped.
    custom = (xs.astype(np.float32) * 0.7 + ys.astype(np.float32) * 1.3) * 0.01 + 50.0

    coverage = np.ones((h, w), dtype=np.uint32)

    def local(wx, wy):
        return wy - y0, wx - x0

    rb, cb = local(*black_world)
    rgb[rb, cb] = (0.0, 0.0, 0.0)           # genuine black, stays covered
    rm, cm = local(*missing_world)
    coverage[rm, cm] = 0                    # absent sample, not black

    for (wx, wy), value in special.items():
        r, c = local(wx, wy)
        if value == "nan":
            rgb[r, c] = (np.nan, 0.2, 0.3)
        elif value == "inf":
            rgb[r, c] = (0.4, np.inf, 0.1)
        elif value == "overflow":
            rgb[r, c] = (100000.0, 0.5, 0.5)  # exceeds half max 65504
        elif value == "neg":
            rgb[r, c] = (-2.5, -0.25, -7.0)

    return {
        "window": window,
        "R": np.ascontiguousarray(rgb[..., 0]),
        "G": np.ascontiguousarray(rgb[..., 1]),
        "B": np.ascontiguousarray(rgb[..., 2]),
        "Z": np.ascontiguousarray(depth),
        "customHeat": np.ascontiguousarray(custom),
        "coverage": coverage,
    }


def write_plates(out_dir: str) -> tuple[str, str]:
    # Special world positions chosen inside each plate's own window.
    special_a = {
        (-70, -50): "nan",
        (-50, 15): "inf",
        (-70, 10): "overflow",
        (-60, -40): "neg",
    }
    special_b = {
        (0, 0): "nan",
        (20, 30): "inf",
        (30, 50): "overflow",
        (-5, 40): "neg",
    }
    a = _build_plate(WINDOW_A, 1, BLACK_POS, MISSING_POS, special_a,
                     signed_strip_rows=(4, 10))
    b = _build_plate(WINDOW_B, 7, MISSING_POS, BLACK_POS, special_b,
                     signed_strip_rows=None)

    paths = []
    for plate, name in ((a, "plate_a.exr"), (b, "plate_b.exr")):
        h, w = _shape(plate["window"])
        x0, y0, x1, y1 = plate["window"]
        header = OpenEXR.Header(w, h)
        header["dataWindow"] = Imath.Box2i(Imath.V2i(x0, y0), Imath.V2i(x1, y1))
        header["displayWindow"] = Imath.Box2i(Imath.V2i(*DISPLAY[:2]), Imath.V2i(*DISPLAY[2:]))
        header["channels"] = {
            "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            "customHeat": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            "coverage": Imath.Channel(Imath.PixelType(Imath.PixelType.UINT)),
        }
        path = os.path.join(out_dir, name)
        out = OpenEXR.OutputFile(path, header)
        out.writePixels({k: v.tobytes() for k, v in plate.items() if k != "window"})
        out.close()
        paths.append(path)
    return paths[0], paths[1]
