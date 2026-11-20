"""OpenEXR adapter: read display/data windows and named channels, write back."""

from __future__ import annotations

import os

import Imath
import numpy as np
import OpenEXR

from .model import FLOAT16, FLOAT32, UINT32, Channel, FloatImage

_PIXEL_TYPE = {
    Imath.PixelType.UINT: UINT32,
    Imath.PixelType.HALF: FLOAT16,
    Imath.PixelType.FLOAT: FLOAT32,
}
_NP_DTYPE = {UINT32: np.uint32, FLOAT16: np.float16, FLOAT32: np.float32}
_IMATH_TYPE = {UINT32: Imath.PixelType.UINT, FLOAT16: Imath.PixelType.HALF, FLOAT32: Imath.PixelType.FLOAT}
_COVERAGE_CANDIDATES = ("coverage", "mask")


def _box_tuple(box) -> tuple[int, int, int, int]:
    return (int(box.min.x), int(box.min.y), int(box.max.x), int(box.max.y))


def read_exr(path: str) -> FloatImage:
    inp = OpenEXR.InputFile(path)
    header = inp.header()
    data_window = _box_tuple(header["dataWindow"])
    display_window = _box_tuple(header["displayWindow"])
    height = data_window[3] - data_window[1] + 1
    width = data_window[2] - data_window[0] + 1

    channels: dict[str, Channel] = {}
    coverage_name = None
    for name, ch in header["channels"].items():
        dtype = _PIXEL_TYPE[ch.type.v]
        raw = inp.channel(name)
        arr = np.frombuffer(raw, dtype=_NP_DTYPE[dtype]).reshape(height, width).copy()
        if name.lower() in _COVERAGE_CANDIDATES:
            coverage_name = name
            continue
        channels[name] = Channel(name=name, dtype=dtype, data=arr)

    if coverage_name is not None:
        raw = inp.channel(coverage_name)
        cov_arr = np.frombuffer(raw, dtype=np.uint32).reshape(height, width)
        coverage = cov_arr > 0
    else:
        # EXR always stores a value per pixel inside its data window; without an
        # explicit mask every stored pixel counts as a real sample.
        coverage = np.ones((height, width), dtype=bool)

    inp.close()
    return FloatImage(
        name=os.path.basename(path),
        display_window=display_window,
        data_window=data_window,
        channels=channels,
        coverage=coverage,
    )


def write_exr(path: str, image: FloatImage) -> None:
    x0, y0, x1, y1 = image.data_window
    width, height = image.width, image.height
    header = OpenEXR.Header(width, height)
    header["dataWindow"] = Imath.Box2i(Imath.V2i(x0, y0), Imath.V2i(x1, y1))
    header["displayWindow"] = Imath.Box2i(
        Imath.V2i(*image.display_window[:2]), Imath.V2i(*image.display_window[2:])
    )
    header["channels"] = {
        name: Imath.Channel(Imath.PixelType(_IMATH_TYPE[ch.dtype]))
        for name, ch in image.channels.items()
    }
    header["channels"]["coverage"] = Imath.Channel(Imath.PixelType(Imath.PixelType.UINT))

    out = OpenEXR.OutputFile(path, header)
    payload = {
        name: np.ascontiguousarray(
            ch.data.astype(_NP_DTYPE[ch.dtype], copy=False)
        ).tobytes()
        for name, ch in image.channels.items()
    }
    payload["coverage"] = np.ascontiguousarray(image.coverage.astype(np.uint32)).tobytes()
    out.writePixels(payload)
    out.close()
