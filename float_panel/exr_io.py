"""OpenEXR adapter: read display/data windows and named channels, write back.

Two read paths:

* :func:`read_exr` reads and crops in one call.
* :class:`DeferredExrRead` separates *header/raw read* from *window crop*.  It
  deliberately stops before cropping (``pause_before_crop``) and can be
  completed later (``finish``).  Every stage pins a window token, so a consumer
  cannot accidentally combine samples from a newly-read window with coordinates
  taken from an older one; :func:`assert_window_coherent` enforces it.
"""

from __future__ import annotations

import os

import Imath
import numpy as np
import OpenEXR

from .model import (
    FLOAT16, FLOAT32, UINT32, Channel, ChannelDescriptor, FloatImage,
    new_window_token,
)

_PIXEL_TYPE = {
    Imath.PixelType.UINT: UINT32,
    Imath.PixelType.HALF: FLOAT16,
    Imath.PixelType.FLOAT: FLOAT32,
}
_NP_DTYPE = {UINT32: np.uint32, FLOAT16: np.float16, FLOAT32: np.float32}
_IMATH_TYPE = {UINT32: Imath.PixelType.UINT, FLOAT16: Imath.PixelType.HALF, FLOAT32: Imath.PixelType.FLOAT}
_COVERAGE_CANDIDATES = ("coverage", "mask")


class WindowMismatchError(RuntimeError):
    """Samples and coordinates came from different window snapshots."""


def _box_tuple(box) -> tuple[int, int, int, int]:
    return (int(box.min.x), int(box.min.y), int(box.max.x), int(box.max.y))


def assert_window_coherent(*sources) -> str:
    """Raise unless every source pins the same window token; return that token.

    A source is a :class:`FloatImage` (its ``window_token``) or a bare token
    string that another stage captured earlier.  Passing a stale token together
    with a freshly-read image raises :class:`WindowMismatchError`, which is what
    stops "new-window colours, old-window coordinates" mixing.
    """
    tokens = set()
    for src in sources:
        if isinstance(src, str):
            tokens.add(src)
        elif hasattr(src, "window_token"):
            tokens.add(src.window_token)
        else:
            raise TypeError(f"cannot read a window token from {type(src)!r}")
    if len(tokens) != 1:
        raise WindowMismatchError(f"window token mismatch across sources: {sorted(tokens)}")
    return tokens.pop()


class DeferredExrRead:
    """Read EXR headers now; defer both the payload read and the crop.

    Stage 1 (constructor) reads only the header: display window, data window and
    channel list.  ``pause_before_crop`` opens the file just enough to snapshot
    that window state and stops *before* any channel payload is read or cropped.
    ``finish`` then reads the pixels and crops them with the window captured in
    stage 1 -- so a file (or another reader) that changes windows in between
    cannot silently supply samples for coordinates from a different window.
    """

    def __init__(self, path: str):
        self.path = path
        inp = OpenEXR.InputFile(path)
        header = inp.header()
        inp.close()
        self.display_window = _box_tuple(header["displayWindow"])
        self.data_window = _box_tuple(header["dataWindow"])
        self.window_token = new_window_token(self.display_window, self.data_window)
        self.channel_names = list(header["channels"])
        self._channels_header = header["channels"]
        self._paused = False
        self._finished = False

    # -- phase 1 ---------------------------------------------------------
    def pause_before_crop(self) -> dict:
        """Stop with only the header read; no payload has been touched yet."""
        self._paused = True
        return {
            "path": self.path,
            "window_token": self.window_token,
            "display_window": list(self.display_window),
            "data_window": list(self.data_window),
            "raw_channels": sorted(self.channel_names),
            "cropped": False,
            "payload_read": False,
        }

    # -- phase 2 ---------------------------------------------------------
    def finish(self) -> FloatImage:
        """Read payloads now, then crop with the stage-1 window."""
        if not self._paused:
            raise RuntimeError("call pause_before_crop() before finish()")
        if self._finished:
            raise RuntimeError("deferred read already finished")

        x0, y0, x1, y1 = self.data_window
        height = y1 - y0 + 1
        width = x1 - x0 + 1
        channels: dict[str, Channel] = {}
        coverage = None

        inp = OpenEXR.InputFile(self.path)
        for name in self.channel_names:
            dtype = _PIXEL_TYPE[self._channels_header[name].type.v]
            raw = inp.channel(name)
            arr = np.frombuffer(raw, dtype=_NP_DTYPE[dtype]).reshape(height, width).copy()
            if name.lower() in _COVERAGE_CANDIDATES:
                coverage = arr > 0
                continue
            channels[name] = Channel(
                name=name,
                dtype=dtype,
                data=arr,
                descriptor=ChannelDescriptor(
                    name=name,
                    source_name=name,
                    sample_step=(1, 1),
                    window_origin=(x0, y0),
                ),
            )
        inp.close()
        if coverage is None:
            coverage = np.ones((height, width), dtype=bool)
        self._finished = True
        return FloatImage(
            name=os.path.basename(self.path),
            display_window=self.display_window,
            data_window=self.data_window,
            channels=channels,
            coverage=coverage,
            source_path=self.path,
            window_token=self.window_token,
            lineage=(self.window_token,),
        )


def read_exr(path: str) -> FloatImage:
    reader = DeferredExrRead(path)
    reader.pause_before_crop()
    return reader.finish()


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
