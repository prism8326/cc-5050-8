"""In-memory floating-point image model.

A :class:`FloatImage` keeps every channel as a numpy array shaped by the
*data window* (which may be sparse and may carry a negative origin), plus a
boolean coverage array distinguishing real samples from missing samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

FLOAT16 = "FLOAT16"
FLOAT32 = "FLOAT32"
UINT32 = "UINT32"

DTYPE_TO_CODE = {FLOAT16: 1, FLOAT32: 2, UINT32: 0}
CODE_TO_DTYPE = {v: k for k, v in DTYPE_TO_CODE.items()}
NUMPY_DTYPE = {FLOAT16: np.float16, FLOAT32: np.float32, UINT32: np.uint32}


class Semantic(Enum):
    COLOR = "color"
    ALPHA = "alpha"
    DEPTH = "depth"
    MASK = "mask"
    UNKNOWN = "unknown"


_DEPTH_NAMES = {"z", "depth", "zback", "backz", "depthback"}
_MASK_NAMES = {"coverage", "mask", "alpha.mask"}
_ALPHA_NAMES = {"a", "alpha"}


def classify_channel(name: str) -> Semantic:
    """Classify a channel name, including layer-prefixed names like ``diffuse.R``."""
    base = name.split(".")[-1].lower()
    full = name.lower()
    if full in _MASK_NAMES or base in {"coverage", "mask"}:
        return Semantic.MASK
    if base in _ALPHA_NAMES:
        return Semantic.ALPHA
    if base in _DEPTH_NAMES or full in _DEPTH_NAMES:
        return Semantic.DEPTH
    if base in {"r", "g", "b", "red", "green", "blue"}:
        return Semantic.COLOR
    return Semantic.UNKNOWN


@dataclass
class Channel:
    name: str
    dtype: str
    data: np.ndarray  # shape (height, width), pixel order of the data window

    @property
    def semantic(self) -> Semantic:
        return classify_channel(self.name)

    def as_float32(self) -> np.ndarray:
        if self.dtype == UINT32:
            return self.data.astype(np.float32)
        return self.data.astype(np.float32)


@dataclass
class FloatImage:
    name: str
    display_window: tuple[int, int, int, int]  # inclusive x0, y0, x1, y1
    data_window: tuple[int, int, int, int]
    channels: dict[str, Channel]
    coverage: np.ndarray  # bool, shape (height, width)

    @property
    def x_origin(self) -> int:
        return self.data_window[0]

    @property
    def y_origin(self) -> int:
        return self.data_window[1]

    @property
    def width(self) -> int:
        return self.data_window[2] - self.data_window[0] + 1

    @property
    def height(self) -> int:
        return self.data_window[3] - self.data_window[1] + 1

    @property
    def display_width(self) -> int:
        return self.display_window[2] - self.display_window[0] + 1

    @property
    def display_height(self) -> int:
        return self.display_window[3] - self.display_window[1] + 1

    def contains(self, x: int, y: int) -> bool:
        x0, y0, x1, y1 = self.data_window
        return x0 <= x <= x1 and y0 <= y <= y1

    def pixel_index(self, x: int, y: int) -> tuple[int, int]:
        """World coordinate -> local array (row, col)."""
        if not self.contains(x, y):
            raise KeyError(f"world coordinate ({x},{y}) outside data window {self.data_window}")
        return y - self.y_origin, x - self.x_origin

    def channel_names(self) -> list[str]:
        return list(self.channels.keys())

    def summary(self) -> dict:
        chans = []
        for name, ch in self.channels.items():
            arr = ch.as_float32() if ch.dtype != UINT32 else ch.data.astype(np.int64)
            finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.floating) else np.ones(arr.shape, bool)
            entry = {
                "name": name,
                "dtype": ch.dtype,
                "semantic": ch.semantic.value,
            }
            if finite.any():
                entry.update(min=float(np.nanmin(arr[finite])), max=float(np.nanmax(arr[finite])))
            chans.append(entry)
        return {
            "name": self.name,
            "display_window": list(self.display_window),
            "data_window": list(self.data_window),
            "width": self.width,
            "height": self.height,
            "coverage_samples": int(self.coverage.sum()),
            "missing_samples": int((~self.coverage).sum()),
            "channels": chans,
        }
