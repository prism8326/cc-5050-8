"""In-memory floating-point image model.

A :class:`FloatImage` keeps every channel as a numpy array shaped by the
*data window* (which may be sparse and may carry a negative origin), plus a
boolean coverage array distinguishing real samples from missing samples.

Channel-name interpretation lives in :mod:`float_panel.naming`, sampling in
:mod:`float_panel.sampling` and file write-back in the ``*_io`` adapters, so a
channel can be parsed without touching pixels and sampled without touching a
file.  Every :class:`Channel` carries a :class:`ChannelDescriptor` recording
where its samples came from, how densely they were sampled and where its grid
is anchored in world space.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .naming import (  # re-exported for existing callers
    LayerRole,
    Semantic,
    classify_channel,
    parse_channel_name,
    role_of_layer,
    split_name,
    strip_role,
)

FLOAT16 = "FLOAT16"
FLOAT32 = "FLOAT32"
UINT32 = "UINT32"

DTYPE_TO_CODE = {FLOAT16: 1, FLOAT32: 2, UINT32: 0}
CODE_TO_DTYPE = {v: k for k, v in DTYPE_TO_CODE.items()}
NUMPY_DTYPE = {FLOAT16: np.float16, FLOAT32: np.float32, UINT32: np.uint32}

_TOKEN_SEQ = itertools.count(1)


def new_window_token(display_window, data_window) -> str:
    """Opaque identity for one coherent (display, data) window snapshot.

    Consumers compare tokens to refuse combining pixels read with one window
    against coordinates taken from a different one.
    """
    return f"{display_window}|{data_window}|{next(_TOKEN_SEQ)}"


# ---------------------------------------------------------------------------
# Tuple-window helpers (kept tuple-based so the existing public API is stable)
# ---------------------------------------------------------------------------

def window_size(window) -> tuple[int, int]:
    x0, y0, x1, y1 = window
    return x1 - x0 + 1, y1 - y0 + 1


def window_contains(window, x: int, y: int) -> bool:
    x0, y0, x1, y1 = window
    return x0 <= x <= x1 and y0 <= y <= y1


def window_intersection(a, b) -> Optional[tuple[int, int, int, int]]:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x0 > x1 or y0 > y1:
        return None
    return (x0, y0, x1, y1)


def window_union(a, b) -> tuple[int, int, int, int]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------

@dataclass
class ChannelDescriptor:
    """Where a channel's samples came from and how they are laid out.

    ``source_name``  name exactly as stored in the source file (may carry a
                     layer prefix such as ``fg.Z.custom``).
    ``sample_step``  world-space stride between samples, ``(sx, sy)``; ``(1,1)``
                     means full resolution.  Strided channels are sampled
                     lazily at logical coordinates instead of being expanded on
                     import.
    ``window_origin`` world coordinate of the grid's top-left sample.  ``None``
                     until the owning image resolves it from its data window.
    """

    name: str
    source_name: str
    sample_step: tuple[int, int] = (1, 1)
    window_origin: Optional[tuple[int, int]] = None

    def __post_init__(self):
        parts = parse_channel_name(self.name)
        self.layer = parts.layer
        self.base = parts.base
        self.semantic = parts.semantic
        self.role = parts.role

    @property
    def passthrough(self) -> bool:
        """Unrecognized semantics are carried verbatim, never interpreted."""
        return self.semantic is Semantic.UNKNOWN

    def origin(self) -> tuple[int, int]:
        return self.window_origin or (0, 0)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source_name": self.source_name,
            "sample_step": list(self.sample_step),
            "window_origin": list(self.origin()) if self.window_origin else None,
            "semantic": self.semantic.value,
            "role": self.role.value,
            "layer": self.layer,
            "passthrough": self.passthrough,
        }


def descriptor_for(name: str, *, source_name: str | None = None,
                   sample_step=(1, 1), window_origin=None) -> ChannelDescriptor:
    return ChannelDescriptor(
        name=name,
        source_name=source_name if source_name is not None else name,
        sample_step=tuple(sample_step),
        window_origin=window_origin,
    )


@dataclass
class Channel:
    name: str
    dtype: str
    data: np.ndarray  # shape (height, width) of the channel's sample grid
    descriptor: Optional[ChannelDescriptor] = None

    @property
    def semantic(self) -> Semantic:
        if self.descriptor is not None:
            return self.descriptor.semantic
        return classify_channel(self.name)

    @property
    def role(self) -> LayerRole:
        if self.descriptor is not None:
            return self.descriptor.role
        return role_of_layer(split_name(self.name)[0])

    @property
    def passthrough(self) -> bool:
        return self.semantic is Semantic.UNKNOWN

    def as_float32(self) -> np.ndarray:
        return self.data.astype(np.float32)


@dataclass
class FloatImage:
    name: str
    display_window: tuple[int, int, int, int]  # inclusive x0, y0, x1, y1
    data_window: tuple[int, int, int, int]
    channels: dict[str, Channel]
    coverage: np.ndarray  # bool, shape (height, width)
    source_path: str | None = None
    window_token: str | None = None
    lineage: tuple[str, ...] = ()
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        x0, y0 = self.data_window[0], self.data_window[1]
        for name, ch in self.channels.items():
            if ch.descriptor is None:
                ch.descriptor = descriptor_for(name)
            if ch.descriptor.window_origin is None:
                ch.descriptor.window_origin = (x0, y0)
        if self.window_token is None:
            self.window_token = new_window_token(self.display_window, self.data_window)

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
        return window_contains(self.data_window, x, y)

    def pixel_index(self, x: int, y: int) -> tuple[int, int]:
        """World coordinate -> local array (row, col) of the data window."""
        if not self.contains(x, y):
            raise KeyError(f"world coordinate ({x},{y}) outside data window {self.data_window}")
        return y - self.y_origin, x - self.x_origin

    def channel_names(self) -> list[str]:
        return list(self.channels.keys())

    def reorder_channels(self, order: list[str]) -> None:
        """Replace the channel mapping so insertion order matches ``order``.

        Omitted channels keep their relative order at the end.  This only
        changes storage order, never values or descriptors.
        """
        missing = [n for n in order if n not in self.channels]
        if missing:
            raise KeyError(f"unknown channels: {missing}")
        ordered = {name: self.channels[name] for name in order}
        for name, ch in self.channels.items():
            if name not in ordered:
                ordered[name] = ch
        self.channels = ordered

    def clone(self, *, new_token: bool = False) -> "FloatImage":
        """Deep copy with independent arrays; provenance windows stay coherent."""
        channels = {
            name: Channel(name=ch.name, dtype=ch.dtype, data=ch.data.copy(),
                          descriptor=ChannelDescriptor(
                              name=ch.descriptor.name,
                              source_name=ch.descriptor.source_name,
                              sample_step=tuple(ch.descriptor.sample_step),
                              window_origin=tuple(ch.descriptor.origin()),
                          ))
            for name, ch in self.channels.items()
        }
        token = new_window_token(self.display_window, self.data_window) if new_token else self.window_token
        return FloatImage(
            name=self.name,
            display_window=tuple(self.display_window),
            data_window=tuple(self.data_window),
            channels=channels,
            coverage=self.coverage.copy(),
            source_path=self.source_path,
            window_token=token,
            lineage=tuple(self.lineage) or (self.window_token or "",),
            provenance=dict(self.provenance),
        )

    def summary(self) -> dict:
        chans = []
        for name, ch in self.channels.items():
            arr = ch.as_float32() if ch.dtype != UINT32 else ch.data.astype(np.int64)
            finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.floating) else np.ones(arr.shape, bool)
            entry = {
                "name": name,
                "dtype": ch.dtype,
                "semantic": ch.semantic.value,
                "descriptor": ch.descriptor.to_dict() if ch.descriptor else None,
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
            "window_token": self.window_token,
            "channels": chans,
        }
