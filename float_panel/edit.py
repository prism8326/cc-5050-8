"""Window cropping, localized editing with undo, and foreground/background rules.

A project "plate" stores its own display/data windows (possibly negative).  The
brief's legacy case: one project stores its data window in **negative**
coordinates and carries two channels with the same suffix -- one belonging to
the foreground layer, one to the background layer.  After import, an edit must
target only foreground pixels; this module enforces that by channel *role*, not
by suffix alone.  The pre-edit state is captured in an :class:`UndoStack` so a
localized modification can be undone and the file rewritten back to original.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import (
    Channel, ChannelDescriptor, FloatImage, LayerRole,
    new_window_token, window_intersection,
)


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def crop_image(image: FloatImage, new_data_window) -> FloatImage:
    """Return a copy restricted to ``new_data_window`` (clipped to current).

    The display window is preserved (it describes framing, not the valid
    domain).  Coverage and every full-resolution channel are cropped in world
    space; strided channels must be expanded first.
    """
    inter = window_intersection(image.data_window, tuple(new_data_window))
    if inter is None:
        raise ValueError("crop window does not intersect the image")
    x0, y0, x1, y1 = inter
    c0, r0 = x0 - image.x_origin, y0 - image.y_origin
    c1, r1 = x1 - image.x_origin + 1, y1 - image.y_origin + 1

    channels: dict[str, Channel] = {}
    for name, ch in image.channels.items():
        desc = ch.descriptor
        if tuple(desc.sample_step) != (1, 1):
            raise ValueError(f"cannot crop strided channel {name}; expand it first")
        channels[name] = Channel(
            name=name,
            dtype=ch.dtype,
            data=ch.data[r0:r1, c0:c1].copy(),
            descriptor=ChannelDescriptor(
                name=name, source_name=desc.source_name,
                sample_step=(1, 1), window_origin=(x0, y0),
            ),
        )
    return FloatImage(
        name=image.name,
        display_window=image.display_window,
        data_window=inter,
        channels=channels,
        coverage=image.coverage[r0:r1, c0:c1].copy(),
        source_path=image.source_path,
        window_token=new_window_token(image.display_window, inter),
        lineage=tuple(image.lineage) + (image.window_token or "",),
        provenance=dict(image.provenance, cropped_from=image.data_window),
    )


# ---------------------------------------------------------------------------
# Localized edits + undo
# ---------------------------------------------------------------------------

@dataclass
class _Entry:
    region: tuple[slice, slice]          # local (row_slice, col_slice)
    channel_data: dict[str, np.ndarray]  # name -> pre-edit copy in region
    coverage: np.ndarray                 # pre-edit coverage copy in region


class UndoStack:
    """Single-image undo by region snapshot; every region edit is reversible."""

    def __init__(self, image: FloatImage):
        self.image = image
        self._frames: list[_Entry] = []

    def begin(self) -> None:
        self._frames.append(_Entry(region=(slice(0, 0), slice(0, 0)),
                                   channel_data={}, coverage=np.zeros((0, 0), bool)))

    def _capture(self, name: str, region) -> None:
        frame = self._frames[-1]
        ch = self.image.channels[name]
        if not frame.channel_data:
            frame.region = region
            frame.coverage = self.image.coverage[region].copy()
        frame.channel_data[name] = ch.data[region].copy()

    def set_region(self, world_rect, values: dict[str, float],
                   set_coverage: bool = True) -> None:
        """Write constant ``values`` to the named channels of a world rectangle."""
        if not self._frames:
            raise RuntimeError("begin() an undo frame first")
        x0, y0, x1, y1 = world_rect
        for name in values:
            if name not in self.image.channels:
                raise KeyError(name)
        r0, c0 = y0 - self.image.y_origin, x0 - self.image.x_origin
        r1, c1 = y1 - self.image.y_origin + 1, x1 - self.image.x_origin + 1
        region = (slice(r0, r1), slice(c0, c1))
        for name in values:
            self._capture(name, region)
        if set_coverage:
            self.image.coverage[region] = True
        for name, value in values.items():
            ch = self.image.channels[name]
            ch.data[region] = value

    def undo(self) -> None:
        if not self._frames:
            raise RuntimeError("nothing to undo")
        frame = self._frames.pop()
        if not frame.channel_data:
            return
        region = frame.region
        for name, before in frame.channel_data.items():
            self.image.channels[name].data[region] = before
        self.image.coverage[region] = frame.coverage

    @property
    def depth(self) -> int:
        return len(self._frames)


def foreground_channel_names(image: FloatImage) -> list[str]:
    """Names of channels owned by the foreground layer (role from descriptor)."""
    return [n for n, ch in image.channels.items()
            if ch.descriptor.role is LayerRole.FOREGROUND]


def edit_foreground_region(image: FloatImage, world_rect, value: float,
                           base: str | None = None) -> tuple[dict[str, float], UndoStack]:
    """Edit only foreground channels with the given base name in a world rect.

    ``base`` selects which logical channel (e.g. ``"Z"``); omitted means every
    foreground channel.  Background channels are skipped even when they share
    the same suffix, which is exactly the legacy-project hazard.
    """
    targets = []
    for name in foreground_channel_names(image):
        if base is None or image.channels[name].descriptor.base == base:
            targets.append(name)
    if not targets:
        raise KeyError(f"no foreground channel matches base={base!r}")
    undo = UndoStack(image)
    undo.begin()
    undo.set_region(world_rect, {name: value for name in targets})
    return {name: value for name in targets}, undo
