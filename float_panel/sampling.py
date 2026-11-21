"""World-coordinate sampling, decoupled from name parsing and file I/O.

Sampling policy (the brief asks which side of the lazy/dense choice we take):

* Channels keep their **native** grid.  A channel whose samples are spaced
  ``sample_step`` world units apart is *not* expanded on import: the loader
  records the stride in the channel descriptor and keeps the compact payload.
* Values are resolved **lazily at logical (world) coordinates** through
  :func:`sample_channel`.  A world point only has a value for a strided channel
  when it lands on that channel's grid; off-grid points report *absent* rather
  than inventing an interpolated value.
* :func:`expand_channel` / :func:`expand_image` are the explicit opt-in that
  materialises a dense window when some consumer genuinely needs array
  arithmetic.  Callers choose when to pay that cost.
* For unrecognized channels the raw payload is owned by the :class:`Channel`
  itself (``channel.data``) in its native layout.  Nothing converts, clips or
  reinterprets it, so a pass-through channel survives every sampling path
  byte-for-byte identical to what was stored.
"""

from __future__ import annotations

import numpy as np

from .model import Channel, ChannelDescriptor, FloatImage, new_window_token


def _on_grid(origin, step, x, y) -> bool:
    ox, oy = origin
    sx, sy = step
    return (x - ox) % sx == 0 and (y - oy) % sy == 0


def local_sample_index(image: FloatImage, channel: Channel, x: int, y: int) -> tuple[int, int] | None:
    """World point -> (row, col) in the channel's native grid, or ``None``.

    ``None`` means off the data window *or* off the channel's strided grid; both
    are genuine "no sample here" answers, not errors.
    """
    desc = channel.descriptor
    if not image.contains(x, y):
        return None
    if not _on_grid(desc.origin(), desc.sample_step, x, y):
        return None
    ox, oy = desc.origin()
    sx, sy = desc.sample_step
    row = (y - oy) // sy
    col = (x - ox) // sx
    if not (0 <= row < channel.data.shape[0] and 0 <= col < channel.data.shape[1]):
        return None
    return row, col


def sample_channel(image: FloatImage, name: str, x: int, y: int) -> dict:
    """Sample one channel at a world coordinate.

    Returns ``{"present", "on_grid", "value", "source_name", "semantic",
    "role", "passthrough"}``.  ``value`` is the raw stored value (never tone
    mapped); presence requires both coverage and an on-grid sample.
    """
    ch = image.channels[name]
    desc = ch.descriptor
    idx = local_sample_index(image, ch, x, y)
    present = idx is not None and bool(image.coverage[idx])
    value = None
    if idx is not None:
        raw = ch.data[idx]
        value = int(raw) if ch.dtype == "UINT32" else float(raw)
    return {
        "present": present,
        "on_grid": idx is not None,
        "value": value,
        "source_name": desc.source_name,
        "semantic": desc.semantic.value,
        "role": desc.role.value,
        "passthrough": desc.passthrough,
    }


def sample_all(image: FloatImage, x: int, y: int) -> dict:
    """Sample every channel from one world coordinate (inspection primitive)."""
    return {name: sample_channel(image, name, x, y) for name in image.channels}


def channel_window(channel: Channel, target_window) -> tuple[int, int, int, int]:
    """World window actually covered by a channel's native grid."""
    x0, y0, x1, y1 = target_window
    ox, oy = channel.descriptor.origin()
    sx, sy = channel.descriptor.sample_step
    rows, cols = channel.data.shape
    return (
        max(x0, ox),
        max(y0, oy),
        min(x1, ox + (cols - 1) * sx),
        min(y1, oy + (rows - 1) * sy),
    )


def expand_channel(channel: Channel, target_window, dtype=None, fill=0.0) -> np.ndarray:
    """Materialise a channel over ``target_window`` at full resolution.

    Off-grid and out-of-source points are filled with ``fill``.  This is the
    explicit alternative to lazy sampling; unknown channels are still only
    copied, never converted semantically.
    """
    x0, y0, x1, y1 = target_window
    w, h = x1 - x0 + 1, y1 - y0 + 1
    out = np.full((h, w), fill, dtype=dtype or channel.data.dtype)
    ox, oy = channel.descriptor.origin()
    sx, sy = channel.descriptor.sample_step
    rows, cols = channel.data.shape
    for wy in range(y0, y1 + 1):
        if (wy - oy) % sy != 0:
            continue
        row = (wy - oy) // sy
        if not (0 <= row < rows):
            continue
        for wx in range(x0, x1 + 1):
            if (wx - ox) % sx != 0:
                continue
            col = (wx - ox) // sx
            if 0 <= col < cols:
                out[wy - y0, wx - x0] = channel.data[row, col]
    return out


def expand_image(image: FloatImage, target_window=None) -> FloatImage:
    """Expand every strided channel to a dense image over ``target_window``.

    Only used when a consumer explicitly asks for expansion; the deferred-read
    path never calls this.  Coverage stays the image's own mask restricted to
    the target window.
    """
    tw = tuple(target_window or image.data_window)
    x0, y0, x1, y1 = tw
    channels: dict[str, Channel] = {}
    for name, ch in image.channels.items():
        dense = expand_channel(ch, tw, dtype=ch.data.dtype)
        channels[name] = Channel(
            name=name,
            dtype=ch.dtype,
            data=dense,
            descriptor=ChannelDescriptor(
                name=name, source_name=ch.descriptor.source_name,
                sample_step=(1, 1), window_origin=(x0, y0),
            ),
        )
    cov = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
    ix0, iy0 = max(x0, image.x_origin), max(y0, image.y_origin)
    ix1, iy1 = min(x1, image.data_window[2]), min(y1, image.data_window[3])
    if ix0 <= ix1 and iy0 <= iy1:
        cov[iy0 - y0:iy1 - y0 + 1, ix0 - x0:ix1 - x0 + 1] = image.coverage[
            iy0 - image.y_origin:iy1 - image.y_origin + 1,
            ix0 - image.x_origin:ix1 - image.x_origin + 1,
        ]
    return FloatImage(
        name=image.name,
        display_window=image.display_window,
        data_window=tw,
        channels=channels,
        coverage=cov,
        source_path=image.source_path,
        window_token=new_window_token(image.display_window, tw),
        lineage=tuple(image.lineage) + (image.window_token or "",),
        provenance=dict(image.provenance),
    )
