"""Depth-occlusion compositing driven by channel descriptors.

The merge node does not guess from array order or file order: it reads each
channel's :class:`ChannelDescriptor` to learn the semantic (depth vs. colour),
the depth role (foreground vs. background) and the sampling grid, then resolves
the visible surface per world pixel by comparing foreground/background depth.

Unrecognized channels never take part in the depth test.  They are copied
verbatim (``passthrough``), so a custom channel layered under a depth channel
survives with its original values and is never mistaken for depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import (
    Channel, ChannelDescriptor, FloatImage, LayerRole, Semantic,
    new_window_token, strip_role, window_union,
)


@dataclass
class LayerView:
    role: LayerRole
    origin: tuple[int, int]
    coverage: np.ndarray
    depth: np.ndarray | None
    depth_name: str | None
    colors: dict[str, np.ndarray]          # role-stripped base -> values
    passthrough: dict[str, tuple[str, np.ndarray]]  # base -> (source_name, values)
    shape: tuple[int, int]


def _role_value(role: LayerRole) -> int:
    return {LayerRole.FOREGROUND: 1, LayerRole.BACKGROUND: 2}.get(role, 0)


def discover_layers(image: FloatImage, default_role: LayerRole = LayerRole.NEUTRAL) -> list[LayerView]:
    """Split an image into per-role layer views using channel descriptors."""
    groups: dict[LayerRole, dict] = {}
    for name, ch in image.channels.items():
        desc = ch.descriptor
        role = desc.role if desc.role is not LayerRole.NEUTRAL else default_role
        g = groups.setdefault(role, {"depth": None, "depth_name": None, "colors": {}, "passthrough": {}})
        if desc.semantic is Semantic.DEPTH and g["depth"] is None:
            g["depth"] = ch.as_float32()
            g["depth_name"] = name
        elif desc.semantic is Semantic.COLOR:
            g["colors"][strip_role(name)] = ch.as_float32()
        elif desc.passthrough:
            g["passthrough"][strip_role(name)] = (desc.source_name, ch.as_float32())

    views = []
    for role, g in groups.items():
        views.append(LayerView(
            role=role,
            origin=(image.x_origin, image.y_origin),
            coverage=image.coverage,
            depth=g["depth"],
            depth_name=g["depth_name"],
            colors=g["colors"],
            passthrough=g["passthrough"],
            shape=(image.height, image.width),
        ))
    return views


def _place(arr: np.ndarray, origin, full_window) -> np.ndarray:
    """Place a layer array into the union window, filling the rest with NaN."""
    x0, y0, x1, y1 = full_window
    w, h = x1 - x0 + 1, y1 - y0 + 1
    out = np.full((h, w), np.nan, dtype=np.float32)
    ox, oy = origin
    r0, c0 = oy - y0, ox - x0
    out[r0:r0 + arr.shape[0], c0:c0 + arr.shape[1]] = arr
    return out


def _place_bool(arr: np.ndarray, origin, full_window, fill=False) -> np.ndarray:
    x0, y0, x1, y1 = full_window
    w, h = x1 - x0 + 1, y1 - y0 + 1
    out = np.full((h, w), fill, dtype=bool)
    ox, oy = origin
    r0, c0 = oy - y0, ox - x0
    out[r0:r0 + arr.shape[0], c0:c0 + arr.shape[1]] = arr
    return out


def composite_layers(images: list[FloatImage]) -> tuple[FloatImage, dict]:
    """Merge foreground/background layers by depth, per world pixel.

    Requires exactly one foreground and one background layer carrying a depth
    channel; raises otherwise so callers fall back to coverage stitching.
    Returns the visible surface image plus provenance (visibility map and the
    source name that supplied each pixel's depth).
    """
    views = [v for img in images for v in discover_layers(img)]
    fg = [v for v in views if v.role is LayerRole.FOREGROUND and v.depth is not None]
    bg = [v for v in views if v.role is LayerRole.BACKGROUND and v.depth is not None]
    if len(fg) != 1 or len(bg) != 1:
        raise ValueError("composite_layers needs exactly one foreground and one background depth layer")
    fg, bg = fg[0], bg[0]

    full = window_union(
        (fg.origin[0], fg.origin[1],
         fg.origin[0] + fg.shape[1] - 1, fg.origin[1] + fg.shape[0] - 1),
        (bg.origin[0], bg.origin[1],
         bg.origin[0] + bg.shape[1] - 1, bg.origin[1] + bg.shape[0] - 1),
    )

    fg_cov = _place_bool(fg.coverage, fg.origin, full)
    bg_cov = _place_bool(bg.coverage, bg.origin, full)
    fg_z = _place(fg.depth, fg.origin, full)
    bg_z = _place(bg.depth, bg.origin, full)
    fg_ok = fg_cov & np.isfinite(fg_z)
    bg_ok = bg_cov & np.isfinite(bg_z)

    # Nearest surface wins; foreground wins exact ties.  Legitimate zero depth
    # is a real value, so presence is driven by coverage, never by value.
    use_fg = fg_ok & (~bg_ok | (fg_z <= bg_z))
    use_bg = bg_ok & ~use_fg
    visible = use_fg | use_bg

    channels: dict[str, Channel] = {}
    source_names: list[str] = []

    # Colour: take the visible layer's value for each base component.
    for base in sorted(set(fg.colors) | set(bg.colors)):
        out = np.zeros(visible.shape, dtype=np.float32)
        fga = _place(fg.colors[base], fg.origin, full) if base in fg.colors else None
        bga = _place(bg.colors[base], bg.origin, full) if base in bg.colors else None
        if fga is not None:
            out[use_fg] = fga[use_fg]
        if bga is not None:
            out[use_bg] = bga[use_bg]
        channels[base] = Channel(
            name=base, dtype="FLOAT32", data=out,
            descriptor=ChannelDescriptor(base, source_name=f"composite:{base}",
                                         window_origin=(full[0], full[1])),
        )

    # Depth: the visible layer's depth, which is what downstream must receive.
    fg_zf = np.where(use_fg, fg_z, np.nan)
    bg_zf = np.where(use_bg, bg_z, np.nan)
    depth = np.where(use_fg, fg_zf, bg_zf).astype(np.float32)
    depth_name = strip_role(fg.depth_name or "Z")
    channels[depth_name] = Channel(
        name=depth_name, dtype="FLOAT32", data=depth,
        descriptor=ChannelDescriptor(depth_name, source_name="composite:Z",
                                     window_origin=(full[0], full[1])),
    )
    source_names.append(f"{fg.depth_name}|{bg.depth_name}")

    # Pass-through: copied verbatim from the owning layer wherever that layer
    # has coverage, independent of which surface wins the depth test. An
    # unrecognized channel is opaque payload -- never depth-tested, never
    # masked by value; NaN and exact-zero samples survive unchanged because
    # selection uses coverage, not finiteness. Pixels the owning layer does not
    # cover get NaN, never a fabricated zero that would masquerade as a real
    # sample.
    for base, (src_name, values) in fg.passthrough.items():
        placed = _place(values, fg.origin, full)
        out = np.where(fg_cov, placed, np.nan).astype(np.float32)
        channels[base] = Channel(
            name=base, dtype="FLOAT32", data=out,
            descriptor=ChannelDescriptor(base, source_name=src_name,
                                         window_origin=(full[0], full[1])),
        )
    for base, (src_name, values) in bg.passthrough.items():
        if base in channels:
            continue
        placed = _place(values, bg.origin, full)
        out = np.where(bg_cov, placed, np.nan).astype(np.float32)
        channels[base] = Channel(
            name=base, dtype="FLOAT32", data=out,
            descriptor=ChannelDescriptor(base, source_name=src_name,
                                         window_origin=(full[0], full[1])),
        )

    image = FloatImage(
        name="depth-composite",
        display_window=images[0].display_window,
        data_window=full,
        channels=channels,
        coverage=visible,
        window_token=new_window_token(images[0].display_window, full),
        lineage=tuple(tok for img in images for tok in (img.lineage or (img.window_token,))),
    )
    provenance = {
        "visible_role": np.where(use_fg, _role_value(LayerRole.FOREGROUND),
                                 np.where(use_bg, _role_value(LayerRole.BACKGROUND), 0)).astype(np.uint8),
        "foreground_pixels": int(use_fg.sum()),
        "background_pixels": int(use_bg.sum()),
        "missing_pixels": int((~visible).sum()),
        "depth_sources": source_names,
        "window_token": image.window_token,
    }
    return image, provenance
