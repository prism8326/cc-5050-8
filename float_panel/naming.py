"""Channel-name parsing, isolated from sampling and from file I/O.

A stored name is split into a layer prefix and a final component::

    fg.Z.custom  ->  layer="fg.Z"  base="custom"
    diffuse.R    ->  layer="diffuse" base="R"
    Z            ->  layer=None      base="Z"

The final component drives the *semantic* (colour / alpha / depth / mask /
unknown).  The first component of the layer prefix may name a depth role
(foreground or background), which the compositor uses for occlusion.  Nothing
here touches pixels or files, so parsers can be reused by any adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

_DEPTH_NAMES = {"z", "depth", "zback", "backz", "depthback"}
_MASK_NAMES = {"coverage", "mask", "alpha.mask"}
_ALPHA_NAMES = {"a", "alpha"}
_COLOR_NAMES = {"r", "g", "b", "red", "green", "blue"}

_FOREGROUND_LAYERS = {"fg", "foreground", "front"}
_BACKGROUND_LAYERS = {"bg", "background", "back", "rear"}


class Semantic(Enum):
    COLOR = "color"
    ALPHA = "alpha"
    DEPTH = "depth"
    MASK = "mask"
    UNKNOWN = "unknown"


class LayerRole(Enum):
    FOREGROUND = "foreground"
    BACKGROUND = "background"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class NameParts:
    source: str
    layer: str | None
    base: str
    semantic: Semantic
    role: LayerRole

    @property
    def is_unrecognized(self) -> bool:
        return self.semantic is Semantic.UNKNOWN


def split_name(name: str) -> tuple[str | None, str]:
    """Return ``(layer_prefix, final_component)`` without interpreting either."""
    if "." not in name:
        return None, name
    layer, base = name.rsplit(".", 1)
    return layer, base


def classify_channel(name: str) -> Semantic:
    """Classify the final component, including layered names like ``diffuse.R``."""
    _, base = split_name(name)
    base_l = base.lower()
    full = name.lower()
    if full in _MASK_NAMES or base_l in {"coverage", "mask"}:
        return Semantic.MASK
    if base_l in _ALPHA_NAMES:
        return Semantic.ALPHA
    if base_l in _DEPTH_NAMES or full in _DEPTH_NAMES:
        return Semantic.DEPTH
    if base_l in _COLOR_NAMES:
        return Semantic.COLOR
    return Semantic.UNKNOWN


def role_of_layer(layer: str | None) -> LayerRole:
    if not layer:
        return LayerRole.NEUTRAL
    head = layer.split(".", 1)[0].lower()
    if head in _FOREGROUND_LAYERS:
        return LayerRole.FOREGROUND
    if head in _BACKGROUND_LAYERS:
        return LayerRole.BACKGROUND
    return LayerRole.NEUTRAL


def strip_role(name: str) -> str:
    """Drop a leading role token, keeping the rest of the layer path.

    ``fg.Z.custom -> Z.custom``; ``bg.Z -> Z``; ``diffuse.R -> diffuse.R``.
    """
    layer, base = split_name(name)
    if layer is None:
        return name
    head, _, rest = layer.partition(".")
    if head.lower() in _FOREGROUND_LAYERS | _BACKGROUND_LAYERS:
        return f"{rest}.{base}" if rest else base
    return name


def parse_channel_name(name: str) -> NameParts:
    layer, base = split_name(name)
    return NameParts(
        source=name,
        layer=layer,
        base=base,
        semantic=classify_channel(name),
        role=role_of_layer(layer),
    )
