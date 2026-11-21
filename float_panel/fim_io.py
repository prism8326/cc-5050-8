"""FIM: a self-describing, re-readable floating-point image container.

Layout (all integers little-endian)::

    magic "FIM1"
    u8  version (2)
    u8  flags: bit0 = full-frame representation (otherwise sparse window)
    u16 channel_count
    i32 display_window[4]
    i32 data_window[4]      # valid-domain box, carried in both representations
    per channel (in stored order):
        u16 name_len, name bytes
        u16 source_name_len, source_name bytes
        i32 step_x, step_y          # sampling stride in world space
        i32 origin_x, origin_y      # world origin of this channel's grid
        u8  dtype_code (0=UINT32, 1=FLOAT16, 2=FLOAT32)
        u8  compression (0=raw, 1=zlib)
        u32 payload_len
        payload bytes
    coverage block:
        u8 compression
        u32 payload_len
        payload bytes (one byte/pixel, 1 = real sample, 0 = missing)
    status block:
        u8 compression
        u32 payload_len
        payload bytes (one byte/pixel; bits: 1 overflowed to half,
                       2 nan, 4 +-inf, only present for float channels)
    footer (version >= 2, optional; absent footer == all channels in-window):
        i32 -1 (sentinel; a real u16 name length can never be negative, so
                version-1 readers that ignore the footer are unaffected)
        u16 offscreen_count
        per offscreen channel:
            u16 name_len, name bytes
            i32 world_window[4]   # the rectangle this payload actually covers
            u8 dtype_code
            u8 compression
            u32 payload_len
            payload bytes         # height*width samples of that world rectangle

Version 1 (no source/stride/origin fields) is still readable.  Channel records
are written and re-read in insertion order, so callers control storage order
via ``channel_order``; the reader preserves whatever order it finds.  Channels
whose world window differs from the image data window (foreground-only
channels in a legacy project) are stored in the footer with their own window,
so their data outside the data window is preserved rather than cropped away.

Channels whose ``sample_step`` is not ``(1, 1)`` are stored on their native
grid and are *not* expanded on import; the descriptor carries enough to lazily
sample them.  Such images must use the sparse representation.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

from .model import (
    CODE_TO_DTYPE,
    DTYPE_TO_CODE,
    UINT32,
    Channel,
    ChannelDescriptor,
    FloatImage,
    NUMPY_DTYPE,
    new_window_token,
)

MAGIC = b"FIM1"
VERSION = 2
VERSION_MIN = 1
FLAG_FULL_FRAME = 1

STATUS_OVERFLOW = 1
STATUS_NAN = 2
STATUS_INF = 4

# Stored where the next channel's u16 name length would be; a name length is
# non-negative, so a negative i32 marks the optional offscreen-channel footer.
FOOTER_SENTINEL = -1


def _encode_block(data: bytes) -> tuple[int, bytes]:
    compressed = zlib.compress(data, level=6)
    if len(compressed) < len(data):
        return 1, compressed
    return 0, data


def _decode_block(raw: bytes, compression: int) -> bytes:
    if compression == 0:
        return raw
    if compression == 1:
        return zlib.decompress(raw)
    raise ValueError(f"unknown compression {compression}")


def _status_for(arr_f32: np.ndarray, stored_f16: np.ndarray) -> np.ndarray:
    status = np.zeros(arr_f32.shape, dtype=np.uint8)
    finite = np.isfinite(arr_f32)
    status[np.isnan(arr_f32)] |= STATUS_NAN
    status[np.isposinf(arr_f32) | np.isneginf(arr_f32)] |= STATUS_INF
    # Half range tops out at 65504; finite values beyond it clamp to +-inf,
    # i.e. information is destroyed -> flag the overflow.
    overflow = finite & (np.abs(arr_f32) > 65504.0)
    status[overflow] |= STATUS_OVERFLOW
    back = stored_f16.astype(np.float32)
    quant = finite & ~overflow & (back != arr_f32)
    status[quant] |= 0  # quantization is expected, not an exceptional status
    return status


def _channel_world_window(desc: ChannelDescriptor, shape) -> tuple[int, int, int, int]:
    ox, oy = desc.origin()
    sx, sy = desc.sample_step
    rows, cols = shape
    return (ox, oy, ox + (cols - 1) * sx, oy + (rows - 1) * sy)


def write_fim(path: str, image: FloatImage, precision: str, full_frame: bool,
              channel_order: list[str] | None = None) -> dict:
    """Serialize an image. precision is FLOAT16 or FLOAT32. Returns stats.

    ``channel_order`` fixes the on-disk record order; omitted channels are
    appended in their original order.  Float precision applies only to float
    channels; UINT32 and pass-through payloads keep their source dtype.
    Channels whose world window differs from the image data window are written
    to a footer with their own window so their out-of-window data survives.
    """
    if precision not in ("FLOAT16", "FLOAT32"):
        raise ValueError("precision must be FLOAT16 or FLOAT32")

    order = list(channel_order) if channel_order is not None else list(image.channels)
    missing = [n for n in order if n not in image.channels]
    if missing:
        raise KeyError(f"channel_order names unknown channels: {missing}")
    order += [n for n in image.channels if n not in order]

    strided = any(tuple(ch.descriptor.sample_step) != (1, 1) for ch in image.channels.values())
    if strided and full_frame:
        raise ValueError("strided (lazily sampled) channels require the sparse representation")

    # Strided channels are always stored in the main record (their descriptor
    # carries the grid); they must be anchored to the data window and fill it.
    # Full-resolution channels whose rectangle differs from the data window are
    # the legacy per-layer case and go to the footer with their own window.
    offscreen: list[str] = []
    for n in order:
        ch = image.channels[n]
        step = tuple(ch.descriptor.sample_step)
        if step != (1, 1):
            ox, oy = ch.descriptor.origin()
            if (ox, oy) != (image.x_origin, image.y_origin):
                raise ValueError(
                    f"strided channel {n} must be anchored to the data window origin"
                )
            exp_h = (image.height - 1) // step[1] + 1
            exp_w = (image.width - 1) // step[0] + 1
            if ch.data.shape != (exp_h, exp_w):
                raise ValueError(f"strided channel {n} shape does not match its window")
        elif ch.data.shape != (image.height, image.width):
            offscreen.append(n)
    in_window = [n for n in order if n not in offscreen]

    dx0, dy0, dx1, dy1 = image.display_window
    if full_frame:
        out_h, out_w = image.display_height, image.display_width
        r0, c0 = image.y_origin - dy0, image.x_origin - dx0
    else:
        out_h, out_w = image.height, image.width
        r0 = c0 = 0

    flags = FLAG_FULL_FRAME if full_frame else 0
    header = bytearray()
    header += MAGIC
    header += struct.pack("<BBH", VERSION, flags, len(in_window))
    header += struct.pack("<4i", *image.display_window)
    header += struct.pack("<4i", *image.data_window)

    stats = {"overflow_pixels": 0, "nan_pixels": 0, "inf_pixels": 0, "quantized_pixels": 0}
    body = bytearray()
    combined_status = np.zeros((out_h, out_w), dtype=np.uint8)

    def encode_channel(ch, in_window_data: np.ndarray | None = None):
        """Return (stored_dtype, payload_bytes, status_or_None)."""
        if ch.dtype == UINT32:
            return UINT32, np.ascontiguousarray(ch.data.astype(np.uint32)).tobytes(), None
        src_f32 = ch.as_float32()
        if precision == "FLOAT16":
            with np.errstate(over="ignore"):
                stored = src_f32.astype(np.float16)
            status = _status_for(src_f32, stored)
            if in_window_data is not None and in_window_data.shape == src_f32.shape:
                back = stored.astype(np.float32)
                finite = np.isfinite(src_f32)
                stats["quantized_pixels"] += int(
                    (finite & in_window_data & (back != src_f32)).sum()
                )
        else:
            stored = src_f32.astype(np.float32)
            status = np.zeros(src_f32.shape, dtype=np.uint8)
            status[np.isnan(src_f32)] |= STATUS_NAN
            status[np.isposinf(src_f32) | np.isneginf(src_f32)] |= STATUS_INF
        return precision, np.ascontiguousarray(stored).tobytes(), status

    for name in in_window:
        ch = image.channels[name]
        rows, cols = ch.data.shape
        stored_dtype, payload, status_src = encode_channel(ch, image.coverage)
        if full_frame:
            arr = np.frombuffer(payload, dtype=NUMPY_DTYPE[stored_dtype]).reshape(rows, cols)
            canvas = np.zeros((out_h, out_w), dtype=arr.dtype)
            canvas[r0:r0 + rows, c0:c0 + cols] = arr
            payload = np.ascontiguousarray(canvas).tobytes()
            if status_src is not None:
                status_canvas = np.zeros((out_h, out_w), dtype=np.uint8)
                status_canvas[r0:r0 + rows, c0:c0 + cols] = status_src
                combined_status |= status_canvas
        elif status_src is not None:
            step = tuple(ch.descriptor.sample_step)
            if step != (1, 1):
                # Status is stored per data-window pixel; scatter the sparse
                # grid onto its world positions.
                dense = np.zeros((image.height, image.width), dtype=np.uint8)
                sx, sy = step
                for gy in range(status_src.shape[0]):
                    for gx in range(status_src.shape[1]):
                        dense[gy * sy, gx * sx] = status_src[gy, gx]
                combined_status |= dense
            else:
                combined_status |= status_src

        comp, block = _encode_block(payload)
        name_b = name.encode("utf-8")
        source_b = ch.descriptor.source_name.encode("utf-8")
        sx, sy = ch.descriptor.sample_step
        ox, oy = ch.descriptor.origin()
        body += struct.pack("<H", len(name_b)) + name_b
        body += struct.pack("<H", len(source_b)) + source_b
        body += struct.pack("<4i", sx, sy, ox, oy)
        body += struct.pack("<BBI", DTYPE_TO_CODE[stored_dtype], comp, len(block))
        body += block

    # Coverage block (the carried valid domain of the image data window).
    if full_frame:
        coverage_full = np.zeros((out_h, out_w), dtype=bool)
        coverage_full[r0:r0 + image.height, c0:c0 + image.width] = image.coverage
        coverage_out = coverage_full
    else:
        coverage_out = image.coverage
    cov_bytes = coverage_out.astype(np.uint8).tobytes()
    comp, block = _encode_block(cov_bytes)
    body += struct.pack("<BI", comp, len(block)) + block

    # Status block.
    comp, block = _encode_block(combined_status.tobytes())
    body += struct.pack("<BI", comp, len(block)) + block

    # Footer for channels carrying their own world window (out-of-data-window).
    # The sentinel negates the max stored world-x so it can never collide with a
    # real next-record name length (which is a u16).
    if offscreen:
        footer = bytearray()
        footer += struct.pack("<i", FOOTER_SENTINEL)
        footer += struct.pack("<H", len(offscreen))
        for name in offscreen:
            ch = image.channels[name]
            stored_dtype, payload, _ = encode_channel(ch)
            win = _channel_world_window(ch.descriptor, ch.data.shape)
            comp, block = _encode_block(payload)
            name_b = name.encode("utf-8")
            footer += struct.pack("<H", len(name_b)) + name_b
            footer += struct.pack("<4i", *win)
            footer += struct.pack("<BBI", DTYPE_TO_CODE[stored_dtype], comp, len(block))
            footer += block
        body += footer

    stats["overflow_pixels"] = int(np.count_nonzero(combined_status & STATUS_OVERFLOW))
    stats["nan_pixels"] = int(np.count_nonzero(combined_status & STATUS_NAN))
    stats["inf_pixels"] = int(np.count_nonzero(combined_status & STATUS_INF))
    stats["representation"] = "full-frame" if full_frame else "sparse-window"
    stats["precision"] = precision
    stats["stored_width"] = out_w
    stats["stored_height"] = out_h
    stats["channel_order"] = order
    stats["offscreen_channels"] = offscreen

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(body)
    return stats


def read_fim(path: str) -> tuple[FloatImage, np.ndarray, dict]:
    """Read a FIM file back. Returns (image, status array in data-window coords, meta)."""
    with open(path, "rb") as fh:
        blob = fh.read()
    pos = 0
    if blob[pos:pos + 4] != MAGIC:
        raise ValueError("not a FIM file")
    pos += 4
    version, flags, channel_count = struct.unpack_from("<BBH", blob, pos)
    pos += 4
    if not (VERSION_MIN <= version <= VERSION):
        raise ValueError(f"unsupported FIM version {version}")
    display_window = struct.unpack_from("<4i", blob, pos)
    pos += 16
    data_window = struct.unpack_from("<4i", blob, pos)
    pos += 16
    full_frame = bool(flags & FLAG_FULL_FRAME)

    dx0, dy0 = display_window[0], display_window[1]
    x0, y0, x1, y1 = data_window
    dw_h = data_window[3] - data_window[1] + 1
    dw_w = data_window[2] - data_window[0] + 1
    r0, c0 = y0 - dy0, x0 - dx0

    channels: dict[str, Channel] = {}
    for _ in range(channel_count):
        (name_len,) = struct.unpack_from("<H", blob, pos)
        pos += 2
        name = blob[pos:pos + name_len].decode("utf-8")
        pos += name_len
        if version >= 2:
            (source_len,) = struct.unpack_from("<H", blob, pos)
            pos += 2
            source_name = blob[pos:pos + source_len].decode("utf-8") or name
            pos += source_len
            sx, sy, ox, oy = struct.unpack_from("<4i", blob, pos)
            pos += 16
        else:
            source_name = name
            sx, sy, ox, oy = 1, 1, x0, y0
        dtype_code, comp, payload_len = struct.unpack_from("<BBI", blob, pos)
        pos += 6
        payload = _decode_block(blob[pos:pos + payload_len], comp)
        pos += payload_len
        dtype = CODE_TO_DTYPE[dtype_code]
        arr = np.frombuffer(payload, dtype=NUMPY_DTYPE[dtype])
        strided = (sx, sy) != (1, 1)
        if full_frame:
            disp_h = display_window[3] - display_window[1] + 1
            disp_w = display_window[2] - display_window[0] + 1
            arr = arr.reshape(disp_h, disp_w)[r0:r0 + dw_h, c0:c0 + dw_w].copy()
        elif strided:
            dh = (dw_h - 1) // sy + 1
            dw = (dw_w - 1) // sx + 1
            arr = arr.reshape(dh, dw).copy()
        else:
            arr = arr.reshape(dw_h, dw_w).copy()
        channels[name] = Channel(
            name=name,
            dtype=dtype,
            data=arr,
            descriptor=ChannelDescriptor(
                name=name,
                source_name=source_name,
                sample_step=(sx, sy),
                window_origin=(ox, oy),
            ),
        )

    comp, payload_len = struct.unpack_from("<BI", blob, pos)
    pos += 5
    cov_payload = _decode_block(blob[pos:pos + payload_len], comp)
    pos += payload_len
    if full_frame:
        disp_h = display_window[3] - display_window[1] + 1
        disp_w = display_window[2] - display_window[0] + 1
        cov_full = np.frombuffer(cov_payload, dtype=np.uint8).reshape(disp_h, disp_w)
        coverage = cov_full[r0:r0 + dw_h, c0:c0 + dw_w].astype(bool).copy()
    else:
        coverage = np.frombuffer(cov_payload, dtype=np.uint8).reshape(dw_h, dw_w).astype(bool).copy()

    comp, payload_len = struct.unpack_from("<BI", blob, pos)
    pos += 5
    status_payload = _decode_block(blob[pos:pos + payload_len], comp)
    pos += payload_len
    if full_frame:
        disp_h = display_window[3] - display_window[1] + 1
        disp_w = display_window[2] - display_window[0] + 1
        status_full = np.frombuffer(status_payload, dtype=np.uint8).reshape(disp_h, disp_w)
        status = status_full[r0:r0 + dw_h, c0:c0 + dw_w].copy()
    else:
        status = np.frombuffer(status_payload, dtype=np.uint8).reshape(dw_h, dw_w).copy()

    image = FloatImage(
        name=path.split("/")[-1],
        display_window=tuple(display_window),
        data_window=tuple(data_window),
        channels=channels,
        coverage=coverage,
        source_path=path,
        window_token=new_window_token(tuple(display_window), tuple(data_window)),
        lineage=(new_window_token(tuple(display_window), tuple(data_window)),),
    )

    # Optional footer: channels whose own world window differs from the image
    # data window.  Parsed after the three required blocks.
    offscreen_order: list[str] = []
    if version >= 2 and pos + 4 <= len(blob):
        (sentinel,) = struct.unpack_from("<i", blob, pos)
        if sentinel == FOOTER_SENTINEL:
            pos += 4
            (count,) = struct.unpack_from("<H", blob, pos)
            pos += 2
            for _ in range(count):
                (name_len,) = struct.unpack_from("<H", blob, pos)
                pos += 2
                name = blob[pos:pos + name_len].decode("utf-8")
                pos += name_len
                wx0, wy0, wx1, wy1 = struct.unpack_from("<4i", blob, pos)
                pos += 16
                dtype_code, comp, payload_len = struct.unpack_from("<BBI", blob, pos)
                pos += 6
                payload = _decode_block(blob[pos:pos + payload_len], comp)
                pos += payload_len
                dtype = CODE_TO_DTYPE[dtype_code]
                w = wx1 - wx0 + 1
                h = wy1 - wy0 + 1
                arr = np.frombuffer(payload, dtype=NUMPY_DTYPE[dtype]).reshape(h, w).copy()
                prior = channels.get(name)
                source_name = prior.descriptor.source_name if prior else name
                channels[name] = Channel(
                    name=name, dtype=dtype, data=arr,
                    descriptor=ChannelDescriptor(
                        name=name, source_name=source_name,
                        sample_step=(1, 1), window_origin=(wx0, wy0),
                    ),
                )
                offscreen_order.append(name)

    meta = {
        "full_frame": full_frame,
        "version": version,
        "consumed_bytes": pos,
        "channel_order": list(channels),
        "offscreen_channels": offscreen_order,
    }
    return image, status, meta
