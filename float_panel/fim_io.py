"""FIM: a self-describing, re-readable floating-point image container.

Layout (all integers little-endian):

    magic "FIM1"
    u8  version (1)
    u8  flags: bit0 = full-frame representation (otherwise sparse window)
    u16 channel_count
    i32 display_window[4]
    i32 data_window[4]      # valid-domain box, carried in both representations
    per channel:
        u16 name_len, name bytes
        u8  dtype_code (0=UINT32, 1=FLOAT16, 2=FLOAT32)
        u8  compression (0=raw, 1=zlib)
        u64 payload_len
        payload bytes
    coverage block:
        u8 compression
        u64 payload_len
        payload bytes (one byte/pixel, 1 = real sample, 0 = missing)
    status block:
        u8 compression
        u64 payload_len
        payload bytes (one byte/pixel; bits: 1 overflowed to half,
                       2 nan, 4 +-inf, only present for float channels)

The sparse representation stores only the data-window rectangle. The full
representation expands to the display window while keeping the data window as
the carried valid domain; pixels outside the domain have coverage 0. Both are
lossless relative to the source for the chosen precision and stitch identically.
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
    FloatImage,
    NUMPY_DTYPE,
)

MAGIC = b"FIM1"
VERSION = 1
FLAG_FULL_FRAME = 1

STATUS_OVERFLOW = 1
STATUS_NAN = 2
STATUS_INF = 4


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


def _place_in_display(image: FloatImage) -> tuple[np.ndarray | None, np.ndarray, tuple[int, int, int, int]]:
    """Return (channel-independent placement helper) coverage in display coords."""
    dx0, dy0, dx1, dy1 = image.display_window
    dw, dh = image.display_width, image.display_height
    x0, y0, x1, y1 = image.data_window
    coverage_full = np.zeros((dh, dw), dtype=bool)
    r0, c0 = y0 - dy0, x0 - dx0
    coverage_full[r0:r0 + image.height, c0:c0 + image.width] = image.coverage
    return None, coverage_full, (r0, c0, r0 + image.height, c0 + image.width)


def _status_for(arr_f32: np.ndarray, stored_f16: np.ndarray) -> np.ndarray:
    status = np.zeros(arr_f32.shape, dtype=np.uint8)
    finite = np.isfinite(arr_f32)
    status[np.isnan(arr_f32)] |= STATUS_NAN
    status[np.isposinf(arr_f32) | np.isneginf(arr_f32)] |= STATUS_INF
    # Half range tops out at 65504; finite values beyond it clamp to +-inf,
    # i.e. information is destroyed -> flag the overflow.
    overflow = finite & (np.abs(arr_f32) > 65504.0)
    status[overflow] |= STATUS_OVERFLOW
    # Also flag finite values whose half round-trip changes magnitude visibly.
    back = stored_f16.astype(np.float32)
    quant = finite & ~overflow & (back != arr_f32)
    status[quant] |= 0  # quantization is expected, not an exceptional status
    return status


def write_fim(path: str, image: FloatImage, precision: str, full_frame: bool) -> dict:
    """Serialize an image. precision is FLOAT16 or FLOAT32. Returns stats."""
    if precision not in ("FLOAT16", "FLOAT32"):
        raise ValueError("precision must be FLOAT16 or FLOAT32")

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
    header += struct.pack("<BBH", VERSION, flags, len(image.channels))
    header += struct.pack("<4i", *image.display_window)
    header += struct.pack("<4i", *image.data_window)

    stats = {"overflow_pixels": 0, "nan_pixels": 0, "inf_pixels": 0, "quantized_pixels": 0}
    body = bytearray()
    combined_status = np.zeros((out_h, out_w), dtype=np.uint8)

    for name, ch in image.channels.items():
        if ch.dtype == UINT32:
            stored = ch.data.astype(np.uint32)
            stored_dtype = UINT32
            src_f32 = None
        else:
            src_f32 = ch.as_float32()
            if precision == "FLOAT16":
                with np.errstate(over="ignore"):
                    stored = src_f32.astype(np.float16)
                ch_status_src = _status_for(src_f32, stored)
                if full_frame:
                    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
                    canvas[r0:r0 + image.height, c0:c0 + image.width] = ch_status_src
                    combined_status |= canvas
                else:
                    combined_status |= ch_status_src
                back = stored.astype(np.float32)
                finite = np.isfinite(src_f32)
                stats["quantized_pixels"] += int(
                    (finite & image.coverage & (back != src_f32)).sum()
                )
            else:
                stored = src_f32.astype(np.float32)
                src_status = np.zeros(src_f32.shape, dtype=np.uint8)
                src_status[np.isnan(src_f32)] |= STATUS_NAN
                src_status[np.isposinf(src_f32) | np.isneginf(src_f32)] |= STATUS_INF
                if full_frame:
                    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
                    canvas[r0:r0 + image.height, c0:c0 + image.width] = src_status
                    combined_status |= canvas
                else:
                    combined_status |= src_status
            stored_dtype = precision

        if full_frame:
            canvas = np.zeros((out_h, out_w), dtype=stored.dtype)
            canvas[r0:r0 + image.height, c0:c0 + image.width] = stored
            stored = canvas

        payload = np.ascontiguousarray(stored).tobytes()
        comp, block = _encode_block(payload)
        name_b = name.encode("utf-8")
        body += struct.pack("<H", len(name_b)) + name_b
        body += struct.pack("<BBI", DTYPE_TO_CODE[stored_dtype], comp, len(block))
        body += block

    # Coverage block (the carried valid domain).
    if full_frame:
        _, coverage_out, _ = _place_in_display(image)
    else:
        coverage_out = image.coverage
    cov_bytes = coverage_out.astype(np.uint8).tobytes()
    comp, block = _encode_block(cov_bytes)
    body += struct.pack("<BI", comp, len(block)) + block

    # Status block.
    comp, block = _encode_block(combined_status.tobytes())
    body += struct.pack("<BI", comp, len(block)) + block

    stats["overflow_pixels"] = int(np.count_nonzero(combined_status & STATUS_OVERFLOW))
    stats["nan_pixels"] = int(np.count_nonzero(combined_status & STATUS_NAN))
    stats["inf_pixels"] = int(np.count_nonzero(combined_status & STATUS_INF))
    stats["representation"] = "full-frame" if full_frame else "sparse-window"
    stats["precision"] = precision
    stats["stored_width"] = out_w
    stats["stored_height"] = out_h

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
    if version != VERSION:
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
        dtype_code, comp, payload_len = struct.unpack_from("<BBI", blob, pos)
        pos += 6
        payload = _decode_block(blob[pos:pos + payload_len], comp)
        pos += payload_len
        dtype = CODE_TO_DTYPE[dtype_code]
        arr = np.frombuffer(payload, dtype=NUMPY_DTYPE[dtype])
        if full_frame:
            disp_h = display_window[3] - display_window[1] + 1
            disp_w = display_window[2] - display_window[0] + 1
            arr = arr.reshape(disp_h, disp_w)[r0:r0 + dw_h, c0:c0 + dw_w].copy()
        else:
            arr = arr.reshape(dw_h, dw_w).copy()
        channels[name] = Channel(name=name, dtype=dtype, data=arr)

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
    )
    meta = {"full_frame": full_frame, "consumed_bytes": pos}
    return image, status, meta
