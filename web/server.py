"""HTTP panel for the floating-point image exchange workflow.

Stdlib-only server. State: the two loaded plates, the stitched composite and the
last exported/verified FIM artifacts. Run:  python3 -m web.server [port]
"""

from __future__ import annotations

import json
import os
import struct
import sys
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from float_panel import (  # noqa: E402
    FLOAT16, FLOAT32, read_exr, read_fim, stitch, verify_fim_roundtrip, write_fim,
)
from float_panel.display import render_rgba, sample_tooltip  # noqa: E402
from float_panel.fim_io import STATUS_INF, STATUS_NAN, STATUS_OVERFLOW  # noqa: E402
from float_panel.synth import write_plates  # noqa: E402

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_LOCK = threading.Lock()
STATE: dict = {
    "plates": {},       # name -> FloatImage
    "order": [],
    "composite": None,
    "owner": None,
    "last_export": None,  # dict with paths, stats, reports
}
WORK_DIR = os.path.join(os.path.dirname(WEB_DIR), "work")


def _json(obj):
    return json.dumps(obj, default=str).encode("utf-8")


def _ensure_assets():
    os.makedirs(WORK_DIR, exist_ok=True)
    marker = os.path.join(WORK_DIR, "plate_a.exr")
    if not os.path.exists(marker):
        write_plates(WORK_DIR)


def _load_plates():
    with STATE_LOCK:
        for fname in ("plate_a.exr", "plate_b.exr"):
            img = read_exr(os.path.join(WORK_DIR, fname))
            STATE["plates"][img.name] = img
        STATE["order"] = ["plate_a.exr", "plate_b.exr"]
        composite, owner = stitch([STATE["plates"][n] for n in STATE["order"]])
        STATE["composite"] = composite
        STATE["owner"] = owner
    return composite


def _image_status(image: FloatImage):
    """Reconstruct a status array from the image itself (NaN/Inf/half overflow)."""
    status = np.zeros((image.height, image.width), np.uint8)
    for ch in image.channels.values():
        if ch.dtype == "UINT32":
            continue
        a = ch.as_float32()
        status[np.isnan(a)] |= STATUS_NAN
        status[np.isposinf(a) | np.isneginf(a)] |= STATUS_INF
        finite = np.isfinite(a)
        status[finite & (np.abs(a) > 65504.0)] |= STATUS_OVERFLOW
    return status


def _encode_png(rgba: np.ndarray) -> bytes:
    """Minimal PNG encoder (RGBA, 8-bit) using zlib, no external deps."""
    h, w = rgba.shape[:2]
    raw = bytearray()
    for row in rgba:
        raw.append(0)
        raw.extend(row.tobytes())

    def chunk(tag, data):
        out = struct.pack(">I", len(data)) + tag + data
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return out + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code=200):
        self._send(code, _json(obj), "application/json")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/state":
                return self._handle_state()
            if path == "/api/preview":
                return self._handle_preview()
            if path == "/api/sample":
                return self._handle_sample()
            if path == "/api/export":
                return self._handle_export()
            if path == "/api/reopen":
                return self._handle_reopen()
            if path.startswith("/api/download"):
                return self._handle_download()
            if path == "/api/acceptance":
                return self._handle_acceptance()
            return self._serve_static(path)
        except Exception as exc:  # surface errors to the panel
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _query(self):
        q = self.path.split("?", 1)[1] if "?" in self.path else ""
        return dict(pair.split("=", 1) for pair in q.split("&") if "=" in pair)

    def _handle_state(self):
        comp = STATE["composite"] or _load_plates()
        self._send_json({
            "plates": [STATE["plates"][n].summary() for n in STATE["order"]],
            "composite": comp.summary(),
            "stitch_meta": getattr(comp, "meta", {}),
            "last_export": STATE["last_export"],
        })

    def _handle_preview(self):
        q = self._query()
        target = q.get("target", "composite")
        precision = q.get("precision", FLOAT32)
        image = STATE["composite"] if target == "composite" else STATE["plates"].get(target)
        if image is None:
            return self._send_json({"error": "unknown target"}, 404)

        settings = {
            "mode": q.get("mode", "color"),
            "exposure": {},
            "gamma": {},
        }
        semantics = {n: ch.semantic for n, ch in image.channels.items()}
        arrays = {n: (ch.data if ch.dtype == "UINT32" else ch.as_float32()) for n, ch in image.channels.items()}

        # Half preview: quantize a copy so the UI shows the actual stored precision.
        if precision == FLOAT16:
            with np.errstate(over="ignore"):
                arrays = {
                    n: (a if image.channels[n].dtype == "UINT32"
                        else a.astype(np.float16).astype(np.float32))
                    for n, a in arrays.items()
                }
        status = _image_status(image)
        rgba = render_rgba(arrays, image.coverage, status, semantics, settings)
        self._send(200, _encode_png(rgba), "image/png")

    def _handle_sample(self):
        q = self._query()
        x, y = int(q.get("x", 0)), int(q.get("y", 0))
        target = q.get("target", "composite")
        image = STATE["composite"] if target == "composite" else STATE["plates"].get(target)
        if image is None:
            return self._send_json({"error": "unknown target"}, 404)
        self._send_json(sample_tooltip(image, x, y))

    def _handle_export(self):
        q = self._query()
        precision = FLOAT16 if q.get("precision") == FLOAT16 else FLOAT32
        representation = q.get("representation", "both")
        comp = STATE["composite"] or _load_plates()
        results = {}
        modes = [("sparse", False), ("full", True)] if representation == "both" else [
            ("sparse", False) if representation == "sparse" else ("full", True)
        ]
        for label, full in modes:
            path = os.path.join(WORK_DIR, f"export_{label}_{precision}.fim")
            stats = write_fim(path, comp, precision, full_frame=full)
            report = verify_fim_roundtrip(path, comp)
            results[label] = {"path": path, "stats": stats, "report": report}
        STATE["last_export"] = {
            "precision": precision,
            "sparse": results.get("sparse", results.get("full"))["stats"],
            "full": results.get("full", results.get("sparse"))["stats"],
        }
        self._send_json({k: {"stats": v["stats"], "report": v["report"]} for k, v in results.items()})

    def _handle_reopen(self):
        q = self._query()
        label = q.get("representation", "sparse")
        precision = q.get("precision", FLOAT32)
        path = os.path.join(WORK_DIR, f"export_{label}_{precision}.fim")
        if not os.path.exists(path):
            return self._send_json({"error": "export first"}, 404)
        image, status, meta = read_fim(path)
        reopened_summary = image.summary()
        reopened_summary["status_flags"] = {
            "overflow": int(np.count_nonzero(status & STATUS_OVERFLOW)),
            "nan": int(np.count_nonzero(status & STATUS_NAN)),
            "inf": int(np.count_nonzero(status & STATUS_INF)),
        }
        self._send_json({
            "meta": meta,
            "image": reopened_summary,
            "coverage_signature": {
                "origin": [image.x_origin, image.y_origin],
                "covered": int(image.coverage.sum()),
                "missing": int((~image.coverage).sum()),
            },
        })

    def _handle_download(self):
        q = self._query()
        name = os.path.basename(q.get("file", ""))
        path = os.path.join(WORK_DIR, name)
        if not name or not os.path.exists(path):
            return self._send_json({"error": "not found"}, 404)
        with open(path, "rb") as fh:
            data = fh.read()
        self._send(200, data, "application/octet-stream",
                   {"Content-Disposition": f'attachment; filename="{name}"'})

    def _handle_acceptance(self):
        # Run the full assertion suite and return its machine-readable report.
        from tests.acceptance import run_acceptance
        report = run_acceptance(WORK_DIR)
        self._send_json(report)

    def _serve_static(self, path):
        if path == "/":
            path = "/index.html"
        safe = os.path.normpath(path).lstrip("/")
        full = os.path.join(WEB_DIR, "static", safe)
        if not full.startswith(os.path.join(WEB_DIR, "static")) or not os.path.isfile(full):
            return self._send(404, b"not found", "text/plain")
        ctype = "text/html" if full.endswith(".html") else "text/javascript" if full.endswith(".js") else "text/css"
        with open(full, "rb") as fh:
            self._send(200, fh.read(), ctype)


def main():
    _ensure_assets()
    _load_plates()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"float panel on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
