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
    FLOAT16, FLOAT32,
    DeferredExrRead, WindowMismatchError, assert_window_coherent,
    read_exr, read_fim, stitch, verify_fim_roundtrip, write_fim,
)
from float_panel.display import render_rgba, sample_tooltip  # noqa: E402
from float_panel.edit import crop_image, edit_foreground_region  # noqa: E402
from float_panel.fim_io import STATUS_INF, STATUS_NAN, STATUS_OVERFLOW  # noqa: E402
from float_panel.layers import composite_layers  # noqa: E402
from float_panel.sampling import sample_all  # noqa: E402
from float_panel.synth import write_depth_plates, write_plates  # noqa: E402

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_LOCK = threading.Lock()
STATE: dict = {
    "plates": {},       # name -> FloatImage
    "order": [],
    "composite": None,
    "owner": None,
    "last_export": None,  # dict with paths, stats, reports
    "depth_fg": None,
    "depth_bg": None,
    "depth_composite": None,
    "depth_provenance": None,
    "deferred": None,     # paused DeferredExrRead snapshot
    "edits": {},          # name -> UndoStack for the depth composite
}
WORK_DIR = os.path.join(os.path.dirname(WEB_DIR), "work")


def _json(obj):
    return json.dumps(obj, default=str).encode("utf-8")


def _ensure_assets():
    os.makedirs(WORK_DIR, exist_ok=True)
    marker = os.path.join(WORK_DIR, "plate_a.exr")
    if not os.path.exists(marker):
        write_plates(WORK_DIR)
    if not os.path.exists(os.path.join(WORK_DIR, "depth_fg.exr")):
        write_depth_plates(WORK_DIR)


def _load_depth_layers():
    fg = read_exr(os.path.join(WORK_DIR, "depth_fg.exr"))
    bg = read_exr(os.path.join(WORK_DIR, "depth_bg.exr"))
    composite, provenance = composite_layers([fg, bg])
    with STATE_LOCK:
        STATE["depth_fg"] = fg
        STATE["depth_bg"] = bg
        STATE["depth_composite"] = composite
        STATE["depth_provenance"] = provenance
    return composite


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
            if path == "/api/inspect":
                return self._handle_inspect()
            if path == "/api/deferred":
                return self._handle_deferred()
            if path == "/api/depth":
                return self._handle_depth()
            if path == "/api/edit":
                return self._handle_edit()
            if path == "/api/legacy":
                return self._handle_legacy()
            if path == "/api/order":
                return self._handle_order()
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

    # -- inspection: trace every channel from one world pixel ---------------
    def _handle_inspect(self):
        q = self._query()
        x, y = int(q.get("x", 0)), int(q.get("y", 0))
        target = q.get("target", "depth")
        table = {
            "depth": lambda: STATE["depth_composite"] or _load_depth_layers(),
            "composite": lambda: STATE["composite"] or _load_plates(),
            "fg": lambda: STATE["depth_fg"],
            "bg": lambda: STATE["depth_bg"],
        }
        src = table.get(target, table["depth"])()
        if src is None:
            return self._send_json({"error": "target not loaded"}, 404)
        self._send_json({
            "target": target,
            "sample": sample_tooltip(src, x, y),
            "channels": sample_all(src, x, y) if src.contains(x, y) else {},
        })

    # -- deferred EXR read: pause before crop, complete later ---------------
    def _handle_deferred(self):
        q = self._query()
        stage = q.get("stage", "pause")
        if stage == "pause":
            reader = DeferredExrRead(os.path.join(WORK_DIR, "plate_a.exr"))
            STATE["deferred"] = reader
            snapshot = reader.pause_before_crop()
            fresh = read_exr(os.path.join(WORK_DIR, "plate_a.exr"))
            coherent = True
            try:
                assert_window_coherent(reader.window_token, fresh)
            except WindowMismatchError:
                coherent = False
            return self._send_json({
                "stage": "paused",
                "snapshot": snapshot,
                # A freshly loaded plate has its own token; pairing the paused
                # (old) window with these (new) samples would be incoherent.
                "fresh_token": fresh.window_token,
                "old_and_new_coherent": coherent,
            })
        if stage == "finish":
            reader = STATE.get("deferred")
            if reader is None:
                return self._send_json({"error": "pause first"}, 400)
            image = reader.finish()
            fresh = read_exr(os.path.join(WORK_DIR, "plate_a.exr"))
            return self._send_json({
                "stage": "finished",
                "data_window": list(image.data_window),
                "window_token": image.window_token,
                "fresh_token": fresh.window_token,
                "tokens_distinct": image.window_token != fresh.window_token,
                "samples_match_fresh": all(
                    np.array_equal(image.channels[n].data, fresh.channels[n].data, equal_nan=True)
                    for n in image.channels
                ),
                "origin_from_paused_window": [image.x_origin, image.y_origin],
            })
        return self._send_json({"error": "unknown stage"}, 400)

    # -- descriptor-driven depth occlusion ----------------------------------
    def _handle_depth(self):
        image = STATE["depth_composite"] or _load_depth_layers()
        self._send_json({
            "summary": image.summary(),
            "provenance": {
                k: (v if not isinstance(v, np.ndarray) else {"shape": list(v.shape)})
                for k, v in STATE["depth_provenance"].items()
            },
            "depth_sources": [n for n, ch in image.channels.items()
                              if ch.semantic.value == "depth"],
            "passthrough": [n for n, ch in image.channels.items() if ch.passthrough],
        })

    # -- localized regional edit + undo on the foreground layer -------------
    def _handle_edit(self):
        q = self._query()
        action = q.get("action", "apply")
        if STATE["depth_fg"] is None:
            _load_depth_layers()
        # Edits target the foreground layer, whose channels carry the fg role;
        # a background pixel with the same suffix must never be touched.
        image = STATE["depth_fg"]
        if action == "apply":
            x, y = int(q.get("x", 0)), int(q.get("y", 0))
            size = int(q.get("size", 4))
            value = float(q.get("value", 1.5))
            base = q.get("base", "Z")
            rect = (x, y, x + size - 1, y + size - 1)
            changed, undo = edit_foreground_region(image, rect, value, base)
            STATE["edits"]["undo_stack"] = undo
            self._send_json({"action": "apply", "rect": list(rect), "changed": changed,
                             "undo_depth": undo.depth})
        elif action == "undo":
            undo = STATE["edits"].get("undo_stack")
            if undo is None:
                return self._send_json({"error": "nothing to undo"}, 400)
            undo.undo()
            self._send_json({"action": "undo", "undo_depth": undo.depth})
        elif action == "crop":
            x0, y0, x1, y1 = (int(v) for v in (q.get("x0", 0), q.get("y0", 0),
                                               q.get("x1", 1), q.get("y1", 1)))
            cropped = crop_image(image, (x0, y0, x1, y1))
            STATE["depth_fg"] = cropped
            STATE["edits"].clear()
            self._send_json({"action": "crop", "data_window": list(cropped.data_window),
                             "shape": [cropped.height, cropped.width]})
        else:
            self._send_json({"error": "unknown action"}, 400)

    # -- legacy project: same-suffix fg/bg channels in a negative window ----
    def _handle_legacy(self):
        from tests.acceptance import legacy_project_scenario
        report = legacy_project_scenario(WORK_DIR)
        self._send_json(report)

    # -- reorder channel storage in the FIM file ----------------------------
    def _handle_order(self):
        q = self._query()
        order = [n for n in q.get("channels", "").split(",") if n]
        action = q.get("action", "")
        image = STATE["depth_composite"] or _load_depth_layers()
        if action == "swap":
            # Swap the foreground custom channel ahead of the depth component.
            current = image.channel_names()
            custom = next((n for n in current if n.endswith(".custom")), None)
            depth = next((n for n, c in image.channels.items() if c.semantic.value == "depth"), None)
            if custom and depth:
                order = [n for n in current if n != custom]
                order.insert(current.index(depth), custom)
            image = image.clone(new_token=False)
            image.reorder_channels(order)
            STATE["depth_composite"] = image
        precision = FLOAT16 if q.get("precision") == FLOAT16 else FLOAT32
        path = os.path.join(WORK_DIR, "export_ordered.fim")
        if not order:
            order = list(reversed(image.channel_names()))
        stats = write_fim(path, image, precision, full_frame=False, channel_order=order)
        reopened, _, meta = read_fim(path)
        # Verify logical identity survives: compare by channel name, not order.
        values = {}
        for name, ch in image.channels.items():
            values[name] = bool(np.array_equal(ch.data, reopened.channels[name].data,
                                               equal_nan=True))
        self._send_json({
            "requested_order": order,
            "stored_order": meta["channel_order"],
            "order_preserved": meta["channel_order"] == order,
            "logical_values_match": values,
            "download": os.path.basename(path),
        })

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
    _load_depth_layers()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"float panel on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
