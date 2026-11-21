"""Programmatic acceptance suite.

Every requirement from the brief is checked as an assertion with measured
numbers; nothing here depends on a screenshot. Run:

    python3 -m tests.acceptance            # generates fresh plates in ./work
    python3 -m tests.acceptance /tmp/dir   # uses (or creates) a given dir
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from float_panel import (  # noqa: E402
    FLOAT16, FLOAT32, LayerRole, Semantic, parse_channel_name,
    read_exr, read_fim, stitch, verify_fim_roundtrip, write_fim,
)
from float_panel.composite import coverage_signature, union_data_window  # noqa: E402
from float_panel.edit import (  # noqa: E402
    crop_image, edit_foreground_region, foreground_channel_names,
)
from float_panel.exr_io import (  # noqa: E402
    DeferredExrRead, WindowMismatchError, assert_window_coherent, write_exr,
)
from float_panel.fim_io import STATUS_OVERFLOW  # noqa: E402
from float_panel.layers import composite_layers  # noqa: E402
from float_panel.model import Channel, ChannelDescriptor, FloatImage  # noqa: E402
from float_panel.sampling import expand_image, sample_all, sample_channel  # noqa: E402
from float_panel.synth import (  # noqa: E402
    BLACK_POS, DEPTH_BACKGROUND_Z, DEPTH_FOREGROUND_Z, DEPTH_WINDOW,
    MISSING_POS, OVERLAP, write_depth_plates, write_plates,
)


def _build_legacy_project() -> FloatImage:
    """A legacy project: negative data window, fg/bg channels sharing the suffix.

    ``fg.Z`` and ``bg.Z`` both end in ``Z``; only the layer prefix distinguishes
    them.  ``fg.Z.custom`` is an unrecognized channel layered under the depth
    component and exists only on the foreground.  The plate records a missing
    sample and a legitimate exact zero so both survive crop/edit/rewrite.
    """
    win = (-20, -10, 19, 9)          # 40 x 20, negative origin
    x0, y0, x1, y1 = win
    h, w = y1 - y0 + 1, x1 - x0 + 1
    xs = np.arange(x0, x0 + w)[None, :]
    ys = np.arange(y0, y0 + h)[:, None]

    fg_z = np.full((h, w), 5.0, np.float32)
    bg_z = np.full((h, w), 25.0, np.float32)
    # Legitimate exact zero depth at a real, covered foreground pixel.
    fg_z[(ys == 0) & (xs == -10)] = 0.0
    custom = (xs.astype(np.float32) + ys.astype(np.float32) * 0.5)
    custom[0, 0] = np.nan             # preserved verbatim by pass-through
    custom[1, 1] = 0.0                # exact zero preserved verbatim
    fg_r = np.full((h, w), 0.9, np.float32)
    bg_r = np.full((h, w), 0.1, np.float32)

    coverage = np.ones((h, w), bool)
    # A missing sample inside the later crop rectangle, so "missing" must be
    # carried through the crop rather than turning into a real zero.
    coverage[(ys == 0) & (xs == -15)] = False   # missing sample, not black
    coverage[(ys == 2) & (xs == 0)] = False     # inside crop (-12,-5,12,5)

    def chan(name, data):
        return Channel(name, FLOAT32, np.ascontiguousarray(data),
                       ChannelDescriptor(name, name, (1, 1), (x0, y0)))

    return FloatImage(
        name="legacy_project",
        display_window=(-30, -20, 29, 19),
        data_window=win,
        channels={
            "fg.Z": chan("fg.Z", fg_z),
            "bg.Z": chan("bg.Z", bg_z),
            "fg.R": chan("fg.R", fg_r),
            "bg.R": chan("bg.R", bg_r),
            "fg.Z.custom": chan("fg.Z.custom", custom),
        },
        coverage=coverage,
    )


def legacy_project_scenario(work_dir: str, recorder: "CheckRecorder | None" = None) -> dict:
    """Full legacy-project regression: import, fg-only edit, crop, undo, rewrite.

    Returns a machine-readable report (used by ``/api/legacy``); when a recorder
    is supplied the same assertions are logged as acceptance checks.
    """
    checks: list[dict] = []

    def check(name, condition, detail=""):
        ok = bool(condition)
        checks.append({"name": name, "passed": ok, "detail": detail})
        if recorder is not None:
            recorder.check(name, ok, detail)
        return ok

    os.makedirs(work_dir, exist_ok=True)
    project = _build_legacy_project()
    fim_path = os.path.join(work_dir, "acc_legacy.fim")
    write_fim(fim_path, project, FLOAT32, full_frame=False)
    imported, _, _ = read_fim(fim_path)

    # Negative window survived the round trip.
    check("legacy: negative data window preserved",
          imported.data_window == project.data_window and imported.x_origin < 0,
          f"window={imported.data_window}")

    # Same suffix, different layers: descriptor role disambiguates.
    fg_names = foreground_channel_names(imported)
    check("legacy: fg/bg depth share suffix but split by role",
          parse_channel_name("fg.Z").base == parse_channel_name("bg.Z").base == "Z"
          and parse_channel_name("fg.Z").role is LayerRole.FOREGROUND
          and parse_channel_name("bg.Z").role is LayerRole.BACKGROUND
          and set(fg_names) == {"fg.Z", "fg.R", "fg.Z.custom"},
          f"fg={sorted(fg_names)}")

    # Edit one foreground region only.
    edit_rect = (-18, -8, -16, -6)
    changed, undo = edit_foreground_region(imported, edit_rect, 1.0, base="Z")
    check("legacy: edit touches only the foreground depth channel",
          set(changed) == {"fg.Z"}, f"changed={sorted(changed)}")
    ex, ey = -17, -7
    fr, fc = imported.pixel_index(ex, ey)
    check("legacy: foreground depth edited in the region",
          imported.channels["fg.Z"].data[fr, fc] == 1.0,
          f"fg.Z={imported.channels['fg.Z'].data[fr, fc]}")
    check("legacy: background depth untouched by foreground edit",
          imported.channels["bg.Z"].data[fr, fc] == 25.0,
          f"bg.Z={imported.channels['bg.Z'].data[fr, fc]}")

    # Crop: outside-window coordinates become absent; missing stays missing.
    cropped = crop_image(imported, (-12, -5, 12, 5))
    check("legacy: crop shrinks the world window",
          cropped.data_window == (-12, -5, 12, 5), str(cropped.data_window))
    check("legacy: cropped-away coordinates no longer sample",
          not cropped.contains(-19, -9), "corner outside crop")
    mr, mc = cropped.pixel_index(-10, 0)
    check("legacy: legitimate zero depth survives crop",
          cropped.channels["fg.Z"].data[mr, mc] == 0.0,
          f"fg.Z={cropped.channels['fg.Z'].data[mr, mc]}")
    # A missing sample *inside* the crop must remain missing (not become zero,
    # and not be filled by the crop operation).
    in_r, in_c = cropped.pixel_index(0, 2)
    check("legacy: in-crop missing sample stays missing after crop",
          not cropped.coverage[in_r, in_c]
          and cropped.channels["fg.Z"].data[in_r, in_c] is not None,
          f"coverage={cropped.coverage[in_r, in_c]}")
    # A missing sample outside the crop is simply gone, not reinterpreted.
    missing_r, missing_c = imported.pixel_index(-15, 0)
    check("legacy: out-of-crop missing sample is absent from the crop",
          not cropped.contains(-15, 0) and not imported.coverage[missing_r, missing_c],
          "coverage=0")

    # Undo restores the pre-edit foreground value.
    undo.undo()
    check("legacy: undo restores foreground depth",
          imported.channels["fg.Z"].data[fr, fc] == 5.0,
          f"fg.Z={imported.channels['fg.Z'].data[fr, fc]}")
    check("legacy: undo leaves background depth unchanged",
          imported.channels["bg.Z"].data[fr, fc] == 25.0)

    # Rewrite the (undone) plate and re-read: background, holes and zeros intact.
    rewritten_path = os.path.join(work_dir, "acc_legacy_rewritten.fim")
    write_fim(rewritten_path, imported, FLOAT32, full_frame=False)
    back, _, _ = read_fim(rewritten_path)
    check("legacy: rewritten file keeps negative origin",
          back.data_window == imported.data_window)
    check("legacy: rewritten background depth identical",
          np.array_equal(back.channels["bg.Z"].data, imported.channels["bg.Z"].data))
    br, bc = back.pixel_index(-15, 0)
    check("legacy: rewritten outside/absent sample still missing",
          not back.coverage[br, bc], "coverage=0")
    # Also rewrite the cropped plate: the in-crop hole must survive.
    cropped_path = os.path.join(work_dir, "acc_legacy_cropped.fim")
    write_fim(cropped_path, cropped, FLOAT32, full_frame=False)
    crop_back, _, _ = read_fim(cropped_path)
    cb_r, cb_c = crop_back.pixel_index(0, 2)
    check("legacy: in-crop missing sample stays missing after rewrite",
          not crop_back.coverage[cb_r, cb_c])
    check("legacy: crop result keeps shrunken data window on rewrite",
          crop_back.data_window == (-12, -5, 12, 5))
    zr, zc = back.pixel_index(-10, 0)
    check("legacy: rewritten legitimate zero depth still exact",
          back.channels["fg.Z"].data[zr, zc] == 0.0)

    # Custom pass-through channel: raw payload owned by the channel, verbatim.
    custom = back.channels["fg.Z.custom"]
    check("legacy: custom channel is unknown and pass-through",
          custom.semantic is Semantic.UNKNOWN and custom.passthrough)
    nr, nc = back.pixel_index(-20, -10)   # custom[0,0] == NaN
    check("legacy: pass-through NaN preserved verbatim",
          np.isnan(custom.data[nr, nc]))
    z0r, z0c = back.pixel_index(-19, -9)  # custom[1,1] == 0.0
    check("legacy: pass-through exact zero preserved verbatim",
          custom.data[z0r, z0c] == 0.0)
    check("legacy: unknown channel raw bytes unchanged by rewrite",
          np.array_equal(custom.data, imported.channels["fg.Z.custom"].data,
                         equal_nan=True))

    return {
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
        "measured": {
            "fg_channels": sorted(fg_names),
            "edit_rect": list(edit_rect),
            "crop_window": list(cropped.data_window),
        },
    }


class CheckRecorder:
    def __init__(self):
        self.checks = []

    def check(self, name, condition, detail=""):
        self.checks.append({"name": name, "passed": bool(condition), "detail": detail})
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name} {detail}")
        return bool(condition)

    def all_passed(self):
        return all(c["passed"] for c in self.checks)


def run_acceptance(work_dir: str) -> dict:
    os.makedirs(work_dir, exist_ok=True)
    r = CheckRecorder()

    # 1. Generate plates and read windows / named channels back from EXR.
    path_a, path_b = write_plates(work_dir)
    a = read_exr(path_a)
    b = read_exr(path_b)

    r.check("EXR read: negative data-window origin A",
            a.x_origin < 0 and a.y_origin < 0, f"origin={a.x_origin,a.y_origin}")
    r.check("EXR read: negative data-window origin B",
            b.x_origin < 0 and b.y_origin < 0, f"origin={b.x_origin,b.y_origin}")
    r.check("EXR read: display window matches on both",
            a.display_window == b.display_window, str(a.display_window))
    expected_channels = {"R", "G", "B", "Z", "customHeat"}
    r.check("EXR read: named channels parsed",
            set(a.channels) == expected_channels and set(b.channels) == expected_channels,
            str(sorted(a.channels)))
    r.check("EXR read: custom channel classified unknown, Z depth",
            a.channels["customHeat"].semantic.value == "unknown"
            and a.channels["Z"].semantic.value == "depth")
    r.check("EXR read: data windows actually overlap",
            a.contains(*OVERLAP[:2]) and b.contains(*OVERLAP[:2]), str(OVERLAP))

    # 2. Real black vs missing sample are distinct concepts.
    ra, ca = a.pixel_index(*BLACK_POS)
    rb_, cb_ = b.pixel_index(*BLACK_POS)
    r.check("black point: real opaque zero on A",
            a.coverage[ra, ca] and np.all(a.channels[c].data[ra, ca] == 0.0
                                          for c in ("R", "G", "B")))
    r.check("black point: missing sample on B at same world coord",
            not b.coverage[rb_, cb_], f"world={BLACK_POS}")
    rm, cm = a.pixel_index(*MISSING_POS)
    r.check("missing point: absent on A, real black on B",
            (not a.coverage[rm, cm]) and b.coverage[b.pixel_index(*MISSING_POS)[0],
                                                     b.pixel_index(*MISSING_POS)[1]])

    # 3. Negative light, >1 highlights, front/back signed depth.
    red = a.channels["R"].as_float32()
    r.check("content: negative light values present",
            bool((red[a.coverage] < 0).any()), f"min={red[a.coverage].min():.3f}")
    green = a.channels["G"].as_float32()
    r.check("content: highlights above 1.0 present",
            bool((green[a.coverage] > 1.0).any()), f"max={green[a.coverage].max():.3f}")
    z = a.channels["Z"].as_float32()
    r.check("content: signed depth (behind-camera) and far depth",
            bool((z < 0).any()) and bool((z > 15).any()),
            f"min={z.min():.2f} max={z.max():.2f}")

    # 4. Non-finite and half-overflow values are carried through.
    def flags(img):
        nan = inf = ovf = 0
        for ch in img.channels.values():
            if ch.dtype == "UINT32":
                continue
            v = ch.as_float32()
            nan += int(np.isnan(v).sum())
            inf += int(np.isinf(v).sum())
            ovf += int((np.isfinite(v) & (np.abs(v) > 65504.0)).sum())
        return nan, inf, ovf

    nan_a, inf_a, ovf_a = flags(a)
    r.check("content: NaN and Inf samples in plates", nan_a >= 1 and inf_a >= 1,
            f"nan={nan_a} inf={inf_a}")
    r.check("content: half-range overflow sample present", ovf_a >= 1, f"overflow={ovf_a}")

    # 5. Stitch: world-coordinate placement, coverage-driven merge.
    total_cover = int(a.coverage.sum()) + int(b.coverage.sum())
    comp, owner = stitch([a, b])
    r.check("stitch: composite window is union of data windows",
            comp.data_window == union_data_window([a, b]),
            str(comp.data_window))
    overlap_box = OVERLAP
    oh = overlap_box[3] - overlap_box[1] + 1
    ow = overlap_box[2] - overlap_box[0] + 1
    shared_overlap_pixels = oh * ow  # geometry of the rectangle intersection
    # Union covered = sum of per-plate samples minus world positions covered
    # by both plates (holes do not displace any real sample).
    overlap_covered_positions = int(comp.meta["overlap_agreements"] / len(comp.channels))
    expected_covered = total_cover - overlap_covered_positions
    r.check("stitch: covered count = plate samples minus doubly-covered positions",
            int(comp.coverage.sum()) == expected_covered,
            f"composite covered={int(comp.coverage.sum())} expected={expected_covered} "
            f"agreements={comp.meta['overlap_agreements']}")
    # The black/missing pair must resolve to a covered, exactly-zero pixel.
    rk, ck = comp.pixel_index(*BLACK_POS)
    r.check("stitch: real black survives where other plate had a hole",
            comp.coverage[rk, ck] and comp.channels["R"].data[rk, ck] == 0.0)
    rk2, ck2 = comp.pixel_index(*MISSING_POS)
    r.check("stitch: hole filled from other plate, still exact zero",
            comp.coverage[rk2, ck2] and comp.channels["G"].data[rk2, ck2] == 0.0)
    # Black/missing pairs never reach the conflict counter (one side has no
    # sample), so every position where both plates are covered must agree.
    r.check("stitch: zero conflicts where both plates carry real samples",
            comp.meta["overlap_conflicts"] == 0,
            f"conflicts={comp.meta['overlap_conflicts']} "
            f"agreements={comp.meta['overlap_agreements']}")
    r.check("stitch: overlap contains substantial agreeing real samples",
            comp.meta["overlap_agreements"] > 1000,
            f"agreements={comp.meta['overlap_agreements']}")

    # 6. Sparse-window vs full-frame FIM must give identical coverage results.
    fim_sparse = os.path.join(work_dir, "acc_sparse.fim")
    fim_full = os.path.join(work_dir, "acc_full.fim")
    stats_s = write_fim(fim_sparse, comp, FLOAT32, full_frame=False)
    stats_f = write_fim(fim_full, comp, FLOAT32, full_frame=True)
    img_s, _, meta_s = read_fim(fim_sparse)
    img_f, _, meta_f = read_fim(fim_full)
    r.check("fim: representations flagged correctly",
            meta_s["full_frame"] is False and meta_f["full_frame"] is True)
    r.check("fim: both reopen at identical world origin",
            img_s.x_origin == comp.x_origin and img_f.x_origin == comp.x_origin
            and img_s.y_origin == comp.y_origin)
    r.check("fim: sparse vs full coverage masks identical after placement",
            coverage_signature(img_s) == coverage_signature(img_f))
    r.check("fim: pixel values identical between representations",
            all(np.array_equal(img_s.channels[n].data, img_f.channels[n].data, equal_nan=True)
                for n in img_s.channels))
    # Receiver stitches independently from the two representations.
    re_s, _ = stitch([img_s])
    re_f, _ = stitch([img_f])
    r.check("fim: receiver coverage result identical for both representations",
            coverage_signature(re_s) == coverage_signature(re_f)
            == coverage_signature(comp))
    r.check("fim: full frame carries explicit valid-domain box",
            stats_f["stored_width"] == comp.display_width and img_f.data_window == comp.data_window)

    # 7. Half precision: quantization measured, overflow flagged, NaN/Inf kind kept.
    fim_half = os.path.join(work_dir, "acc_half.fim")
    stats_h = write_fim(fim_half, comp, FLOAT16, full_frame=False)
    half_img, half_status, _ = read_fim(fim_half)
    r.check("half: stored dtype is FLOAT16",
            all(ch.dtype == FLOAT16 for ch in half_img.channels.values()
                if ch.dtype != "UINT32"))
    r.check("half: overflow pixels flagged in status",
            stats_h["overflow_pixels"] >= 1 and np.count_nonzero(half_status & STATUS_OVERFLOW) >= 1)
    r.check("half: NaN/Inf counts preserved in status block",
            stats_h["nan_pixels"] >= 1 and stats_h["inf_pixels"] >= 1)
    src_R, back_R = comp.channels["R"].as_float32(), half_img.channels["R"].as_float32()
    finite = np.isfinite(src_R) & comp.coverage & (np.abs(src_R) <= 65504.0)
    max_err = float(np.abs(src_R[finite] - back_R[finite]).max())
    r.check("half: finite-channel max abs error bounded (~half epsilon level)",
            np.isfinite(max_err) and max_err <= 2.0,
            f"max_abs_err={max_err:.3e}")

    # NaN must always survive; source Inf keeps its sign; half-range finite
    # overflows are allowed to become Inf precisely because they are flagged.
    for name, ch in comp.channels.items():
        if ch.dtype == "UINT32":
            continue
        src, out = ch.as_float32(), half_img.channels[name].as_float32()
        nan_ok = np.array_equal(np.isnan(src), np.isnan(out))
        src_inf = np.isposinf(src) | np.isneginf(src)
        sign_ok = (np.array_equal(np.isposinf(src[src_inf]), np.isposinf(out[src_inf]))
                   and np.array_equal(np.isneginf(src[src_inf]), np.isneginf(out[src_inf])))
        r.check(f"half: NaN and Inf sign preserved ({name})", nan_ok and sign_ok)
    r.check("half: single precision round trip is exact",
            all(np.array_equal(comp.channels[n].as_float32(),
                               img_s.channels[n].as_float32(), equal_nan=True)
                for n in comp.channels if comp.channels[n].dtype != "UINT32"))

    # 8. Reopened FIM full verification report (world coords, mask, quantization).
    report = verify_fim_roundtrip(fim_half, comp)
    r.check("verify: reopened data/display windows match source",
            report["pixel_comparison"]["data_window_equal"]
            and report["pixel_comparison"]["display_window_equal"]
            and report["coverage_equal_to_source"])
    r.check("verify: non-finite kinds preserved on every float channel",
            all(c.get("nan_preserved", True) for c in report["pixel_comparison"]["channels"].values()))
    r.check("verify: covered/missing counts reported",
            report["covered_pixels"] == int(comp.coverage.sum())
            and report["missing_pixels"] == int((~comp.coverage).sum()))

    # 9. EXR re-write keeps window origin and mask.
    exr_back = os.path.join(work_dir, "acc_roundtrip.exr")
    write_exr(exr_back, a)
    again = read_exr(exr_back)
    r.check("exr: rewritten file keeps negative origin and coverage mask",
            again.data_window == a.data_window
            and np.array_equal(again.coverage, a.coverage))
    r.check("exr: rewritten pixel values exact (float32)",
            all(np.array_equal(again.channels[n].data, a.channels[n].data, equal_nan=True)
                for n in a.channels))

    # 10. Unknown/depth channel numeric meaning retained (no tone mapping in data).
    r.check("semantics: customHeat raw numbers preserved through FIM",
            np.allclose(comp.channels["customHeat"].as_float32()[comp.coverage],
                        img_s.channels["customHeat"].as_float32()[comp.coverage]))
    r.check("semantics: depth raw numbers preserved through FIM",
            np.allclose(comp.channels["Z"].as_float32()[comp.coverage],
                        img_s.channels["Z"].as_float32()[comp.coverage]))

    # 11. Descriptors carry source name, sampling step and window origin.
    for name, ch in comp.channels.items():
        desc = ch.descriptor
        r.check(f"descriptor: {name} has source/step/origin",
                desc.source_name and desc.sample_step == (1, 1)
                and desc.window_origin == (comp.x_origin, comp.y_origin),
                f"source={desc.source_name} step={desc.sample_step} origin={desc.window_origin}")

    # 12. Deferred EXR read: pause before crop, complete later, windows coherent.
    reader = DeferredExrRead(path_a)
    snapshot = reader.pause_before_crop()
    fresh = read_exr(path_a)
    r.check("deferred: pause happens before cropping",
            snapshot["cropped"] is False and snapshot["payload_read"] is False
            and snapshot["raw_channels"],
            f"raw={snapshot['raw_channels']} payload_read={snapshot['payload_read']}")
    deferred_coherent = True
    try:
        assert_window_coherent(reader.window_token, fresh)
    except WindowMismatchError:
        deferred_coherent = False
    r.check("deferred: paused token differs from a fresh reload",
            not deferred_coherent)
    finished = reader.finish()
    r.check("deferred: completion uses the paused window, not the reloaded one",
            list(finished.data_window) == snapshot["data_window"]
            and finished.window_token == snapshot["window_token"]
            and finished.window_token != fresh.window_token,
            f"paused={finished.window_token} fresh={fresh.window_token}")
    r.check("deferred: old-window payload matches an independent read of same file",
            all(np.array_equal(finished.channels[n].data, fresh.channels[n].data, equal_nan=True)
                for n in finished.channels))
    # The loaded plate keeps its single coherent token; sampling never mixes.
    r.check("deferred: loaded image pins one coherent window token",
            assert_window_coherent(a) == a.window_token)

    # 13. Descriptor-driven depth occlusion with background pass-through.
    fg_p, bg_p = write_depth_plates(work_dir)
    fg_img = read_exr(fg_p)
    bg_img = read_exr(bg_p)
    depth_comp, prov = composite_layers([fg_img, bg_img])
    fx, ocx = -10, 5
    fz = depth_comp.channels["Z"].as_float32()
    fr, fc = depth_comp.pixel_index(fx, 0)
    orr, occ = depth_comp.pixel_index(ocx, 0)
    r.check("depth: nearer foreground wins where it is in front",
            fz[fr, fc] == DEPTH_FOREGROUND_Z, f"Z={fz[fr, fc]}")
    r.check("depth: background wins where foreground is behind it",
            fz[orr, occ] == DEPTH_BACKGROUND_Z, f"Z={fz[orr, occ]}")
    r.check("depth: provenance reports where each depth came from",
            prov["depth_sources"] == ["fg.Z|bg.Z"]
            and prov["foreground_pixels"] > 0 and prov["background_pixels"] > 0,
            str(prov["depth_sources"]))
    custom = depth_comp.channels["Z.custom"]
    r.check("depth: unrecognized channel passes through, never treated as depth",
            custom.semantic is Semantic.UNKNOWN and custom.passthrough
            and depth_comp.channels["Z"].semantic is Semantic.DEPTH)
    # Pass-through is opaque: verbatim regardless of which surface wins depth.
    fg_src = fg_img.channels["fg.Z.custom"].data
    fg_r_, fg_c_ = fg_img.pixel_index(fx, 0)
    oc_bg_r, oc_bg_c = fg_img.pixel_index(ocx, 0)
    dr_, dc_ = depth_comp.pixel_index(fx, 0)
    o_r, o_c = depth_comp.pixel_index(ocx, 0)
    r.check("depth: pass-through value verbatim where foreground wins",
            custom.data[dr_, dc_] == fg_src[fg_r_, fg_c_])
    r.check("depth: pass-through value still verbatim where background wins",
            custom.data[o_r, o_c] == fg_src[oc_bg_r, oc_bg_c],
            "custom never follows the depth winner")
    # Non-finite pass-through survives the depth composite.
    nan_pos = (DEPTH_WINDOW[0], DEPTH_WINDOW[1])   # fg custom[0,0] is NaN
    nr, nc = depth_comp.pixel_index(*nan_pos)
    r.check("depth: pass-through NaN survives compositing",
            np.isnan(custom.data[nr, nc]))

    # 14. Lazy sampling of a strided channel; explicit expansion is opt-in.
    sx0, sy0 = DEPTH_WINDOW[0], DEPTH_WINDOW[1]
    sh, sw = (DEPTH_WINDOW[3] - sy0) // 2 + 1, (DEPTH_WINDOW[2] - sx0) // 2 + 1
    stride_data = np.arange(sh * sw, dtype=np.float32).reshape(sh, sw)
    strided = FloatImage(
        name="strided",
        display_window=DEPTH_WINDOW,
        data_window=DEPTH_WINDOW,
        channels={"probe": Channel("probe", FLOAT32, stride_data,
                                   ChannelDescriptor("probe", "probe", (2, 2), (sx0, sy0)))},
        coverage=np.ones((DEPTH_WINDOW[3] - sy0 + 1, DEPTH_WINDOW[2] - sx0 + 1), bool),
    )
    on = sample_channel(strided, "probe", sx0, sy0)
    off = sample_channel(strided, "probe", sx0 + 1, sy0)
    r.check("sampling: strided channel resolves on-grid, absent off-grid",
            on["present"] and on["value"] == 0.0 and not off["on_grid"],
            f"on={on['value']} off_grid={off['on_grid']}")
    sp = os.path.join(work_dir, "acc_strided.fim")
    write_fim(sp, strided, FLOAT32, full_frame=False)
    strided_back, _, _ = read_fim(sp)
    r.check("sampling: strided channel stored compactly, not expanded on import",
            strided_back.channels["probe"].descriptor.sample_step == (2, 2)
            and strided_back.channels["probe"].data.shape == (sh, sw),
            f"step={strided_back.channels['probe'].descriptor.sample_step} "
            f"shape={strided_back.channels['probe'].data.shape}")
    expanded = expand_image(strided_back)
    r.check("sampling: expansion is explicit and dense",
            expanded.channels["probe"].descriptor.sample_step == (1, 1)
            and expanded.channels["probe"].data.shape
            == (DEPTH_WINDOW[3] - sy0 + 1, DEPTH_WINDOW[2] - sx0 + 1))

    # 15. FIM upholds requested channel order and logical identity.
    forward = os.path.join(work_dir, "acc_order_forward.fim")
    reverse = os.path.join(work_dir, "acc_order_reverse.fim")
    names = list(depth_comp.channels)
    write_fim(forward, depth_comp, FLOAT32, full_frame=False, channel_order=names)
    write_fim(reverse, depth_comp, FLOAT32, full_frame=False, channel_order=names[::-1])
    fwd, _, fmeta = read_fim(forward)
    rev, _, rmeta = read_fim(reverse)
    r.check("order: requested storage order is preserved on read",
            fmeta["channel_order"] == names and rmeta["channel_order"] == names[::-1],
            f"fwd={fmeta['channel_order']} rev={rmeta['channel_order']}")
    r.check("order: swapping storage order keeps every logical value identical",
            all(np.array_equal(fwd.channels[n].data, rev.channels[n].data, equal_nan=True)
                for n in names))

    # 16. Regression precision by world coordinate AND raw payload.
    #      'Looks the same' is not enough: the depth handed downstream at a
    #      background pixel must be the *background* layer's payload, not the
    #      foreground's.
    world = (ocx, 0)
    sampled = sample_all(depth_comp, *world)
    r.check("regression: depth at a background world pixel is background payload",
            sampled["Z"]["value"] == DEPTH_BACKGROUND_Z
            and sampled["Z"]["value"] != DEPTH_FOREGROUND_Z,
            f"Z={sampled['Z']['value']}")
    # Round-trip the depth composite and confirm the same world coordinate holds
    # the same numeric depth (guards against a stale/other-layer depth).
    dc_path = os.path.join(work_dir, "acc_depth.fim")
    write_fim(dc_path, depth_comp, FLOAT32, full_frame=False)
    dc_back, _, _ = read_fim(dc_path)
    br_, bc_ = dc_back.pixel_index(*world)
    r.check("regression: rewritten depth at the same world pixel matches",
            dc_back.channels["Z"].data[br_, bc_] == DEPTH_BACKGROUND_Z,
            f"Z={dc_back.channels['Z'].data[br_, bc_]}")
    # And the foreground pixel keeps the foreground value, proving the test can
    # tell the two apart rather than merely passing on a uniform field.
    fr2, fc2 = dc_back.pixel_index(fx, 0)
    r.check("regression: foreground depth at its own world pixel is foreground payload",
            dc_back.channels["Z"].data[fr2, fc2] == DEPTH_FOREGROUND_Z,
            f"Z={dc_back.channels['Z'].data[fr2, fc2]}")

    # 17. Storage-order swap for the foreground custom channel in the FIM file.
    #      'fg.Z.custom' (unknown, pass-through) must survive a reorder while the
    #      depth channel still carries the visible surface.
    swapped = os.path.join(work_dir, "acc_swapped.fim")
    swap_order = [n for n in names[::-1]]
    write_fim(swapped, depth_comp, FLOAT32, full_frame=False, channel_order=swap_order)
    sw_back, _, sw_meta = read_fim(swapped)
    r.check("pass-through: custom channel raw payload identical after reorder",
            np.array_equal(sw_back.channels["Z.custom"].data,
                           depth_comp.channels["Z.custom"].data, equal_nan=True))
    r.check("pass-through: reordering does not swap depth payloads between layers",
            sw_back.channels["Z"].data[dc_back.pixel_index(*world)] == DEPTH_BACKGROUND_Z)

    # 18. Legacy project: negative window, fg/bg same suffix, fg-only edit.
    legacy = legacy_project_scenario(work_dir, r)

    return {
        "work_dir": work_dir,
        "total": len(r.checks),
        "passed": sum(1 for c in r.checks if c["passed"]),
        "failed": sum(1 for c in r.checks if not c["passed"]),
        "all_passed": r.all_passed(),
        "checks": r.checks,
        "legacy": legacy,
        "measured": {
            "overflow_pixels_half": stats_h["overflow_pixels"],
            "nan_pixels_half": stats_h["nan_pixels"],
            "inf_pixels_half": stats_h["inf_pixels"],
            "quantized_pixels_half": stats_h["quantized_pixels"],
            "half_max_abs_error_R": max_err,
            "composite_covered": int(comp.coverage.sum()),
            "composite_missing": int((~comp.coverage).sum()),
        },
    }


def main():
    work_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work")
    report = run_acceptance(work_dir)
    print("\n=== acceptance summary ===")
    print(f"{report['passed']}/{report['total']} checks passed")
    print("measured:", report["measured"])
    sys.exit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
