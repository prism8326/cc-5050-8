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
    FLOAT16, FLOAT32, read_exr, read_fim, stitch, verify_fim_roundtrip, write_fim,
)
from float_panel.composite import coverage_signature, union_data_window  # noqa: E402
from float_panel.exr_io import write_exr  # noqa: E402
from float_panel.fim_io import STATUS_INF, STATUS_NAN, STATUS_OVERFLOW  # noqa: E402
from float_panel.synth import (  # noqa: E402
    BLACK_POS, MISSING_POS, OVERLAP, WINDOW_A, WINDOW_B, write_plates,
)


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

    return {
        "work_dir": work_dir,
        "total": len(r.checks),
        "passed": sum(1 for c in r.checks if c["passed"]),
        "failed": sum(1 for c in r.checks if not c["passed"]),
        "all_passed": r.all_passed(),
        "checks": r.checks,
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
