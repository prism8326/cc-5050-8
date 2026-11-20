"""Command-line entry point for EXR <-> FIM conversion and verification.

Examples:
  python3 -m tools.cli synth ./work
  python3 -m tools.cli convert input.exr out.fim --precision float16 --frame full
  python3 -m tools.cli reopen out.fim
  python3 -m tools.cli verify out.fim --source input.exr
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from float_panel import FLOAT16, FLOAT32, read_exr, read_fim, write_fim  # noqa: E402
from float_panel.verify import verify_fim_roundtrip  # noqa: E402


def cmd_synth(args):
    from float_panel.synth import write_plates
    os.makedirs(args.dir, exist_ok=True)
    a, b = write_plates(args.dir)
    print("wrote", a)
    print("wrote", b)


def cmd_convert(args):
    image = read_exr(args.input)
    precision = FLOAT16 if args.precision == "float16" else FLOAT32
    stats = write_fim(args.output, image, precision, full_frame=(args.frame == "full"))
    print(json.dumps(stats, indent=2))


def cmd_reopen(args):
    image, status, meta = read_fim(args.input)
    print(json.dumps({
        "meta": meta,
        "summary": image.summary(),
        "status": {
            "overflow": int(np.count_nonzero(status & 1)),
            "nan": int(np.count_nonzero(status & 2)),
            "inf": int(np.count_nonzero(status & 4)),
        },
    }, indent=2, default=str))


def cmd_verify(args):
    source = read_exr(args.source)
    report = verify_fim_roundtrip(args.fim, source)
    print(json.dumps(report, indent=2, default=str))
    sys.exit(0 if report["coverage_equal_to_source"] else 1)


def main():
    p = argparse.ArgumentParser(description="EXR/FIM floating-point exchange CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth"); s.add_argument("dir"); s.set_defaults(func=cmd_synth)

    c = sub.add_parser("convert"); c.add_argument("input"); c.add_argument("output")
    c.add_argument("--precision", choices=["float16", "float32"], default="float32")
    c.add_argument("--frame", choices=["sparse", "full"], default="sparse")
    c.set_defaults(func=cmd_convert)

    r = sub.add_parser("reopen"); r.add_argument("input"); r.set_defaults(func=cmd_reopen)

    v = sub.add_parser("verify"); v.add_argument("fim"); v.add_argument("--source", required=True)
    v.set_defaults(func=cmd_verify)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
