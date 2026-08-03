"""Command line interface.

`detbench demo` is the five-second version of the whole argument: it runs on the smoke
fixture, prints a table, and the table's own `n/a` columns explain why you should not
believe a small evaluation. That is the intended first experience.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .attacks import available as available_attacks
from .data.fixtures import SMOKE_WARNING, load_smoke
from .detectors import available as available_detectors, build as build_detector
from .harness import render_notes, render_table, run
from .preprocessing import available as available_preprocessors
from .sensitivity import measure
from .sensitivity import render_notes as sensitivity_notes
from .sensitivity import render_table as sensitivity_table


def cmd_list(args: argparse.Namespace) -> int:
    print("detectors:")
    for d in available_detectors():
        print(f"  {d}")
    print("attacks:")
    for a in available_attacks():
        print(f"  {a}")
    print("preprocessors:")
    for p in available_preprocessors():
        print(f"  {p}")
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    """How much does a pipeline choice move the answer?"""
    paths = [Path(p) for p in args.paths]
    if paths:
        texts = [p.read_text(encoding="utf-8", errors="replace") for p in paths]
        labels = None  # arbitrary files carry no ground truth
        source = f"{len(texts)} file(s)"
    else:
        print(f"!! {SMOKE_WARNING}\n")
        docs = load_smoke()
        texts = [d.text for d in docs]
        labels = [d.label for d in docs]
        source = "smoke fixture"

    det = build_detector(args.detector)
    reports = measure(det, texts, args.preprocessors, labels=labels)
    print(f"preprocessing sensitivity — {det.name} on {source}\n")
    print(sensitivity_table(reports))
    print("\nnotes:")
    print(sensitivity_notes(reports))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    text = sys.stdin.read() if args.text == "-" else args.text
    det = build_detector(args.detector)
    v = det.score_one(text)

    if v.refused:
        print(f"{v.detector}: REFUSED ({v.reason})")
        for k, val in v.meta.items():
            print(f"  {k}: {val}")
        # Refusal is a legitimate answer, so it is not an error exit.
        return 0

    print(f"{v.detector}: score={v.score:.4f}")
    if v.p_machine is None:
        print("  p(machine): not reported — this detector is uncalibrated.")
        print("  A raw score is comparable within one run. It is not a probability,")
        print("  and converting it to a percentage would be inventing precision.")
    else:
        print(f"  p(machine): {v.p_machine:.3f}")
        print(f"  calibrated on: {v.meta.get('fitted_on', 'unknown')}")
    return 0


def cmd_raid_fetch(args: argparse.Namespace) -> int:
    from .data import raid

    size = raid.APPROX_BYTES.get((args.split, args.adversarial), 0)
    if args.sample_mb:
        print(f"fetching first {args.sample_mb} MB of RAID {args.split} (range request)")
    else:
        print(f"fetching RAID {args.split} — approximately {size / 1e6:.0f} MB")
    path = raid.download(
        args.split,
        adversarial=args.adversarial,
        max_bytes=int(args.sample_mb * 1e6) if args.sample_mb else None,
        force=args.force,
    )
    print(f"cached at {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {raid.CITATION}")
    return 0


def cmd_raid(args: argparse.Namespace) -> int:
    from .data import raid

    try:
        docs = raid.load(
            split=args.split,
            adversarial=args.adversarial,
            domains=set(args.domains) if args.domains else None,
            limit_per_class=args.limit,
            seed=args.seed,
        )
    except (ValueError, FileNotFoundError) as exc:
        # These carry the diagnostic (which domains are actually present, which command
        # to run); a traceback would bury it.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(raid.describe(docs), "\n")

    # Only the model-bearing detectors take device/dtype; `stylometric` loads no model and
    # would reject them. Listed explicitly rather than swallowed by a **kwargs, so a typo
    # in a detector name fails loudly instead of being silently ignored.
    MODEL_BEARING = {"binoculars", "binoculars-small", "fast-detectgpt"}
    detectors = [
        build_detector(
            k, **({"device": args.device, "dtype": args.dtype} if k in MODEL_BEARING else {})
        )
        for k in args.detectors
    ]
    reports = run(detectors, docs, args.attacks, seed=args.seed)
    print(render_table(reports))
    print("\nnotes:")
    print(render_notes(reports))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    print(f"!! {SMOKE_WARNING}\n")
    docs = load_smoke()
    detectors = [build_detector(k) for k in args.detectors]
    reports = run(detectors, docs, args.attacks, seed=args.seed)
    print(render_table(reports))
    print("\nnotes:")
    print(render_notes(reports))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="detbench",
        description="Measure AI-text detectors at deployment-relevant operating points.",
    )
    p.add_argument("--version", action="version", version=f"detbench {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list registered detectors and attacks").set_defaults(
        func=cmd_list
    )

    sc = sub.add_parser("score", help="score one document")
    sc.add_argument("text", help="text to score, or '-' to read stdin")
    sc.add_argument("--detector", default="stylometric", help="detector key")
    sc.set_defaults(func=cmd_score)

    dm = sub.add_parser("demo", help="run the smoke fixture end to end")
    dm.add_argument("--detectors", nargs="+", default=["stylometric"])
    dm.add_argument(
        "--attacks", nargs="+", default=["homoglyph", "zero_width", "synonym"]
    )
    dm.add_argument("--seed", type=int, default=0)
    dm.set_defaults(func=cmd_demo)

    rf = sub.add_parser("raid-fetch", help="download a RAID split into the cache")
    rf.add_argument("--split", default="train", choices=sorted(["train", "extra", "test"]))
    rf.add_argument("--adversarial", action="store_true", help="the attacked variant (much larger)")
    rf.add_argument("--sample-mb", type=float, default=0, help="fetch only a prefix, in MB")
    rf.add_argument("--force", action="store_true")
    rf.set_defaults(func=cmd_raid_fetch)

    rd = sub.add_parser("raid", help="run the bench on a RAID subset")
    rd.add_argument("--split", default="train", choices=["train", "extra"])
    rd.add_argument("--adversarial", action="store_true")
    rd.add_argument("--limit", type=int, default=500, help="documents per class")
    rd.add_argument("--domains", nargs="*", default=None, help="default: English prose only")
    rd.add_argument("--detectors", nargs="+", default=["stylometric"])
    rd.add_argument("--attacks", nargs="*", default=["homoglyph", "zero_width", "synonym"])
    rd.add_argument("--seed", type=int, default=0)
    rd.add_argument("--device", default=None, help="cpu or cuda; default auto-detect")
    rd.add_argument(
        "--dtype",
        default="float32",
        help="float32 (default, matches the validated configuration), float16, bfloat16",
    )
    rd.set_defaults(func=cmd_raid)

    sn = sub.add_parser(
        "sensitivity",
        help="measure how much preprocessing choices move the score",
    )
    sn.add_argument(
        "paths", nargs="*", help="documents to measure; defaults to the smoke fixture"
    )
    sn.add_argument("--detector", default="stylometric")
    sn.add_argument(
        "--preprocessors",
        nargs="+",
        default=[
            "strip_front_matter",
            "strip_code",
            "strip_markdown",
            "strip_urls",
            "collapse_whitespace",
            "prose_only",
        ],
    )
    sn.set_defaults(func=cmd_sensitivity)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
