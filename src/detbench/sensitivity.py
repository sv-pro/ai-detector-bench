"""Measure how much a preprocessing choice moves a detector's answer.

The question this module exists to answer: **does an undocumented plumbing decision move
the score more than the thing the detector is supposed to be detecting?**

If it does, then "we ran detector X" is not a reproducible statement. Two people can run
the same detector on the same document, make different unremarked choices about code
blocks or front matter, and get answers that disagree by more than the human/machine
signal itself. Every published detector number carries an invisible dependency on a
pipeline nobody described.

The headline is `sensitivity_ratio`:

    sensitivity_ratio = mean |score(variant) - score(raw)|  /  |mean(machine) - mean(human)|

read on raw text. Above **1.0**, the pipeline choice matters more than the signal, and the
detector's output is not reproducible across reasonable implementations. Below **0.1**, the
choice is genuinely plumbing and can be ignored.

A second finding this catches for free: a preprocessor that pushes documents *below the
refusal gate*. Stripping code from a code-heavy document can leave too little prose to
judge, and a detector that scored it before will now decline. That shows up as
`n_refused_variant` exceeding `n_refused_raw`, and it is a real deployment hazard rather
than an artifact — the same document, the same tool, a different answer about whether an
answer is even possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .core import Detector, Verdict
from .preprocessing import Preprocessor, get as get_preprocessor


@dataclass(frozen=True)
class SensitivityReport:
    """One detector's stability under one preprocessing variant."""

    detector: str
    preprocessor: str
    n_docs: int
    n_compared: int
    mean_abs_shift: float | None
    max_abs_shift: float | None
    signal_gap: float | None
    sensitivity_ratio: float | None
    flip_rate: float | None
    n_refused_raw: int
    n_refused_variant: int

    @property
    def verdict(self) -> str:
        """A word for the ratio, so a table row reads without arithmetic."""
        r = self.sensitivity_ratio
        if r is None:
            return "unlabelled"
        if r >= 1.0:
            return "DOMINATES"
        if r >= 0.25:
            return "material"
        if r >= 0.1:
            return "minor"
        return "negligible"

    def as_row(self) -> dict:
        def num(x: float | None, fmt: str = "{:+.3f}") -> str:
            return "n/a" if x is None else fmt.format(x)

        return {
            "detector": self.detector,
            "preprocessor": self.preprocessor,
            "n": self.n_compared,
            "mean |shift|": num(self.mean_abs_shift, "{:.3f}"),
            "max |shift|": num(self.max_abs_shift, "{:.3f}"),
            "signal gap": num(self.signal_gap),
            "ratio": num(self.sensitivity_ratio, "{:.2f}"),
            "flips": "n/a" if self.flip_rate is None else f"{100 * self.flip_rate:.0f}%",
            "newly refused": self.n_refused_variant - self.n_refused_raw,
            "verdict": self.verdict,
        }


def _score(det: Detector, text: str) -> Verdict:
    return det.score_one(text)


def _signal_gap(labels: Sequence[int] | None, scores: Sequence[float | None]) -> float | None:
    """mean(machine) - mean(human) on raw text. The scale everything else is read against."""
    if labels is None:
        return None
    h = [s for lbl, s in zip(labels, scores) if lbl == 0 and s is not None]
    m = [s for lbl, s in zip(labels, scores) if lbl == 1 and s is not None]
    if not h or not m:
        return None
    return (sum(m) / len(m)) - (sum(h) / len(h))


def measure(
    detector: Detector,
    texts: Sequence[str],
    preprocessors: Sequence[str | Preprocessor],
    labels: Sequence[int] | None = None,
) -> list[SensitivityReport]:
    """Score every text raw, then under each preprocessor, and report the displacement.

    `labels` is optional. Without it you still get shift magnitudes — useful for asking
    "is this tool stable?" — but `signal_gap`, `sensitivity_ratio` and `flip_rate` are
    `None`, because there is no signal to compare the shift against and inventing a scale
    would make the headline number meaningless.
    """
    raw_verdicts = [_score(detector, t) for t in texts]
    raw_scores: list[float | None] = [None if v.refused else v.score for v in raw_verdicts]
    n_refused_raw = sum(1 for v in raw_verdicts if v.refused)

    gap = _signal_gap(labels, raw_scores)

    # Reference line for `flip_rate`. Deliberately the midpoint between class means and
    # NOT a deployment threshold — it exists to count documents crossing a fixed line, so
    # displacement can be expressed as decisions changed rather than only as a magnitude.
    threshold: float | None = None
    if labels is not None and gap is not None:
        h = [s for lbl, s in zip(labels, raw_scores) if lbl == 0 and s is not None]
        m = [s for lbl, s in zip(labels, raw_scores) if lbl == 1 and s is not None]
        threshold = ((sum(h) / len(h)) + (sum(m) / len(m))) / 2.0

    reports: list[SensitivityReport] = []
    for p in preprocessors:
        pre = get_preprocessor(p) if isinstance(p, str) else p
        var_verdicts = [_score(detector, pre.apply(t)) for t in texts]
        n_refused_variant = sum(1 for v in var_verdicts if v.refused)

        shifts: list[float] = []
        flips = 0
        compared = 0
        for raw_v, var_v in zip(raw_verdicts, var_verdicts):
            # Only documents scored under *both* conditions can be compared. A document
            # that refuses on one side is counted in the refusal columns instead, never
            # silently treated as zero displacement.
            if raw_v.refused or var_v.refused:
                continue
            compared += 1
            shifts.append(abs(var_v.score - raw_v.score))
            if threshold is not None:
                if (raw_v.score >= threshold) != (var_v.score >= threshold):
                    flips += 1

        mean_shift = (sum(shifts) / len(shifts)) if shifts else None
        max_shift = max(shifts) if shifts else None
        ratio = (
            mean_shift / abs(gap)
            if (mean_shift is not None and gap not in (None, 0.0))
            else None
        )

        reports.append(
            SensitivityReport(
                detector=detector.name,
                preprocessor=pre.name,
                n_docs=len(texts),
                n_compared=compared,
                mean_abs_shift=mean_shift,
                max_abs_shift=max_shift,
                signal_gap=gap,
                sensitivity_ratio=ratio,
                flip_rate=(flips / compared) if (compared and threshold is not None) else None,
                n_refused_raw=n_refused_raw,
                n_refused_variant=n_refused_variant,
            )
        )
    return reports


def render_table(reports: list[SensitivityReport]) -> str:
    if not reports:
        return "(no results)"
    rows = [r.as_row() for r in reports]
    headers = list(rows[0].keys())
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    def line(cells: dict) -> str:
        return "  ".join(str(cells[h]).ljust(widths[h]) for h in headers)

    sep = "  ".join("-" * widths[h] for h in headers)
    return "\n".join([line({h: h for h in headers}), sep] + [line(r) for r in rows])


def render_notes(reports: list[SensitivityReport]) -> str:
    notes = [
        "ratio = mean |shift| / |signal gap|. Above 1.00 the preprocessing choice moves the "
        "score more than the human/machine signal does, so the detector's output is not "
        "reproducible across reasonable pipelines.",
        "'signal gap' is mean(machine) - mean(human) on raw text. A near-zero gap means the "
        "detector barely separates the classes at all, and the ratio inflates for that reason "
        "rather than because preprocessing is unusually damaging — read the gap first.",
    ]
    if any(r.n_refused_variant > r.n_refused_raw for r in reports):
        notes.append(
            "'newly refused' > 0 means a preprocessor pushed documents below the length gate. "
            "Same document, same tool, different answer about whether an answer is possible."
        )
    if any(r.sensitivity_ratio is None for r in reports):
        notes.append(
            "ratio and flips are n/a without labels: there is no signal to measure the shift "
            "against, and inventing a scale would make the headline number meaningless."
        )
    return "\n".join(f"  - {n}" for n in notes)
