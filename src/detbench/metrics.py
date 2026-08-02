"""Metrics, chosen to expose what AUROC hides.

The headline number in most detector papers and every vendor page is AUROC, and it is
close to useless for the decision people actually make with these tools. AUROC averages
over every operating point, including ones nobody would deploy. A detector can post a
0.98 AUROC and still label most human writing as machine-generated once the topic drifts
away from its training set (Rethinking AI-Generated Text Detection, arXiv:2607.03680,
reports 60.4% of human samples receiving p >= 0.95 under leave-one-domain-out).

So this module leads with three numbers instead:

1. `tpr_at_fpr` — how much machine text you catch at a false-positive rate you could
   defend to the person being accused. This is the deployment question.
2. `overconfidence_rate` — how much *human* text the detector is loudly certain about.
   This is the harm question, and essentially nobody publishes it.
3. `refusal_rate` — how often the detector declined. A detector that never refuses is
   not more capable, it is less honest.

AUROC is still computed, because refusing to report the number everyone else leads with
would make the comparison unfalsifiable. It is reported last, with its caveat attached.

Label convention throughout: 1 = machine-generated, 0 = human-written.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .core import Verdict


def _split(labels: Sequence[int], scores: Sequence[float]) -> tuple[list[float], list[float]]:
    pos = [s for lbl, s in zip(labels, scores) if lbl == 1]
    neg = [s for lbl, s in zip(labels, scores) if lbl == 0]
    return pos, neg


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Area under the ROC curve, via the rank-sum identity, with tie correction.

    Reported for comparability with the literature. See the module docstring for why
    it should not be the number you decide on.
    """
    pos, neg = _split(labels, scores)
    if not pos or not neg:
        return None

    paired = sorted(zip(scores, labels), key=lambda t: t[0])
    ranks: list[float] = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        # Average rank across the tied block (ranks are 1-based).
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1

    rank_sum_pos = sum(r for r, (_, lbl) in zip(ranks, paired) if lbl == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def threshold_at_fpr(
    labels: Sequence[int], scores: Sequence[float], target_fpr: float
) -> float | None:
    """Lowest score threshold whose false-positive rate stays within `target_fpr`.

    Returns None when the human sample is too small to resolve the requested rate —
    you cannot honestly quote a 0.01% false-positive rate from 500 human documents,
    because the smallest non-zero FPR you can even observe is 0.2%. Refusing here is
    the same discipline the detectors themselves are held to.
    """
    _, neg = _split(labels, scores)
    if not neg:
        return None
    if target_fpr * len(neg) < 1:
        return None

    # Allowed number of human documents at or above the threshold.
    k = math.floor(target_fpr * len(neg))

    # Ties matter, and getting this wrong silently inflates every reported TPR. Because
    # scoring uses `score >= threshold`, picking the k-th highest human score admits the
    # entire block of humans tied at that value — which can blow the false-positive
    # budget wide open while the number on the page still says 1%. So walk the distinct
    # values upward and take the lowest threshold whose *realised* count stays within
    # budget.
    asc = sorted(neg)
    n = len(asc)
    i = 0
    while i < n:
        v = asc[i]
        if n - i <= k:  # i is the first index of v, so n - i counts the values >= v
            return v
        while i < n and asc[i] == v:
            i += 1
    # No threshold admits any human document within budget: exclude them all.
    return asc[-1] + 1e-12


def tpr_at_fpr(
    labels: Sequence[int], scores: Sequence[float], target_fpr: float
) -> float | None:
    """Fraction of machine text caught while holding false positives at `target_fpr`.

    None when the sample cannot support the requested rate — see `threshold_at_fpr`.
    """
    thr = threshold_at_fpr(labels, scores, target_fpr)
    if thr is None:
        return None
    pos, _ = _split(labels, scores)
    if not pos:
        return None
    return sum(1 for s in pos if s >= thr) / len(pos)


def overconfidence_rate(
    labels: Sequence[int], p_machine: Sequence[float | None], threshold: float = 0.95
) -> float | None:
    """Fraction of **human** documents the detector called machine with p >= threshold.

    This is the number that corresponds to a real person being wrongly accused, and it
    is the one the category does not publish. Only defined for calibrated detectors;
    returns None when no human document carries a probability.
    """
    vals = [p for lbl, p in zip(labels, p_machine) if lbl == 0 and p is not None]
    if not vals:
        return None
    return sum(1 for p in vals if p >= threshold) / len(vals)


def refusal_rate(verdicts: Sequence[Verdict]) -> float:
    if not verdicts:
        return 0.0
    return sum(1 for v in verdicts if v.refused) / len(verdicts)


@dataclass(frozen=True)
class Report:
    """One detector's result on one slice of data.

    `n_scored` excludes refusals: every rate above is computed over documents the
    detector actually answered, and `refusal` tells you how much was set aside. Quoting
    accuracy without the refusal rate next to it would let a detector look good by
    answering only the easy documents.
    """

    detector: str
    slice_name: str
    n_total: int
    n_scored: int
    refusal: float
    tpr_at_1pct: float | None
    tpr_at_0_1pct: float | None
    overconfidence: float | None
    auroc: float | None

    def as_row(self) -> dict:
        def pct(x: float | None) -> str:
            return "n/a" if x is None else f"{100 * x:.1f}%"

        return {
            "detector": self.detector,
            "slice": self.slice_name,
            "n": self.n_total,
            "refused": pct(self.refusal),
            "TPR@1%FPR": pct(self.tpr_at_1pct),
            "TPR@0.1%FPR": pct(self.tpr_at_0_1pct),
            "human called machine (p>=.95)": pct(self.overconfidence),
            "AUROC": "n/a" if self.auroc is None else f"{self.auroc:.3f}",
        }


def evaluate(
    detector_name: str,
    slice_name: str,
    labels: Sequence[int],
    verdicts: Sequence[Verdict],
) -> Report:
    """Score one detector on one slice, dropping refusals from the rate calculations."""
    if len(labels) != len(verdicts):
        raise ValueError("labels and verdicts must be the same length")

    kept = [(lbl, v) for lbl, v in zip(labels, verdicts) if not v.refused]
    kept_labels = [lbl for lbl, _ in kept]
    kept_scores = [v.score for _, v in kept if v.score is not None]
    kept_probs = [v.p_machine for _, v in kept]

    return Report(
        detector=detector_name,
        slice_name=slice_name,
        n_total=len(labels),
        n_scored=len(kept),
        refusal=refusal_rate(verdicts),
        tpr_at_1pct=tpr_at_fpr(kept_labels, kept_scores, 0.01),
        tpr_at_0_1pct=tpr_at_fpr(kept_labels, kept_scores, 0.001),
        overconfidence=overconfidence_rate(kept_labels, kept_probs),
        auroc=auroc(kept_labels, kept_scores),
    )
