"""The evaluation loop: every detector, on clean text and under every attack.

The output shape is the argument. A row is not "detector X scores Y" — it is "detector X
scores Y on clean text, Y' under this attack, refused Z% of documents, and called W% of
human writing machine-generated with high confidence." A single accuracy number cannot be
printed by this harness, because there is nowhere in the row for it to go.

Attacks are applied to machine text only. Attacking human text would be measuring
something else — an evader has no reason to launder writing that was already their own,
and mixing the two would let a detector's post-attack score improve for the wrong reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from .attacks import build as build_attack
from .attacks.base import Attack
from .core import Detector, Verdict, score_many
from .data.fixtures import Document
from .metrics import Report, evaluate


@dataclass(frozen=True)
class Slice:
    """One labelled evaluation set, already transformed if an attack applies."""

    name: str
    texts: list[str]
    labels: list[int]
    mean_edit_rate: float | None = None


def build_slices(docs: list[Document], attacks: list[Attack], seed: int = 0) -> list[Slice]:
    """The clean slice, plus one slice per attack with machine text transformed."""
    texts = [d.text for d in docs]
    labels = [d.label for d in docs]
    slices = [Slice(name="clean", texts=texts, labels=labels)]

    for atk in attacks:
        out_texts: list[str] = []
        edits: list[float] = []
        for d in docs:
            if d.label == 1:
                res = atk.apply(d.text, seed=seed)
                out_texts.append(res.attacked)
                edits.append(res.edit_rate)
            else:
                out_texts.append(d.text)
        slices.append(
            Slice(
                name=f"attack:{atk.name}",
                texts=out_texts,
                labels=labels,
                mean_edit_rate=(sum(edits) / len(edits)) if edits else None,
            )
        )
    return slices


def run(
    detectors: list[Detector], docs: list[Document], attack_keys: list[str], seed: int = 0
) -> list[Report]:
    """Score every detector on every slice. Returns one Report per (detector, slice)."""
    attacks = [build_attack(k) for k in attack_keys]
    slices = build_slices(docs, attacks, seed=seed)

    reports: list[Report] = []
    for det in detectors:
        for sl in slices:
            verdicts: list[Verdict] = score_many(det, sl.texts)
            reports.append(evaluate(det.name, sl.name, sl.labels, verdicts))
    return reports


def render_table(reports: list[Report]) -> str:
    """Fixed-width table. Columns are ordered by what a reader should decide on.

    TPR at a defensible false-positive rate comes first; AUROC comes last, because a
    reader's eye lands left and the left-hand number should be the one that means
    something.
    """
    if not reports:
        return "(no results)"

    rows = [r.as_row() for r in reports]
    headers = list(rows[0].keys())
    widths = {
        h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers
    }

    def line(cells: dict) -> str:
        return "  ".join(str(cells[h]).ljust(widths[h]) for h in headers)

    sep = "  ".join("-" * widths[h] for h in headers)
    out = [line({h: h for h in headers}), sep]
    out += [line(r) for r in rows]
    return "\n".join(out)


def render_notes(reports: list[Report]) -> str:
    """The caveats that must travel with the table.

    Emitted as part of the output rather than left to a README, so that a screenshot of
    the results carries its own qualifications.
    """
    notes = []
    if any(r.tpr_at_0_1pct is None for r in reports):
        notes.append(
            "TPR@0.1%FPR is n/a where the human sample is too small to resolve that rate. "
            "Resolving 0.1% needs at least 1,000 human documents; 0.01% needs 10,000."
        )
    if any(r.overconfidence is None for r in reports):
        notes.append(
            "'human called machine' is n/a for uncalibrated detectors, which report a raw "
            "score and no probability. That is the honest default, not a missing feature."
        )
    if any(r.refusal > 0 for r in reports):
        notes.append(
            "Refused documents are excluded from every rate above. A high refusal rate "
            "next to a high TPR means the detector answered only the easy documents."
        )
    notes.append(
        "AUROC is reported last on purpose: it averages over operating points nobody "
        "deploys, and can stay high while most human text is being mislabelled."
    )
    return "\n".join(f"  - {n}" for n in notes)
