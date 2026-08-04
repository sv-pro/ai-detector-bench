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
from .preprocessing import get as get_preprocessor
from .metrics import Report, evaluate


@dataclass(frozen=True)
class Slice:
    """One labelled evaluation set, already transformed if an attack applies.

    `attack` and `defence` are kept separately from `name` so `recovery_table` can pair a
    defended slice with its undefended counterpart without parsing strings.
    """

    name: str
    texts: list[str]
    labels: list[int]
    attack: str | None = None
    defence: str | None = None
    mean_edit_rate: float | None = None


def build_slices(
    docs: list[Document],
    attacks: list[Attack],
    defences: list[str] | None = None,
    seed: int = 0,
) -> list[Slice]:
    """Clean and attacked slices, each optionally repeated with a defence applied.

    **The defence is applied to every document, human and machine alike.** A deployment
    normalises whatever arrives; it does not know which documents were attacked, and it
    cannot apply a repair selectively. Defending only the attacked half would measure a
    capability nobody has and would flatter the defence.
    """
    labels = [d.label for d in docs]
    defences = defences or []

    def variants(name: str, texts: list[str], attack: str | None, edit: float | None):
        out = [
            Slice(
                name=name, texts=texts, labels=labels, attack=attack, mean_edit_rate=edit
            )
        ]
        for key in defences:
            pre = get_preprocessor(key)
            out.append(
                Slice(
                    name=f"{name} +{key}",
                    texts=[pre.apply(t) for t in texts],
                    labels=labels,
                    attack=attack,
                    defence=key,
                    mean_edit_rate=edit,
                )
            )
        return out

    slices = variants("clean", [d.text for d in docs], None, None)

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
        slices += variants(
            f"attack:{atk.name}",
            out_texts,
            atk.name,
            (sum(edits) / len(edits)) if edits else None,
        )
    return slices


def run(
    detectors: list[Detector],
    docs: list[Document],
    attack_keys: list[str],
    defences: list[str] | None = None,
    seed: int = 0,
) -> list[Report]:
    """Score every detector on every slice. Returns one Report per (detector, slice)."""
    attacks = [build_attack(k) for k in attack_keys]
    slices = build_slices(docs, attacks, defences=defences, seed=seed)

    reports: list[Report] = []
    for det in detectors:
        for sl in slices:
            verdicts: list[Verdict] = score_many(det, sl.texts)
            reports.append(evaluate(det.name, sl.name, sl.labels, verdicts))
    return reports


def recovery_table(reports: list[Report], metric: str = "tpr_at_1pct") -> str:
    """How much of each attack's damage a defence undoes.

        recovery = (defended - attacked) / (clean - attacked)

    100% means the defence fully restored clean-text performance; 0% means it did nothing;
    a negative value means it made matters worse. Undefined — and reported as `n/a` — when
    the attack did no measurable damage, because dividing by roughly zero would manufacture
    a dramatic number out of noise.
    """
    by_det: dict[str, dict[str, Report]] = {}
    for r in reports:
        by_det.setdefault(r.detector, {})[r.slice_name] = r

    rows = []
    for det, slices in by_det.items():
        clean = slices.get("clean")
        if clean is None:
            continue
        for name, rep in slices.items():
            if " +" not in name or name.startswith("clean"):
                continue
            undefended = slices.get(name.split(" +")[0])
            if undefended is None:
                continue
            c = getattr(clean, metric)
            a = getattr(undefended, metric)
            d = getattr(rep, metric)
            if c is None or a is None or d is None:
                rec = None
            elif abs(c - a) < 1e-9:
                rec = None  # no damage to recover
            else:
                rec = (d - a) / (c - a)
            rows.append(
                {
                    "detector": det,
                    "slice": name,
                    "clean": f"{100 * c:.1f}%" if c is not None else "n/a",
                    "attacked": f"{100 * a:.1f}%" if a is not None else "n/a",
                    "defended": f"{100 * d:.1f}%" if d is not None else "n/a",
                    "recovery": "n/a" if rec is None else f"{100 * rec:.0f}%",
                }
            )

    if not rows:
        return "(no defended slices)"
    headers = list(rows[0].keys())
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    def line(cells: dict) -> str:
        return "  ".join(str(cells[h]).ljust(widths[h]) for h in headers)

    sep = "  ".join("-" * widths[h] for h in headers)
    title = f"attack recovery under defence ({metric})"
    return "\n".join([title, "", line({h: h for h in headers}), sep] + [line(r) for r in rows])


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
