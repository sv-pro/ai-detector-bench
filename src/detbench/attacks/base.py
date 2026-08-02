"""Adversarial transforms applied to machine-generated text before scoring.

The premise of the benchmark: a detector's clean-text score is the number it advertises,
and its post-attack score is the number that matters. Published results are unambiguous
that these diverge sharply — the strongest zero-shot method (Binoculars) also degrades
the most under paraphrase, losing 0.196 F1, while a plain stylometric feature model loses
0.053 from a much lower starting point (arXiv:2605.14240). A leaderboard that reports only
clean-text accuracy actively misinforms.

Every attack here is **deterministic given a seed** so that a published row can be
reproduced exactly. Attacks that require a model to run (LLM paraphrase) declare that
dependency rather than silently degrading to something weaker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Attack(Protocol):
    """A text-to-text transform that attempts to evade detection.

    An attack must preserve meaning well enough that a human reader would accept the
    result as the same document. An attack that destroys the text wins trivially and
    tells us nothing, so `AttackResult` carries the edit rate for auditing.
    """

    name: str
    requires_model: bool

    def apply(self, text: str, seed: int = 0) -> "AttackResult": ...


@dataclass(frozen=True)
class AttackResult:
    """The transformed text plus enough provenance to audit the transform.

    `edit_rate` is the fraction of whitespace tokens changed. It exists to catch an
    attack that "succeeds" by mangling the document: if evasion rises together with a
    high edit rate, the finding is about text destruction, not about the detector.
    """

    attack: str
    original: str
    attacked: str
    edit_rate: float
    meta: dict


def edit_rate(original: str, attacked: str) -> float:
    """Fraction of whitespace tokens that differ, positionally.

    A crude measure on purpose — it is an audit signal, not a similarity metric, and a
    crude one cannot be gamed by tuning against it.
    """
    a, b = original.split(), attacked.split()
    if not a:
        return 0.0
    n = max(len(a), len(b))
    same = sum(1 for i in range(min(len(a), len(b))) if a[i] == b[i])
    return 1.0 - (same / n)
