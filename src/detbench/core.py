"""The contract every detector in this benchmark must satisfy.

Two design choices here are load-bearing, and both exist to stop this project from
making the claim the rest of the category makes.

**Refusal is a first-class outcome.** A detector may return `refused=True` instead of a
score. Short text carries too little signal for any published method to separate machine
from human, so a detector that answers anyway is guessing with a confident face. Refusal
is not a failure mode we tolerate; it is a result we record and report.

**A score is not a probability.** `Verdict.score` is a raw, method-specific number on
whatever scale the method defines. `Verdict.p_machine` stays `None` until a detector has
been explicitly calibrated against a held-out set, and calibration is only valid for the
distribution it was fitted on. Reporting an uncalibrated score as a percentage is the
single most common way this category misleads people, so the type system refuses to do it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

# Below this many whitespace tokens, every method surveyed in docs/METHODOLOGY.md
# degrades toward chance. The default is deliberately conservative.
MIN_TOKENS_DEFAULT = 50


class RefusalReason:
    """Why a detector declined to answer. Recorded, aggregated, and published."""

    TOO_SHORT = "too_short"
    EMPTY = "empty"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    MODEL_UNAVAILABLE = "model_unavailable"


@dataclass(frozen=True)
class Verdict:
    """One detector's answer about one document.

    Attributes:
        detector: Registry name of the detector that produced this.
        score: Raw method-specific score, or None if refused. Higher means
            "more machine-like" for every detector in this package; methods whose
            native scale runs the other way invert it at the source, so that
            comparisons across methods are meaningful.
        p_machine: Calibrated probability in [0, 1], or None. None is the default
            and the honest answer for an uncalibrated detector.
        refused: True if the detector declined to score this document.
        reason: A `RefusalReason` value when `refused`, else None.
        meta: Method-specific diagnostics (perplexities, feature values, token counts).
    """

    detector: str
    score: float | None = None
    p_machine: float | None = None
    refused: bool = False
    reason: str | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.refused:
            if self.score is not None or self.p_machine is not None:
                raise ValueError("a refused verdict must not carry a score")
            if self.reason is None:
                raise ValueError("a refused verdict must carry a reason")
        else:
            if self.score is None:
                raise ValueError("a non-refused verdict must carry a score")
        if self.p_machine is not None and not 0.0 <= self.p_machine <= 1.0:
            raise ValueError(f"p_machine out of range: {self.p_machine}")

    @classmethod
    def refuse(cls, detector: str, reason: str, **meta) -> "Verdict":
        return cls(detector=detector, refused=True, reason=reason, meta=meta)


@runtime_checkable
class Detector(Protocol):
    """What the harness requires of anything it will score.

    Implementations live in `detbench.detectors`. Third-party and commercial
    detectors are wrapped to this same interface so the leaderboard compares like
    with like — including their refusal behaviour, which most of them lack.
    """

    name: str

    def score_one(self, text: str) -> Verdict: ...


def token_count(text: str) -> int:
    """Whitespace token count.

    Deliberately not a model tokenizer: the length gate must mean the same thing
    for every detector, including ones that never load a model.
    """
    return len(text.split())


def gate_length(
    detector: str, text: str, min_tokens: int = MIN_TOKENS_DEFAULT
) -> Verdict | None:
    """Shared refusal gate. Returns a refusal Verdict, or None to proceed.

    Every detector calls this first so that "too short to judge" is answered
    identically across methods, rather than each method degrading in its own way.
    """
    if not text or not text.strip():
        return Verdict.refuse(detector, RefusalReason.EMPTY, n_tokens=0)
    n = token_count(text)
    if n < min_tokens:
        return Verdict.refuse(
            detector, RefusalReason.TOO_SHORT, n_tokens=n, min_tokens=min_tokens
        )
    return None


def score_many(detector: Detector, texts: Sequence[str]) -> list[Verdict]:
    """Default batch path. Detectors with a real batch mode override this."""
    return [detector.score_one(t) for t in texts]
