"""The contract tests. These encode the two claims the package refuses to break."""

import pytest

from detbench.core import (
    MIN_TOKENS_DEFAULT,
    RefusalReason,
    Verdict,
    gate_length,
    token_count,
)

LONG = " ".join(["word"] * 100)


def test_refused_verdict_cannot_carry_a_score():
    # The whole point of refusal is that no number escapes. A refusal that also
    # carried a score would let a caller quietly use it.
    with pytest.raises(ValueError):
        Verdict(detector="x", score=1.0, refused=True, reason=RefusalReason.TOO_SHORT)


def test_refusal_requires_a_reason():
    with pytest.raises(ValueError):
        Verdict(detector="x", refused=True)


def test_scored_verdict_requires_a_score():
    with pytest.raises(ValueError):
        Verdict(detector="x")


def test_probability_must_be_in_range():
    with pytest.raises(ValueError):
        Verdict(detector="x", score=1.0, p_machine=1.4)


def test_uncalibrated_verdict_reports_no_probability():
    v = Verdict(detector="x", score=2.0)
    assert v.p_machine is None


def test_gate_refuses_short_text():
    v = gate_length("x", "too short")
    assert v is not None and v.refused
    assert v.reason == RefusalReason.TOO_SHORT
    assert v.meta["min_tokens"] == MIN_TOKENS_DEFAULT


def test_gate_refuses_empty_and_whitespace():
    for text in ("", "   \n\t "):
        v = gate_length("x", text)
        assert v is not None and v.reason == RefusalReason.EMPTY


def test_gate_passes_long_text():
    assert gate_length("x", LONG) is None


def test_token_count_is_whitespace_based():
    assert token_count("a b  c\nd") == 4
