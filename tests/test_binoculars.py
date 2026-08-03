"""Binoculars tests — the reference semantics, pinned with synthetic logits.

`scripts/validate_binoculars.py` proves equivalence against the real reference but needs
model downloads. These run offline on hand-built tensors and exist to stop a regression
into the two mistakes validation actually caught:

  1. taking perplexity from the observer instead of the performer, and
  2. shifting the cross-entropy term, which the reference does not shift.

Both produced plausible-looking scores that were 4.6-12.4% wrong — more than the 5.6% gap
between the reference's own low-FPR and accuracy thresholds, so either could flip a verdict.
"""

import math

import pytest

torch = pytest.importorskip("torch", reason="model-bearing detectors need detbench[torch]")

from detbench.detectors.binoculars import (  # noqa: E402
    REFERENCE_ACCURACY_THRESHOLD,
    REFERENCE_FPR_THRESHOLD,
    _cross_entropy,
    _perplexity,
)

LN2 = math.log(2.0)


def _uniform(batch: int, seq: int, vocab: int):
    """Zero logits → a uniform distribution, so cross-entropy is exactly log(vocab)."""
    return torch.zeros(batch, seq, vocab)


def test_perplexity_of_a_uniform_model_is_log_vocab():
    ids = torch.tensor([[0, 1, 0, 1]])
    mask = torch.ones_like(ids)
    assert _perplexity(_uniform(1, 4, 2), ids, mask) == pytest.approx(LN2, abs=1e-6)


def test_perplexity_shifts_and_therefore_ignores_the_first_position():
    # Only positions 0..T-2 predict a token, so changing the final row of logits — which
    # predicts nothing — must not move the result.
    ids = torch.tensor([[0, 1, 0, 1]])
    mask = torch.ones_like(ids)
    a = _uniform(1, 4, 2)
    b = a.clone()
    b[0, -1, :] = torch.tensor([50.0, -50.0])
    assert _perplexity(a, ids, mask) == pytest.approx(_perplexity(b, ids, mask), abs=1e-6)


def test_perplexity_honours_the_attention_mask():
    ids = torch.tensor([[0, 1, 0, 1]])
    logits = _uniform(1, 4, 2)
    # Make one predicted position very confident, then mask it out; the masked result
    # must fall back to the uniform value.
    logits[0, 1, :] = torch.tensor([50.0, -50.0])  # predicts ids[2] == 0, near-zero loss
    full = _perplexity(logits, ids, torch.ones_like(ids))
    masked = _perplexity(logits, ids, torch.tensor([[1, 1, 0, 1]]))
    assert full < masked
    assert masked == pytest.approx(LN2, abs=1e-6)


def test_cross_entropy_of_two_uniform_models_is_log_vocab():
    ids = torch.tensor([[0, 1, 0, 1]])
    ce = _cross_entropy(_uniform(1, 4, 2), _uniform(1, 4, 2), ids, pad_token_id=99)
    assert ce == pytest.approx(LN2, abs=1e-6)


def test_cross_entropy_is_NOT_shifted():
    # The regression guard. The reference computes this term over the full sequence, so
    # the final position must matter. A shifted implementation would drop it and this
    # assertion would fail.
    ids = torch.tensor([[0, 1, 0, 1]])
    p = _uniform(1, 4, 2)
    q = _uniform(1, 4, 2)
    q_changed = q.clone()
    q_changed[0, -1, :] = torch.tensor([50.0, -50.0])
    assert _cross_entropy(p, q, ids, 99) != pytest.approx(
        _cross_entropy(p, q_changed, ids, 99), abs=1e-6
    )


def test_cross_entropy_masks_on_pad_token_not_attention():
    # The reference masks with `input_ids != pad_token_id`. Documents containing the pad
    # token (which is the EOS token when none is configured) lose those positions — an
    # upstream quirk we match deliberately rather than silently improve.
    ids = torch.tensor([[0, 1, 0, 1]])
    p = _uniform(1, 4, 2)
    q = _uniform(1, 4, 2)
    q[0, 0, :] = torch.tensor([50.0, -50.0])
    unmasked = _cross_entropy(p, q, ids, pad_token_id=99)
    # Treating token id 0 as padding drops positions 0 and 2 from the average.
    masked = _cross_entropy(p, q, ids, pad_token_id=0)
    assert masked == pytest.approx(LN2, abs=1e-6)
    assert unmasked > masked


def test_the_numerator_and_denominator_use_different_position_counts():
    # Numerator averages over T-1 positions, denominator over T. They deliberately do not
    # align, and a "tidy-up" that aligned them would silently change the metric.
    ids = torch.tensor([[0, 1, 0, 1, 0, 1]])
    logits = torch.randn(1, 6, 5)
    other = torch.randn(1, 6, 5)
    ppl = _perplexity(logits, ids, torch.ones_like(ids))
    xppl = _cross_entropy(other, logits, ids, pad_token_id=99)
    assert ppl > 0 and xppl > 0
    assert ppl != pytest.approx(xppl)


def test_published_thresholds_are_recorded_verbatim():
    # Copied from the reference detector.py. Recorded, not applied — a threshold picked
    # on one model pair and corpus does not transfer.
    assert REFERENCE_ACCURACY_THRESHOLD == 0.9015310749276843
    assert REFERENCE_FPR_THRESHOLD == 0.8536432310785527
    # Low = machine on the native scale, so the low-FPR threshold is the stricter one.
    assert REFERENCE_FPR_THRESHOLD < REFERENCE_ACCURACY_THRESHOLD
