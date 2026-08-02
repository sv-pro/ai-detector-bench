"""Metric tests, including the ones that assert a metric *declines* to answer."""

from detbench.core import RefusalReason, Verdict
from detbench.metrics import (
    auroc,
    evaluate,
    overconfidence_rate,
    threshold_at_fpr,
    tpr_at_fpr,
)


def test_auroc_perfect_separation():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert auroc(labels, scores) == 1.0


def test_auroc_inverted_separation():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    assert auroc(labels, scores) == 0.0


def test_auroc_all_ties_is_chance():
    labels = [0, 0, 1, 1]
    scores = [0.5, 0.5, 0.5, 0.5]
    assert auroc(labels, scores) == 0.5


def test_auroc_needs_both_classes():
    assert auroc([1, 1, 1], [0.1, 0.2, 0.3]) is None


def test_tpr_at_fpr_refuses_when_sample_too_small():
    # Ten human documents cannot resolve a 0.1% false-positive rate: the smallest
    # non-zero rate observable is 10%. Returning a number here would be the exact
    # overclaim this project measures in others.
    labels = [0] * 10 + [1] * 10
    scores = [0.1 * i for i in range(20)]
    assert tpr_at_fpr(labels, scores, 0.001) is None
    assert threshold_at_fpr(labels, scores, 0.001) is None


def test_tpr_at_fpr_resolvable_with_enough_negatives():
    # 200 human docs can resolve 1% (two documents' worth).
    labels = [0] * 200 + [1] * 200
    scores = [0.0] * 200 + [1.0] * 200
    assert tpr_at_fpr(labels, scores, 0.01) == 1.0


def test_tpr_at_fpr_penalises_overlap():
    # Half the machine documents score inside the human range, so at a strict FPR
    # only the clearly separated half is caught.
    labels = [0] * 100 + [1] * 100
    scores = [0.0] * 100 + [0.0] * 50 + [1.0] * 50
    assert tpr_at_fpr(labels, scores, 0.01) == 0.5


def test_overconfidence_counts_only_human_documents():
    labels = [0, 0, 0, 0, 1, 1]
    probs = [0.99, 0.96, 0.10, 0.20, 0.99, 0.99]
    # Two of four human documents were called machine with p >= 0.95.
    assert overconfidence_rate(labels, probs) == 0.5


def test_overconfidence_is_none_without_probabilities():
    assert overconfidence_rate([0, 1], [None, None]) is None


def test_evaluate_excludes_refusals_from_rates_but_reports_them():
    labels = [0, 0, 1, 1]
    verdicts = [
        Verdict(detector="d", score=0.0),
        Verdict.refuse("d", RefusalReason.TOO_SHORT),
        Verdict(detector="d", score=1.0),
        Verdict.refuse("d", RefusalReason.TOO_SHORT),
    ]
    rep = evaluate("d", "clean", labels, verdicts)
    assert rep.n_total == 4
    assert rep.n_scored == 2
    assert rep.refusal == 0.5
    # Only one human and one machine document survived, so strict rates are unresolvable.
    assert rep.tpr_at_0_1pct is None
