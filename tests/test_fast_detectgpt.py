"""Fast-DetectGPT tests — the criterion pinned by independent hand-derivation.

`scripts/validate_fast_detectgpt.py` proves equivalence against the reference and, more
importantly, against the reference's *Monte-Carlo* criterion, which is what actually tests
the closed-form derivation rather than the faithfulness of a transcription.

These run offline. The expected values below are derived from first principles in plain
Python rather than by calling the implementation, so they would catch a shared error that a
transcription check cannot.
"""

import math

import pytest

torch = pytest.importorskip("torch", reason="model-bearing detectors need detbench[torch]")

from detbench.detectors.fast_detectgpt import (  # noqa: E402
    REFERENCE_DISTRIB_PARAMS,
    FastDetectGPTDetector,
    probability_from_curvature,
    sampling_discrepancy_analytic,
)


def _expected_discrepancy(logit_rows, labels):
    """Independent derivation of d(x), in plain Python.

    d = ( Σ_t log p(x_t) − Σ_t E_q[log p] ) / sqrt( Σ_t Var_q[log p] )
    """
    total_ll = 0.0
    total_mean = 0.0
    total_var = 0.0
    for row, label in zip(logit_rows, labels):
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        z = sum(exps)
        probs = [e / z for e in exps]
        lprobs = [math.log(p) for p in probs]
        total_ll += lprobs[label]
        mean = sum(p * lp for p, lp in zip(probs, lprobs))
        second = sum(p * lp * lp for p, lp in zip(probs, lprobs))
        total_mean += mean
        total_var += second - mean**2
    return (total_ll - total_mean) / math.sqrt(total_var)


def test_criterion_matches_an_independent_derivation():
    rows = [[0.0, math.log(3.0)], [2.0, -1.0], [0.5, 0.25]]
    labels = [0, 1, 0]
    logits = torch.tensor([rows], dtype=torch.float64)
    lab = torch.tensor([labels])
    got = sampling_discrepancy_analytic(logits, logits, lab)
    assert got == pytest.approx(_expected_discrepancy(rows, labels), rel=1e-9)


def test_a_confidently_predicted_token_raises_the_curvature():
    # Machine text is text the model finds unusually likely relative to its own
    # alternatives, which is exactly what a higher d means.
    rows_likely = [[6.0, 0.0], [6.0, 0.0], [6.0, 0.0]]
    rows_unlikely = [[0.0, 6.0], [0.0, 6.0], [0.0, 6.0]]
    lab = torch.tensor([[0, 0, 0]])
    d_likely = sampling_discrepancy_analytic(
        torch.tensor([rows_likely]), torch.tensor([rows_likely]), lab
    )
    d_unlikely = sampling_discrepancy_analytic(
        torch.tensor([rows_unlikely]), torch.tensor([rows_unlikely]), lab
    )
    assert d_likely > d_unlikely


def test_uniform_logits_are_degenerate_and_produce_a_non_finite_value():
    # Every position has zero variance, so the denominator is zero. The reference does not
    # guard this; we surface it as a refusal at the detector level rather than clamping the
    # per-position variance, which would silently change the statistic.
    logits = torch.zeros(1, 3, 4)
    d = sampling_discrepancy_analytic(logits, logits, torch.tensor([[0, 1, 2]]))
    assert not math.isfinite(d)


def test_per_position_variance_is_not_clamped():
    # A previous version clamped each position's variance to 1e-12, which changed the sum
    # whenever floating-point error produced a small negative term. The clamped form would
    # give sqrt(3e-12) here instead of a non-finite result.
    logits = torch.zeros(1, 3, 4)
    d = sampling_discrepancy_analytic(logits, logits, torch.tensor([[0, 1, 2]]))
    assert not math.isfinite(d)


def test_mismatched_vocabularies_truncate_to_the_shared_prefix():
    # Reference behaviour, exercised because the two-model path can pair models whose
    # vocabularies differ in size.
    ref = torch.randn(1, 2, 5, dtype=torch.float64)
    score = torch.randn(1, 2, 3, dtype=torch.float64)
    lab = torch.tensor([[0, 1]])
    got = sampling_discrepancy_analytic(ref, score, lab)
    truncated = sampling_discrepancy_analytic(ref[:, :, :3], score, lab)
    assert got == pytest.approx(truncated, rel=1e-12)


def test_probability_mapping_matches_the_reference_formula():
    p = REFERENCE_DISTRIB_PARAMS["falcon-7b_falcon-7b-instruct"]

    def pdf(x, mu, sigma):
        return math.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))

    for d in (-2.0, 0.0, 1.5, 3.0):
        expected = pdf(d, p["mu1"], p["sigma1"]) / (
            pdf(d, p["mu0"], p["sigma0"]) + pdf(d, p["mu1"], p["sigma1"])
        )
        assert probability_from_curvature(d, p) == pytest.approx(expected, rel=1e-12)


def test_probability_rises_with_curvature_across_the_decision_region():
    p = REFERENCE_DISTRIB_PARAMS["falcon-7b_falcon-7b-instruct"]
    values = [probability_from_curvature(d, p) for d in (0.0, 1.0, 2.0, 3.0, 4.0)]
    assert values == sorted(values)
    assert all(0.0 <= v <= 1.0 for v in values)


def test_reference_distribution_params_recorded_verbatim():
    # Copied from the reference local_infer.py. Applied only to their exact model pair.
    assert REFERENCE_DISTRIB_PARAMS["gpt-j-6B_gpt-neo-2.7B"]["mu0"] == 0.2713
    assert REFERENCE_DISTRIB_PARAMS["falcon-7b_falcon-7b-instruct"]["mu1"] == 2.9306
    for params in REFERENCE_DISTRIB_PARAMS.values():
        # Machine text sits at higher curvature with wider spread, by construction.
        assert params["mu1"] > params["mu0"]
        assert params["sigma1"] > params["sigma0"]


def test_the_model_pair_is_part_of_the_detector_identity():
    # Two runs with different pairs are different detectors and must not share a row.
    same = FastDetectGPTDetector(scoring_model="gpt2-medium")
    pair = FastDetectGPTDetector(scoring_model="gpt2-medium", sampling_model="gpt2")
    assert same.name == "fast_detectgpt[gpt2-medium]"
    assert pair.name == "fast_detectgpt[gpt2->gpt2-medium]"
    assert same.name != pair.name


def test_uncalibrated_pairs_get_no_probability_key():
    # gpt2/gpt2-medium is not one of the pairs the reference fitted, so no probability.
    det = FastDetectGPTDetector(scoring_model="gpt2-medium", sampling_model="gpt2")
    assert det._params_key not in REFERENCE_DISTRIB_PARAMS
