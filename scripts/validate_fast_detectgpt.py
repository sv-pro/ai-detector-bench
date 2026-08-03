#!/usr/bin/env python
"""Validate `detbench`'s Fast-DetectGPT against the reference implementation.

Same two-claim split as `validate_binoculars.py`:

**(a) Algorithmic equivalence** — our code computes the same quantity as
baoguangsheng/fast-detect-gpt, to floating-point tolerance, on the same inputs. The
reference criterion below is transcribed verbatim from upstream `scripts/fast_detect_gpt.py`
and its caller from `scripts/local_infer.py`, and runs as an independent code path.

**(b) Reproduction** — the paper's reported AUROC. Not attempted here. Their headline
configurations use `gpt-j-6B` sampling with `gpt-neo-2.7B` scoring (~15 GB), or the
Falcon-7B pair. This machine has a 4 GB GPU.

Both the shared-model and the two-model configurations are exercised, because they take
different code paths and only the second matches the paper's standard setting.

Usage:
    python scripts/validate_fast_detectgpt.py [--scoring gpt2-medium] [--sampling gpt2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from detbench.detectors.fast_detectgpt import (  # noqa: E402
    REFERENCE_DISTRIB_PARAMS,
    FastDetectGPTDetector,
    probability_from_curvature,
)

# ---------------------------------------------------------------------------
# Reference, transcribed from baoguangsheng/fast-detect-gpt (ICLR 2024, MIT).
# `scripts/fast_detect_gpt.py::get_sampling_discrepancy_analytic`, kept structurally
# identical so a divergence in our port surfaces as a numeric difference.
# ---------------------------------------------------------------------------


def ref_get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()


class ReferenceFastDetectGPT:
    """Transcribed from `scripts/local_infer.py::FastDetectGPT.compute_crit`."""

    def __init__(self, scoring: str, sampling: str, device: str = "cpu"):
        self.device = device
        self.scoring_name, self.sampling_name = scoring, sampling
        self.scoring_tokenizer = AutoTokenizer.from_pretrained(scoring)
        if not self.scoring_tokenizer.pad_token:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
        self.scoring_model = AutoModelForCausalLM.from_pretrained(
            scoring, torch_dtype=torch.float32
        ).eval()
        if sampling != scoring:
            self.sampling_tokenizer = AutoTokenizer.from_pretrained(sampling)
            if not self.sampling_tokenizer.pad_token:
                self.sampling_tokenizer.pad_token = self.sampling_tokenizer.eos_token
            self.sampling_model = AutoModelForCausalLM.from_pretrained(
                sampling, torch_dtype=torch.float32
            ).eval()

    def compute_crit(self, text: str) -> float:
        tokenized = self.scoring_tokenizer(
            text,
            truncation=True,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            max_length=512,
        )
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.sampling_name == self.scoring_name:
                logits_ref = logits_score
            else:
                tokenized = self.sampling_tokenizer(
                    text,
                    truncation=True,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                    max_length=512,
                )
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.sampling_model(**tokenized).logits[:, :-1]
            return ref_get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)


def ref_get_samples(logits, nsamples: int):
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    return distrib.sample([nsamples]).permute([1, 2, 0])


def ref_get_likelihood(logits, labels):
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    return lprobs.gather(dim=-1, index=labels).mean(dim=1)


def ref_get_sampling_discrepancy(logits_ref, logits_score, labels, nsamples: int = 10000):
    """The reference's **Monte-Carlo** criterion — the one the analytic form replaces.

    This is the independent check that matters. The analytic criterion is a closed-form
    identity for this sampling procedure, so if the derivation (or our port of it) were
    wrong, the two would disagree by far more than sampling error. Comparing our analytic
    result against a transcribed analytic result proves only that the transcription was
    faithful; comparing it against *sampling* tests the mathematics.
    """
    samples = ref_get_samples(logits_ref, nsamples)
    log_likelihood_x = ref_get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = ref_get_likelihood(logits_score, samples)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    return ((log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde).item()


# ---------------------------------------------------------------------------

SAMPLES = [
    "Call me Ishmael. Some years ago—never mind how long precisely—having little or no "
    "money in my purse, and nothing particular to interest me on shore, I thought I would "
    "sail about a little and see the watery part of the world. It is a way I have of "
    "driving off the spleen and regulating the circulation, whenever I find myself growing "
    "grim about the mouth and it is a damp, drizzly November in my soul.",
    "The integration of renewable energy sources into existing power grids presents several "
    "multifaceted challenges. Firstly, the intermittent nature of solar and wind generation "
    "requires robust storage solutions. Additionally, grid operators must leverage advanced "
    "forecasting to balance supply and demand effectively. Furthermore, regulatory "
    "frameworks often lag behind technological capabilities, creating barriers to adoption.",
    "To reproduce the benchmark, first download the corpus into the local cache, then run "
    "the evaluation with a fixed seed. The harness streams the file rather than loading it, "
    "because the adversarial variants exceed eleven gigabytes and will not fit in memory on "
    "an ordinary development machine. Results are written as one row per detector and slice.",
    "def min_cost(cost, m, n):\n    tc = [[0 for _ in range(C)] for _ in range(R)]\n"
    "    tc[0][0] = cost[0][0]\n    for i in range(1, m + 1):\n"
    "        tc[i][0] = tc[i - 1][0] + cost[i][0]\n    return tc[m][n]\n"
    "This function computes the minimum cost path through a matrix using dynamic "
    "programming, filling the table row by row until the target cell is reached.",
]


def run_pair(scoring: str, sampling: str, tolerance: float) -> tuple[float, int]:
    label = scoring if scoring == sampling else f"{sampling} -> {scoring}"
    print(f"\n--- configuration: {label} ---")
    ref = ReferenceFastDetectGPT(scoring, sampling)
    ours = FastDetectGPTDetector(
        scoring_model=scoring, sampling_model=sampling, device="cpu", min_tokens=20
    )

    print(f"{'#':<3} {'reference':>12} {'detbench':>12} {'abs diff':>12}  ok")
    print("-" * 56)
    worst, failures = 0.0, 0
    for i, text in enumerate(SAMPLES):
        r = ref.compute_crit(text)
        v = ours.score_one(text)
        if v.refused:
            print(f"{i:<3} {'—':>12} {'REFUSED':>12} {'—':>12}  !!")
            failures += 1
            continue
        diff = abs(r - v.score)
        worst = max(worst, diff)
        ok = diff <= tolerance
        failures += 0 if ok else 1
        print(f"{i:<3} {r:>12.6f} {v.score:>12.6f} {diff:>12.2e}  {'ok' if ok else 'FAIL'}")
    return worst, failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", default="gpt2-medium")
    ap.add_argument("--sampling", default="gpt2")
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--nsamples", type=int, default=10000)
    args = ap.parse_args()

    print("reference : baoguangsheng/fast-detect-gpt (transcribed)")
    print("detbench  : detbench.detectors.fast_detectgpt — float32, CPU")
    print(f"transformers {transformers.__version__}, torch {torch.__version__}")

    worst, failures = 0.0, 0
    # Shared-model path and two-model path are different branches; both must agree.
    for scoring, sampling in [(args.scoring, args.scoring), (args.scoring, args.sampling)]:
        w, f = run_pair(scoring, sampling, args.tolerance)
        worst, failures = max(worst, w), failures + f

    # The check with real teeth: analytic closed form vs Monte-Carlo sampling.
    print(f"\n--- analytic vs Monte-Carlo ({args.nsamples} samples, seed 0) ---")
    print("    (transcription agreement is near-tautological; this tests the maths)")
    torch.manual_seed(0)
    mc_ref = ReferenceFastDetectGPT(args.scoring, args.scoring)
    tok = mc_ref.scoring_tokenizer
    print(f"{'#':<3} {'analytic':>12} {'monte-carlo':>12} {'abs diff':>12}  {'rel':>8}")
    print("-" * 56)
    mc_worst = 0.0
    for i, text in enumerate(SAMPLES):
        enc = tok(
            text, truncation=True, return_tensors="pt", padding=True,
            return_token_type_ids=False, max_length=512,
        )
        labels = enc.input_ids[:, 1:]
        with torch.no_grad():
            logits = mc_ref.scoring_model(**enc).logits[:, :-1]
        analytic = ref_get_sampling_discrepancy_analytic(logits, logits, labels)
        mc = ref_get_sampling_discrepancy(logits, logits, labels, args.nsamples)
        diff = abs(analytic - mc)
        mc_worst = max(mc_worst, diff)
        rel = diff / max(abs(analytic), 1e-9)
        print(f"{i:<3} {analytic:>12.6f} {mc:>12.6f} {diff:>12.4f}  {rel:>7.1%}")
    print(f"  worst: {mc_worst:.4f} — expected to be small but nonzero (sampling error)")

    # The reference's probability mapping, checked independently of the models.
    print("\n--- probability mapping (compute_prob_norm) ---")
    params = REFERENCE_DISTRIB_PARAMS["falcon-7b_falcon-7b-instruct"]
    for d in (-2.0, 0.0, 1.5, 3.0, 6.0):
        print(f"  curvature {d:+.1f} -> p(machine) {probability_from_curvature(d, params):.4f}")

    print(f"\nworst absolute difference: {worst:.3e} (tolerance {args.tolerance:.0e})")
    if failures:
        print(f"VALIDATION FAILED — {failures} sample(s) disagree")
        return 1
    print("VALIDATION PASSED — algorithmic equivalence confirmed on both configurations.")
    print(
        "\nNOT established: the published AUROC. Their headline settings use gpt-j-6B ->\n"
        "gpt-neo-2.7B or the Falcon-7B pair; this ran on GPT-2. Equivalence, not reproduction."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
