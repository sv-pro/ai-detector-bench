"""Fast-DetectGPT — conditional probability curvature, computed analytically.

DetectGPT's original insight was that machine text sits at a local maximum of the model's
log-probability surface: perturb it and the probability falls off more sharply than it does
for human text. The original paid for that with ~100 model calls per document to build the
perturbations.

Fast-DetectGPT (Bao et al., ICLR 2024) gets the same signal in a single forward pass, by
observing that the expectation and variance of the log-probability under the model's own
conditional distribution have closed forms:

    d(x) = ( log p(x) - E_q[log p(x̃)] ) / sqrt( Var_q[log p(x̃)] )

Higher d means more machine-like, which already matches this package's convention.

**Two models, not one.** The reference separates the *sampling* model (which supplies the
distribution the expectation is taken over) from the *scoring* model (which supplies the
log-probabilities). They may differ, and the paper's headline configurations mostly do —
`gpt-j-6B` sampling with `gpt-neo-2.7B` scoring is their standard black-box setting. An
earlier version of this file supported only the shared-model case and described it as
"the white-box setting used here", which was true but meant the detector could not be run
in the configuration the published numbers came from.

Validated against the reference implementation — see `scripts/validate_fast_detectgpt.py`
and `docs/METHODOLOGY.md` § 6.
"""

from __future__ import annotations

import math

from ..core import RefusalReason, Verdict, gate_length

DEFAULT_MODEL = "gpt2-medium"

# Normal-distribution parameters fitted by the reference authors on dev samples, keyed by
# `{sampling_model}_{scoring_model}`. They convert a raw curvature into a probability via
#     p(machine|x) = pdf(x; mu1, sigma1) / (pdf(x; mu0, sigma0) + pdf(x; mu1, sigma1))
# assuming balanced classes. Recorded verbatim from the reference `local_infer.py`.
#
# These apply ONLY to their exact model pair. `score_one` reports a probability when the
# configured pair has an entry here and `p_machine = None` otherwise — which is the same
# calibration discipline the rest of this package follows, now satisfied by someone else's
# fitted constants rather than ours.
REFERENCE_DISTRIB_PARAMS: dict[str, dict[str, float]] = {
    "gpt-j-6B_gpt-neo-2.7B": {"mu0": 0.2713, "sigma0": 0.9366, "mu1": 2.2334, "sigma1": 1.8731},
    "gpt-neo-2.7B_gpt-neo-2.7B": {"mu0": -0.2489, "sigma0": 0.9968, "mu1": 1.8983, "sigma1": 1.9935},
    "falcon-7b_falcon-7b-instruct": {"mu0": -0.0707, "sigma0": 0.9520, "mu1": 2.9306, "sigma1": 1.9039},
    "llama3-8b_llama3-8b-instruct": {"mu0": 0.1603, "sigma0": 1.0791, "mu1": 2.4686, "sigma1": 2.1582},
}


def _normal_pdf(x: float, mu: float, sigma: float) -> float:
    return math.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))


def probability_from_curvature(d: float, params: dict[str, float]) -> float:
    """Reference `compute_prob_norm`: two normals, balanced-class posterior."""
    p0 = _normal_pdf(d, params["mu0"], params["sigma0"])
    p1 = _normal_pdf(d, params["mu1"], params["sigma1"])
    total = p0 + p1
    return 0.5 if total == 0 else p1 / total


def sampling_discrepancy_analytic(logits_ref, logits_score, labels) -> float:
    """Transcribed from the reference `get_sampling_discrepancy_analytic`.

    `logits_ref` and `logits_score` must already be shifted by the caller
    (`logits[:, :-1]`), and `labels` must be `input_ids[:, 1:]` — the reference does that
    alignment in its caller, not here, and matching the split matters because the two
    halves must stay consistent.

    Note there is deliberately **no clamping** of the per-position variance. A previous
    version clamped it to 1e-12, which silently changed the sum whenever floating-point
    error produced a small negative term. Guarding a degenerate *total* is the caller's
    job; altering individual terms makes this a different statistic.
    """
    import torch

    if logits_ref.size(-1) != logits_score.size(-1):
        # Reference behaviour: truncate to the shared prefix of the two vocabularies.
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
    return discrepancy.mean().item()


class FastDetectGPTDetector:
    """Curvature detector, analytic form, with optional distinct sampling/scoring models."""

    def __init__(
        self,
        scoring_model: str = DEFAULT_MODEL,
        sampling_model: str | None = None,
        min_tokens: int = 50,
        device: str | None = None,
        max_length: int = 512,
        dtype: str = "float32",
    ):
        # float32 by default so CPU and GPU runs agree and both match the configuration
        # `scripts/validate_fast_detectgpt.py` checked. See the note in `binoculars.py`.
        self.dtype = dtype
        self.scoring_model_name = scoring_model
        # Defaults to the shared-model case, which is a legitimate configuration and the
        # cheapest one; it is simply not the configuration the paper leads with.
        self.sampling_model_name = sampling_model or scoring_model
        self.min_tokens = min_tokens
        self.device = device
        self.max_length = max_length
        self._loaded = False
        self._scoring = None
        self._sampling = None
        self._tokenizer = None

        short_s = self.scoring_model_name.split("/")[-1]
        short_r = self.sampling_model_name.split("/")[-1]
        # The pair is part of the identity: two runs with different pairs are different
        # detectors and must not share a leaderboard row.
        self.name = (
            f"fast_detectgpt[{short_s}]"
            if short_s == short_r
            else f"fast_detectgpt[{short_r}->{short_s}]"
        )
        self._params_key = f"{short_r}_{short_s}"

    def _load(self) -> bool:
        if self._loaded:
            return self._scoring is not None
        self._loaded = True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            return False

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = getattr(torch, self.dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(self.scoring_model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._scoring = AutoModelForCausalLM.from_pretrained(
            self.scoring_model_name, torch_dtype=dtype
        ).to(self.device).eval()
        if self.sampling_model_name == self.scoring_model_name:
            self._sampling = self._scoring
        else:
            self._sampling = AutoModelForCausalLM.from_pretrained(
                self.sampling_model_name, torch_dtype=dtype
            ).to(self.device).eval()
        return True

    def score_one(self, text: str) -> Verdict:
        refusal = gate_length(self.name, text, self.min_tokens)
        if refusal is not None:
            return refusal

        if not self._load():
            return Verdict.refuse(
                self.name,
                RefusalReason.MODEL_UNAVAILABLE,
                detail="install detbench[torch]",
            )

        import torch

        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
        ).to(self.device)
        ids = enc.input_ids
        if ids.shape[1] < 2:
            return Verdict.refuse(self.name, RefusalReason.TOO_SHORT, n_tokens=int(ids.shape[1]))

        labels = ids[:, 1:]
        with torch.no_grad():
            logits_score = self._scoring(**enc).logits[:, :-1].float()
            if self._sampling is self._scoring:
                logits_ref = logits_score
            else:
                # The reference asserts the two tokenizations agree rather than assuming
                # it; a mismatch here misaligns labels against logits silently.
                ref_enc = self._tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    return_token_type_ids=False,
                ).to(self.device)
                if not torch.all(ref_enc.input_ids[:, 1:] == labels):
                    return Verdict.refuse(
                        self.name, RefusalReason.MODEL_UNAVAILABLE, detail="tokenizer mismatch"
                    )
                logits_ref = self._sampling(**ref_enc).logits[:, :-1].float()

        d = sampling_discrepancy_analytic(logits_ref, logits_score, labels)
        if not math.isfinite(d):
            return Verdict.refuse(
                self.name, RefusalReason.MODEL_UNAVAILABLE, detail="degenerate variance"
            )

        params = REFERENCE_DISTRIB_PARAMS.get(self._params_key)
        return Verdict(
            detector=self.name,
            score=d,
            # A probability only where the reference published fitted parameters for this
            # exact pair. Every other pairing reports a raw score and no probability.
            p_machine=probability_from_curvature(d, params) if params else None,
            meta={
                "curvature": d,
                "n_tokens": int(ids.shape[1]),
                "scoring_model": self.scoring_model_name,
                "sampling_model": self.sampling_model_name,
                **(
                    {"fitted_on": f"reference dev samples ({self._params_key})"}
                    if params
                    else {}
                ),
            },
        )
