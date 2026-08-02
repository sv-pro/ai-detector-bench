"""Fast-DetectGPT — conditional probability curvature, computed analytically.

DetectGPT's original insight was that machine text sits at a local maximum of the model's
log-probability surface: perturb it and the probability falls off more sharply than it
does for human text. The original paid for this with a hundred model calls per document
to build the perturbations.

Fast-DetectGPT gets the same signal in a single forward pass by observing that the
expectation and variance of the log-probability under the model's *own* conditional
distribution have closed forms. No sampling, no perturbation model:

    d(x) = ( log p(x) - E_q[log p(x̃)] ) / sqrt( Var_q[log p(x̃)] )

with q = p in the white-box setting used here. Higher d means more machine-like, which
already matches this package's convention, so no sign flip is needed.

Included alongside Binoculars because the two fail differently: Binoculars needs two
models and is the more fragile under paraphrase, while curvature needs one model and
degrades more gently. A benchmark with only one of them would hide that trade-off.

**Not yet validated against the reference implementation** — same release blocker noted
in `binoculars.py`.
"""

from __future__ import annotations

from ..core import RefusalReason, Verdict, gate_length

DEFAULT_MODEL = "gpt2-medium"


class FastDetectGPTDetector:
    """Single-model curvature detector, analytic form."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        min_tokens: int = 50,
        device: str | None = None,
        max_length: int = 512,
    ):
        self.model_name = model
        self.min_tokens = min_tokens
        self.device = device
        self.max_length = max_length
        self._loaded = False
        self._model = None
        self._tokenizer = None
        self.name = f"fast_detectgpt[{model.split('/')[-1]}]"

    def _load(self) -> bool:
        if self._loaded:
            return self._model is not None
        self._loaded = True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            return False

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype
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
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        ids = enc.input_ids
        if ids.shape[1] < 2:
            return Verdict.refuse(self.name, RefusalReason.TOO_SHORT, n_tokens=int(ids.shape[1]))

        with torch.no_grad():
            logits = self._model(**enc).logits

        logits = logits[:, :-1, :].float()
        targets = ids[:, 1:]

        logprobs = torch.log_softmax(logits, dim=-1)
        probs = logprobs.exp()

        # Observed log-likelihood of the tokens that actually occur.
        observed = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        # Closed-form mean and variance of log p over the model's own conditional
        # distribution — this is what replaces DetectGPT's sampling loop.
        mean = (probs * logprobs).sum(dim=-1)
        second = (probs * logprobs.pow(2)).sum(dim=-1)
        var = (second - mean.pow(2)).clamp(min=1e-12)

        numerator = (observed - mean).sum().item()
        denominator = var.sum().sqrt().item()
        if denominator <= 0:
            return Verdict.refuse(
                self.name, RefusalReason.MODEL_UNAVAILABLE, detail="degenerate variance"
            )

        d = numerator / denominator
        return Verdict(
            detector=self.name,
            score=d,
            p_machine=None,
            meta={
                "curvature": d,
                "n_tokens": int(ids.shape[1]),
                "model": self.model_name,
            },
        )
