"""Binoculars — perplexity divided by cross-perplexity (Hans et al., ICML 2024).

The insight the method rests on: raw perplexity confuses "machine-written" with "written
in a predictable style", which is why perplexity-only detectors flag non-native English
writers and formulaic human prose. Binoculars divides a text's perplexity under an
*observer* model by the cross-perplexity between an observer and a *performer* model.
The denominator absorbs "this text is inherently predictable", leaving behind something
closer to "this text is predictable *to a model, in the way models are predictable*".

Reported at over 90% detection of ChatGPT output at a 0.01% false-positive rate, and it
beat commercial APIs at release. It is also, per arXiv:2605.14240, the method that
degrades *most* under paraphrase — the largest F1 drop of the seven tested. That pairing
is the single most important fact this benchmark exists to display.

Sign convention: Binoculars is natively **low = machine**. This module returns the
negated score so that, as everywhere else in `detbench`, higher means more machine-like.

**Not yet validated against the reference implementation.** Until `scripts/validate.py`
reproduces the published AUROC on a shared slice, treat numbers from this module as
indicative. That check is a release blocker for publishing any leaderboard row.
"""

from __future__ import annotations

from ..core import RefusalReason, Verdict, gate_length

# The pairing from the paper. Heavy — roughly 14GB of weights across both models and
# realistically a GPU. `SMALL_PAIR` exists so the harness is exercisable on a laptop;
# it is a different detector in practice and is labelled as such in any output.
DEFAULT_OBSERVER = "tiiuae/falcon-7b"
DEFAULT_PERFORMER = "tiiuae/falcon-7b-instruct"
SMALL_PAIR = ("gpt2", "gpt2-medium")


class BinocularsDetector:
    """Cross-perplexity ratio detector.

    Models are loaded lazily on first use so that importing `detbench` stays instant and
    the dependency-free parts of the package keep working on a machine with no torch.
    """

    def __init__(
        self,
        observer: str = DEFAULT_OBSERVER,
        performer: str = DEFAULT_PERFORMER,
        min_tokens: int = 50,
        device: str | None = None,
        max_length: int = 512,
    ):
        self.observer_name = observer
        self.performer_name = performer
        self.min_tokens = min_tokens
        self.device = device
        self.max_length = max_length
        self._loaded = False
        self._observer = None
        self._performer = None
        self._tokenizer = None
        # Name carries the model pair: a Binoculars run with gpt2 and a Binoculars run
        # with falcon are not comparable, and a leaderboard that labels both
        # "binoculars" would be quietly wrong.
        self.name = f"binoculars[{observer.split('/')[-1]}/{performer.split('/')[-1]}]"

    def _load(self) -> bool:
        if self._loaded:
            return self._observer is not None
        self._loaded = True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            return False

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(self.observer_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._observer = AutoModelForCausalLM.from_pretrained(
            self.observer_name, torch_dtype=dtype
        ).to(self.device).eval()
        self._performer = AutoModelForCausalLM.from_pretrained(
            self.performer_name, torch_dtype=dtype
        ).to(self.device).eval()
        return True

    def score_one(self, text: str) -> Verdict:
        refusal = gate_length(self.name, text, self.min_tokens)
        if refusal is not None:
            return refusal

        if not self._load():
            # Absence of a model is a refusal, not a zero. The harness records it and
            # the leaderboard shows the detector as unavailable rather than as failing.
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
            obs_logits = self._observer(**enc).logits
            perf_logits = self._performer(**enc).logits

        # Shift so position i predicts token i+1.
        obs = obs_logits[:, :-1, :].float()
        perf = perf_logits[:, :-1, :].float()
        targets = ids[:, 1:]

        obs_logprobs = torch.log_softmax(obs, dim=-1)

        # log-perplexity: the observer's negative log-likelihood of the actual tokens.
        nll = -obs_logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        log_ppl = nll.mean().item()

        # Cross-perplexity: the observer's *predicted distribution* scored against the
        # performer's. This is the denominator that cancels inherent predictability.
        perf_logprobs = torch.log_softmax(perf, dim=-1)
        obs_probs = obs_logprobs.exp()
        x_ppl = -(obs_probs * perf_logprobs).sum(dim=-1).mean().item()

        if x_ppl <= 0:
            return Verdict.refuse(
                self.name, RefusalReason.MODEL_UNAVAILABLE, detail="degenerate cross-perplexity"
            )

        raw = log_ppl / x_ppl
        return Verdict(
            detector=self.name,
            score=-raw,  # negate: native scale is low = machine
            p_machine=None,  # uncalibrated; fit a threshold on your own held-out data
            meta={
                "binoculars_raw": raw,
                "log_ppl": log_ppl,
                "x_ppl": x_ppl,
                "n_tokens": int(ids.shape[1]),
                "observer": self.observer_name,
                "performer": self.performer_name,
            },
        )
