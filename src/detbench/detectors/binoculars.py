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

# Published decision thresholds from the reference implementation, selected with the
# Falcon-7B pair at bfloat16. They are on the *native* scale (low = machine), so compare
# `meta["binoculars_raw"]` against them, not `Verdict.score`, which is negated. They are
# recorded rather than applied: a threshold chosen on one model pair and one corpus does
# not transfer, and `detbench` reports scores rather than verdicts.
REFERENCE_ACCURACY_THRESHOLD = 0.9015310749276843  # optimised for F1
REFERENCE_FPR_THRESHOLD = 0.8536432310785527  # optimised for low FPR, chosen at 0.01%


def _perplexity(logits, input_ids, attention_mask) -> float:
    """Mean token-level negative log-likelihood, masked to real tokens.

    Transcribed from the reference `metrics.perplexity`. Note whose logits arrive here:
    the caller passes the **performer's**.
    """
    import torch

    shifted_logits = logits[..., :-1, :].contiguous().float()
    shifted_labels = input_ids[..., 1:].contiguous()
    shifted_mask = attention_mask[..., 1:].contiguous()

    ce = torch.nn.functional.cross_entropy(
        shifted_logits.transpose(1, 2), shifted_labels, reduction="none"
    )
    return ((ce * shifted_mask).sum(1) / shifted_mask.sum(1)).item()


def _cross_entropy(p_logits, q_logits, input_ids, pad_token_id: int) -> float:
    """Mean cross-entropy of the performer's distribution against the observer's.

    Transcribed from the reference `metrics.entropy`, including two choices that look
    like oversights but must be matched for the numbers to agree:

    - **No shifting.** Unlike the perplexity term, this runs over every position of the
      full sequence.
    - **The mask is `input_ids != pad_token_id`**, not the attention mask. When the
      tokenizer has no pad token, `pad_token` is set to `eos_token`, so a genuine
      end-of-text token occurring inside the document is masked out of the denominator.
      That is the reference's behaviour, and deviating from it would mean this detector
      is no longer the published one.
    """
    import torch

    vocab_size = p_logits.shape[-1]
    total_tokens = q_logits.shape[-2]

    p_proba = torch.softmax(p_logits.float(), dim=-1).view(-1, vocab_size)
    q_scores = q_logits.float().view(-1, vocab_size)

    ce = torch.nn.functional.cross_entropy(
        input=q_scores, target=p_proba, reduction="none"
    ).view(-1, total_tokens)

    padding_mask = (input_ids != pad_token_id).type(torch.uint8)
    return ((ce * padding_mask).sum(1) / padding_mask.sum(1)).item()


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
        # Both models must share a vocabulary: the denominator compares the observer's
        # probability distribution against the performer's over the *same* index space,
        # so mismatched tokenizers produce a number that looks fine and means nothing.
        # The reference implementation asserts this; skipping it was a real hole.
        performer_tok = AutoTokenizer.from_pretrained(self.performer_name)
        if self._tokenizer.vocab != performer_tok.vocab:
            raise ValueError(
                f"tokenizers differ between {self.observer_name} and {self.performer_name}; "
                "Binoculars requires a shared vocabulary"
            )
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

        # Both quantities below follow the reference implementation exactly
        # (ahans30/Binoculars, `binoculars/metrics.py`). Two details are easy to get
        # wrong and were wrong here until validation caught them:
        #
        #   1. The numerator is the **performer's** perplexity, not the observer's. The
        #      observer appears only inside the denominator, as the distribution the
        #      performer is scored against.
        #   2. The denominator is computed on **unshifted** logits over the full
        #      sequence, while the numerator is shifted. They deliberately do not align.
        log_ppl = _perplexity(perf_logits, ids, enc.attention_mask)
        x_ppl = _cross_entropy(obs_logits, perf_logits, ids, self._tokenizer.pad_token_id)

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
