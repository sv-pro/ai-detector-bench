#!/usr/bin/env python
"""Validate `detbench`'s Binoculars against the reference implementation.

**What this establishes and what it does not.**

It establishes *algorithmic equivalence*: that `detbench.detectors.binoculars` computes the
same quantity as ahans30/Binoculars, to floating-point tolerance, on the same inputs. The
reference computation below is transcribed verbatim from the upstream `binoculars/metrics.py`
and `binoculars/detector.py` and runs as an independent code path, so a shared bug would
have to be transcribed twice to survive.

It does **not** reproduce the published AUROC or the >90% TPR at 0.01% FPR. Those were
measured with the Falcon-7B / Falcon-7B-Instruct pair at bfloat16 — roughly 28 GB of
weights — and the machine this was written on has a 4 GB GPU. Reproducing the published
numbers is a separate, hardware-gated step and is still open.

So: the implementation is checked, the paper's results are not re-derived. Those are
different claims and the distinction is the whole reason this script exists.

Usage:
    python scripts/validate_binoculars.py [--observer gpt2] [--performer gpt2-medium]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from detbench.detectors.binoculars import BinocularsDetector  # noqa: E402

# ---------------------------------------------------------------------------
# Reference implementation, transcribed from ahans30/Binoculars (ICML 2024).
# Kept structurally identical to upstream — same loss objects, same masking, same
# ordering — so that a divergence in our port shows up as a numeric difference rather
# than being absorbed by a tidier rewrite.
# ---------------------------------------------------------------------------

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)


def ref_perplexity(encoding, logits, temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()
    ppl = (
        ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) * shifted_attention_mask
    ).sum(1) / shifted_attention_mask.sum(1)
    return ppl.to("cpu").float().numpy()


def ref_entropy(p_logits, q_logits, encoding, pad_token_id, temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature
    p_proba = softmax_fn(p_scores).view(-1, vocab_size)
    q_scores = q_scores.view(-1, vocab_size)
    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)
    return (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())


class ReferenceBinoculars:
    def __init__(self, observer: str, performer: str, max_token_observed: int = 512):
        vocab_1 = AutoTokenizer.from_pretrained(observer).vocab
        vocab_2 = AutoTokenizer.from_pretrained(performer).vocab
        if vocab_1 != vocab_2:
            raise ValueError(f"Tokenizers are not identical for {observer} and {performer}.")
        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer, torch_dtype=torch.float32
        ).eval()
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer, torch_dtype=torch.float32
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(observer)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    @torch.inference_mode()
    def compute_score(self, input_text: str) -> float:
        encodings = self.tokenizer(
            [input_text],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        )
        observer_logits = self.observer_model(**encodings).logits
        performer_logits = self.performer_model(**encodings).logits
        ppl = ref_perplexity(encodings, performer_logits)
        x_ppl = ref_entropy(
            observer_logits, performer_logits, encodings, self.tokenizer.pad_token_id
        )
        return float((ppl / x_ppl)[0])


# ---------------------------------------------------------------------------

SAMPLES = [
    # Deliberately varied: 19th-century prose, model-shaped prose, technical writing,
    # and a document containing code. Agreement must not depend on genre.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--observer", default="gpt2")
    ap.add_argument("--performer", default="gpt2-medium")
    ap.add_argument("--tolerance", type=float, default=1e-4)
    args = ap.parse_args()

    print(f"reference : ahans30/Binoculars (transcribed) — {args.observer} / {args.performer}")
    print(f"detbench  : detbench.detectors.binoculars — same pair, float32, CPU")
    print(f"transformers {transformers.__version__}, torch {torch.__version__}\n")

    ref = ReferenceBinoculars(args.observer, args.performer)
    ours = BinocularsDetector(
        observer=args.observer, performer=args.performer, device="cpu", min_tokens=20
    )

    print(f"{'#':<3} {'reference':>12} {'detbench':>12} {'abs diff':>12}  ok")
    print("-" * 56)
    worst = 0.0
    failures = 0
    for i, text in enumerate(SAMPLES):
        r = ref.compute_score(text)
        v = ours.score_one(text)
        if v.refused:
            print(f"{i:<3} {'—':>12} {'REFUSED':>12} {'—':>12}  !!")
            failures += 1
            continue
        # `Verdict.score` is negated so that higher means more machine-like everywhere in
        # this package; the native Binoculars scale is the inverse. Compare on the native
        # value that `meta` preserves.
        mine = v.meta["binoculars_raw"]
        diff = abs(r - mine)
        worst = max(worst, diff)
        ok = diff <= args.tolerance
        failures += 0 if ok else 1
        print(f"{i:<3} {r:>12.6f} {mine:>12.6f} {diff:>12.2e}  {'ok' if ok else 'FAIL'}")

    print(f"\nworst absolute difference: {worst:.3e} (tolerance {args.tolerance:.0e})")
    if failures:
        print(f"VALIDATION FAILED — {failures} sample(s) disagree")
        return 1
    print("VALIDATION PASSED — algorithmic equivalence confirmed on this model pair.")
    print(
        "\nNOT established: the published AUROC / >90% TPR at 0.01% FPR, which require the\n"
        "Falcon-7B pair at bfloat16 (~28 GB). That reproduction remains open, and no\n"
        "leaderboard row may cite the paper's numbers until it is done."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
