"""A stylometric baseline that loads no model and runs anywhere.

Why a weak method earns a place in a benchmark of strong ones: under paraphrase attack
the published ordering inverts. Binoculars leads on clean text and loses 0.196 F1 when
paraphrased; a plain text-feature model starts far lower and loses 0.053
(arXiv:2605.14240). The strongest detector is the most fragile one, and you only see that
if the fragile-but-strong and the weak-but-stable methods sit in the same table.

It is also the honesty control for this repository. It has no model, no training data
requirement, and its features are twelve numbers you can read off the page. If a
sophisticated detector cannot clearly beat it on a given slice, that slice is telling you
something about the slice.

**Uncalibrated by default.** Unfitted, it returns a raw score and `p_machine=None`,
because a hand-weighted feature combination is not a probability and printing one would
be the exact dishonesty this project exists to measure. Call `fit` with labelled data and
it will produce calibrated probabilities — valid only for that distribution.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ..core import Verdict, gate_length

SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s|$)")
WORD = re.compile(r"[A-Za-z']+")

# Discourse markers that appear at conspicuously higher rates in instruction-tuned model
# output. This list is descriptive of published observations, not a claim that any single
# word indicates authorship — a human writer who likes "moreover" is not evidence of
# anything, which is precisely why no single feature is allowed to drive the score.
LLM_MARKERS = {
    "moreover", "furthermore", "additionally", "consequently", "notably",
    "crucially", "importantly", "delve", "underscore", "underscores",
    "leverage", "robust", "nuanced", "multifaceted", "realm", "tapestry",
    "pivotal", "intricate", "comprehensive", "holistic",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "as", "by", "from", "is", "are", "was", "were", "be", "been",
    "it", "this", "that", "these", "those", "which", "who", "not", "no", "so",
}

FEATURE_NAMES = [
    "burstiness",
    "mean_sentence_len",
    "type_token_ratio",
    "hapax_ratio",
    "comma_rate",
    "semicolon_rate",
    "emdash_rate",
    "punct_diversity",
    "llm_marker_rate",
    "stopword_ratio",
    "mean_word_len",
    "paragraph_uniformity",
]


def extract_features(text: str) -> dict[str, float]:
    """Twelve interpretable surface statistics. No model, no training data.

    Every value is a rate or a ratio so that documents of different lengths remain
    comparable; length itself is handled by the refusal gate, not by a feature.
    """
    words = WORD.findall(text)
    n_words = max(len(words), 1)
    lower = [w.lower() for w in words]

    sentences = [s for s in SENTENCE_SPLIT.split(text) if s.strip()]
    sent_lens = [len(WORD.findall(s)) for s in sentences] or [n_words]
    mean_len = sum(sent_lens) / len(sent_lens)
    var = sum((x - mean_len) ** 2 for x in sent_lens) / len(sent_lens)
    # Burstiness: human prose varies sentence length far more than model prose.
    # Normalised by the mean so it does not simply re-encode sentence length.
    burstiness = (math.sqrt(var) / mean_len) if mean_len > 0 else 0.0

    counts: dict[str, int] = {}
    for w in lower:
        counts[w] = counts.get(w, 0) + 1
    hapax = sum(1 for c in counts.values() if c == 1)

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        plens = [len(WORD.findall(p)) for p in paragraphs]
        pmean = sum(plens) / len(plens)
        pvar = sum((x - pmean) ** 2 for x in plens) / len(plens)
        # High uniformity = suspiciously even paragraphs.
        paragraph_uniformity = 1.0 / (1.0 + math.sqrt(pvar) / pmean) if pmean > 0 else 0.0
    else:
        paragraph_uniformity = 0.0

    punct = [c for c in text if c in ".,;:!?—–-()\"'"]

    return {
        "burstiness": burstiness,
        "mean_sentence_len": mean_len,
        "type_token_ratio": len(counts) / n_words,
        "hapax_ratio": hapax / n_words,
        "comma_rate": text.count(",") / n_words,
        "semicolon_rate": text.count(";") / n_words,
        "emdash_rate": (text.count("—") + text.count("–")) / n_words,
        "punct_diversity": len(set(punct)) / 12.0,
        "llm_marker_rate": sum(1 for w in lower if w in LLM_MARKERS) / n_words,
        "stopword_ratio": sum(1 for w in lower if w in STOPWORDS) / n_words,
        "mean_word_len": sum(len(w) for w in words) / n_words,
        "paragraph_uniformity": paragraph_uniformity,
    }


# Direction each feature pushes when unfitted: +1 means "higher value looks more
# machine-generated". These signs come from the published direction of each effect, and
# the magnitudes are deliberately all 1.0 — inventing precise weights without fitting
# them would be making up numbers.
HEURISTIC_SIGNS: dict[str, float] = {
    "burstiness": -1.0,
    "mean_sentence_len": 0.0,
    "type_token_ratio": 0.0,
    "hapax_ratio": -1.0,
    "comma_rate": 0.0,
    "semicolon_rate": 0.0,
    "emdash_rate": 1.0,
    "punct_diversity": -1.0,
    "llm_marker_rate": 1.0,
    "stopword_ratio": 0.0,
    "mean_word_len": 0.0,
    "paragraph_uniformity": 1.0,
}


@dataclass
class Calibration:
    """Fitted standardisation + logistic weights, and the data they came from.

    `fitted_on` is stored and surfaced in every verdict's metadata because a calibration
    is only meaningful on the distribution it was fitted to. A reader who sees a
    probability should be able to see, in the same breath, what it was calibrated against.
    """

    mean: dict[str, float]
    std: dict[str, float]
    weights: dict[str, float]
    bias: float
    fitted_on: str
    n_samples: int


class StylometricDetector:
    """Feature-based detector. Fast, transparent, weak on clean text, stable under attack."""

    name = "stylometric"

    def __init__(self, min_tokens: int = 50, calibration: Calibration | None = None):
        self.min_tokens = min_tokens
        self.calibration = calibration

    def score_one(self, text: str) -> Verdict:
        refusal = gate_length(self.name, text, self.min_tokens)
        if refusal is not None:
            return refusal

        feats = extract_features(text)

        if self.calibration is None:
            # Unfitted: sum of signed, unit-weighted features. Monotone and comparable
            # within a run, meaningless as an absolute number — hence no probability.
            score = sum(HEURISTIC_SIGNS[k] * feats[k] for k in FEATURE_NAMES)
            return Verdict(
                detector=self.name,
                score=score,
                p_machine=None,
                meta={"features": feats, "calibrated": False},
            )

        c = self.calibration
        z = c.bias
        for k in FEATURE_NAMES:
            sd = c.std[k] or 1.0
            z += c.weights[k] * ((feats[k] - c.mean[k]) / sd)
        return Verdict(
            detector=self.name,
            score=z,
            p_machine=1.0 / (1.0 + math.exp(-z)),
            meta={
                "features": feats,
                "calibrated": True,
                "fitted_on": c.fitted_on,
                "n_calibration_samples": c.n_samples,
            },
        )

    def fit(
        self,
        human_texts: list[str],
        machine_texts: list[str],
        fitted_on: str,
        epochs: int = 400,
        lr: float = 0.1,
    ) -> Calibration:
        """Fit standardisation + logistic weights by gradient descent. Pure Python.

        Small and dependency-free on purpose: the point of this detector is that a reader
        can audit every step of it, and a hundred lines they can read beats a library
        call they must trust.
        """
        rows = [(extract_features(t), 0.0) for t in human_texts]
        rows += [(extract_features(t), 1.0) for t in machine_texts]
        if not rows:
            raise ValueError("no training data")

        n = len(rows)
        mean = {k: sum(f[k] for f, _ in rows) / n for k in FEATURE_NAMES}
        std = {
            k: math.sqrt(sum((f[k] - mean[k]) ** 2 for f, _ in rows) / n) or 1.0
            for k in FEATURE_NAMES
        }

        w = {k: 0.0 for k in FEATURE_NAMES}
        b = 0.0
        for _ in range(epochs):
            gw = {k: 0.0 for k in FEATURE_NAMES}
            gb = 0.0
            for f, y in rows:
                z = b + sum(w[k] * ((f[k] - mean[k]) / std[k]) for k in FEATURE_NAMES)
                p = 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
                err = p - y
                gb += err
                for k in FEATURE_NAMES:
                    gw[k] += err * ((f[k] - mean[k]) / std[k])
            b -= lr * gb / n
            for k in FEATURE_NAMES:
                w[k] -= lr * gw[k] / n

        self.calibration = Calibration(
            mean=mean, std=std, weights=w, bias=b, fitted_on=fitted_on, n_samples=n
        )
        return self.calibration
