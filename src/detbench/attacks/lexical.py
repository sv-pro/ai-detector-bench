"""No-model evasion attacks: homoglyphs, invisible characters, synonym swaps.

These three are grouped because they run anywhere, cost nothing, and are what an actual
evader reaches for first — you do not need DIPPER-11B to defeat a detector when a
find-and-replace will do. They are also the attacks a vendor is least likely to have
tested against, which is precisely why the leaderboard runs them.

All three are seeded and deterministic: the same text and seed always produce the same
attacked text, so a published row is reproducible.
"""

from __future__ import annotations

import random
import unicodedata

from .base import AttackResult, edit_rate

# Latin characters and their visually near-identical counterparts in other scripts.
# A reader sees no difference; a tokenizer sees entirely different tokens, which is the
# whole attack — it moves the text off the manifold the detector's model was trained on.
HOMOGLYPHS: dict[str, str] = {
    "a": "а",  # Cyrillic a
    "c": "с",  # Cyrillic es
    "e": "е",  # Cyrillic ie
    "o": "о",  # Cyrillic o
    "p": "р",  # Cyrillic er
    "x": "х",  # Cyrillic ha
    "y": "у",  # Cyrillic u
    "A": "Α",  # Greek Alpha
    "B": "Β",  # Greek Beta
    "E": "Ε",  # Greek Epsilon
    "H": "Η",  # Greek Eta
    "O": "Ο",  # Greek Omicron
}

ZERO_WIDTH = ["​", "‌", "‍", "⁠"]


class HomoglyphAttack:
    """Substitute a fraction of characters with confusable glyphs from other scripts.

    Note the asymmetry this exposes: a defender can neutralise this attack completely
    with a Unicode normalisation pass costing microseconds, yet most deployed detectors
    do not run one. The benchmark reports both the raw and the normalised result, so the
    row shows whether a detector is genuinely robust or merely un-normalised.
    """

    name = "homoglyph"
    requires_model = False

    def __init__(self, rate: float = 0.05):
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rate must be in [0, 1]")
        self.rate = rate

    def apply(self, text: str, seed: int = 0) -> AttackResult:
        rng = random.Random(seed)
        out = []
        swapped = 0
        for ch in text:
            if ch in HOMOGLYPHS and rng.random() < self.rate:
                out.append(HOMOGLYPHS[ch])
                swapped += 1
            else:
                out.append(ch)
        attacked = "".join(out)
        return AttackResult(
            attack=self.name,
            original=text,
            attacked=attacked,
            edit_rate=edit_rate(text, attacked),
            meta={"rate": self.rate, "chars_swapped": swapped, "seed": seed},
        )


class ZeroWidthAttack:
    """Insert invisible characters between words.

    Visually a no-op — copy the output into a document and it looks untouched — but it
    fragments tokenization. Included because it is the cheapest possible attack and
    therefore the floor: a detector that fails here fails against an evader who spent
    ten seconds.
    """

    name = "zero_width"
    requires_model = False

    def __init__(self, rate: float = 0.1):
        self.rate = rate

    def apply(self, text: str, seed: int = 0) -> AttackResult:
        rng = random.Random(seed)
        out = []
        inserted = 0
        for w in text.split(" "):
            # Appended directly to the word, never as its own space-delimited token:
            # inserting " ​ " would add real whitespace, which a reader *can* see.
            # The attack only counts if the rendered document is unchanged.
            if w and rng.random() < self.rate:
                out.append(w + rng.choice(ZERO_WIDTH))
                inserted += 1
            else:
                out.append(w)
        attacked = " ".join(out)
        return AttackResult(
            attack=self.name,
            original=text,
            attacked=attacked,
            edit_rate=edit_rate(text, attacked),
            meta={"rate": self.rate, "inserted": inserted, "seed": seed},
        )


# A deliberately small, hand-checked substitution table. This is *not* a general
# paraphraser and is not presented as one: it is a lower bound on synonym-swap evasion,
# using only pairs that are safe in nearly any context. A real synonym attack with
# WordNet or an LLM will do strictly better, which is the point — if this floor already
# moves a detector's score, the detector is fragile.
SYNONYMS: dict[str, str] = {
    "important": "significant",
    "significant": "important",
    "however": "nevertheless",
    "nevertheless": "however",
    "additionally": "moreover",
    "moreover": "additionally",
    "furthermore": "besides",
    "utilize": "use",
    "demonstrate": "show",
    "numerous": "many",
    "many": "numerous",
    "various": "several",
    "several": "various",
    "obtain": "get",
    "require": "need",
    "provide": "supply",
    "approximately": "roughly",
    "subsequently": "later",
    "therefore": "thus",
    "thus": "therefore",
    "crucial": "vital",
    "vital": "crucial",
    "enhance": "improve",
    "improve": "enhance",
    "essential": "necessary",
    "necessary": "essential",
    "typically": "usually",
    "usually": "typically",
    "particular": "specific",
    "specific": "particular",
}


class SynonymAttack:
    """Swap known-safe synonyms, preserving capitalisation and trailing punctuation.

    Reported separately from paraphrase because it isolates *lexical* choice from
    *structural* rewriting. Methods that lean on token-level likelihood suffer here;
    methods that lean on sentence-length variance largely do not. That split is one of
    the more useful things the leaderboard can show a reader.
    """

    name = "synonym"
    requires_model = False

    def __init__(self, rate: float = 1.0):
        self.rate = rate

    def apply(self, text: str, seed: int = 0) -> AttackResult:
        rng = random.Random(seed)
        out = []
        swapped = 0
        for token in text.split(" "):
            stripped = token.strip(".,;:!?()[]\"'")
            suffix = token[len(token.rstrip(".,;:!?()[]\"'")) :] if stripped else ""
            prefix = token[: len(token) - len(token.lstrip(".,;:!?()[]\"'"))]
            key = stripped.lower()
            if key in SYNONYMS and rng.random() < self.rate:
                repl = SYNONYMS[key]
                if stripped[:1].isupper():
                    repl = repl.capitalize()
                out.append(f"{prefix}{repl}{suffix}")
                swapped += 1
            else:
                out.append(token)
        attacked = " ".join(out)
        return AttackResult(
            attack=self.name,
            original=text,
            attacked=attacked,
            edit_rate=edit_rate(text, attacked),
            meta={"rate": self.rate, "words_swapped": swapped, "seed": seed},
        )


# Inverse of the homoglyph table, used to fold confusables back to Latin.
CONFUSABLE_FOLD: dict[str, str] = {v: k for k, v in HOMOGLYPHS.items()}


def normalize_unicode(text: str) -> str:
    """The defence against `HomoglyphAttack` and `ZeroWidthAttack`.

    Three separate passes, because a single one is not enough and assuming otherwise is
    a common mistake — this function originally shipped with only the first pass and a
    comment claiming it was sufficient, which the test suite promptly falsified:

    1. **NFKC** folds *compatibility* variants (ﬁ → fi, full-width forms). It does **not**
       touch cross-script confusables: Cyrillic 'а' U+0430 and Latin 'a' U+0061 are
       distinct characters that NFKC leaves exactly as it finds them.
    2. **Zero-width strip**, which NFKC also leaves in place.
    3. **Confusable fold**, which is what actually defeats the homoglyph attack. The
       table here covers the substitutions this package makes; a production deployment
       should use the full Unicode TR39 confusables data instead, of which this is a
       small hand-picked subset.

    Exposed so the harness can report each detector both with and without the defence a
    competent deployment would have.
    """
    text = unicodedata.normalize("NFKC", text)
    for zw in ZERO_WIDTH:
        text = text.replace(zw, "")
    return "".join(CONFUSABLE_FOLD.get(ch, ch) for ch in text)
