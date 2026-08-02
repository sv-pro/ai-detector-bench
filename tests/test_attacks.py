"""Attack tests. Determinism and the visual-identity property are the load-bearing ones."""

from detbench.attacks import available, build
from detbench.attacks.lexical import (
    HomoglyphAttack,
    SynonymAttack,
    ZeroWidthAttack,
    normalize_unicode,
)

TEXT = (
    "The comprehensive analysis demonstrates several important considerations. "
    "Additionally, the approach requires numerous resources to obtain useful results. "
    "However, various factors typically improve the specific outcome considerably."
)


def test_registry_lists_the_no_model_attacks():
    assert set(available()) >= {"homoglyph", "zero_width", "synonym"}
    assert build("homoglyph").requires_model is False


def test_attacks_are_deterministic_under_a_seed():
    for atk in (HomoglyphAttack(rate=0.5), ZeroWidthAttack(rate=0.5), SynonymAttack()):
        a = atk.apply(TEXT, seed=7).attacked
        b = atk.apply(TEXT, seed=7).attacked
        assert a == b, f"{atk.name} is not reproducible"


def test_different_seeds_differ_for_probabilistic_attacks():
    atk = HomoglyphAttack(rate=0.5)
    assert atk.apply(TEXT, seed=1).attacked != atk.apply(TEXT, seed=2).attacked


def test_zero_width_is_invisible_but_changes_the_bytes():
    res = ZeroWidthAttack(rate=1.0).apply(TEXT, seed=0)
    assert res.attacked != TEXT
    # A reader sees the same document; only the byte stream moved.
    assert normalize_unicode(res.attacked) == TEXT


def test_homoglyph_survives_reading_but_not_normalisation():
    res = HomoglyphAttack(rate=1.0).apply(TEXT, seed=0)
    assert res.attacked != TEXT
    assert res.meta["chars_swapped"] > 0
    # The defence any competent deployment would run undoes it entirely.
    assert normalize_unicode(res.attacked) == TEXT


def test_synonym_swap_preserves_capitalisation_and_punctuation():
    res = SynonymAttack().apply("However, the important results. Additionally, many.", seed=0)
    assert res.attacked.startswith("Nevertheless,")
    assert res.attacked.endswith(".")
    assert res.meta["words_swapped"] >= 3


def test_edit_rate_is_zero_for_an_untouched_document():
    res = SynonymAttack(rate=0.0).apply(TEXT, seed=0)
    assert res.edit_rate == 0.0
    assert res.attacked == TEXT


def test_edit_rate_is_reported_for_auditing():
    res = SynonymAttack(rate=1.0).apply(TEXT, seed=0)
    # An attack that rewrote everything would be destroying the document, not evading.
    assert 0.0 < res.edit_rate < 0.5
