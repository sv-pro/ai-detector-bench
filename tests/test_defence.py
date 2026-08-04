"""Defended-variant tests.

A defence is a preprocessor applied at the detector's input, to *every* document. These
pin the two things that would quietly invalidate a recovery number: defending only the
attacked half, and pairing the wrong slices when computing recovery.
"""

import pytest

from detbench.attacks import build as build_attack
from detbench.attacks.lexical import normalize_unicode
from detbench.core import RefusalReason, Verdict
from detbench.data.fixtures import load_smoke
from detbench.harness import build_slices, recovery_table, run
from detbench.metrics import Report


def _slice_named(slices, name):
    return next(s for s in slices if s.name == name)


def test_defences_produce_a_paired_slice_for_every_base_slice():
    docs = load_smoke()
    slices = build_slices(docs, [build_attack("homoglyph")], defences=["unicode_fold"])
    assert [s.name for s in slices] == [
        "clean",
        "clean +unicode_fold",
        "attack:homoglyph",
        "attack:homoglyph +unicode_fold",
    ]


def test_no_defences_leaves_the_slice_set_unchanged():
    docs = load_smoke()
    assert len(build_slices(docs, [build_attack("homoglyph")])) == 2


def test_defence_is_applied_to_human_documents_too():
    # A deployment normalises whatever arrives; it cannot know which documents were
    # attacked. Defending only the machine half would measure a capability nobody has.
    docs = load_smoke()
    slices = build_slices(docs, [], defences=["unicode_fold"])
    clean, defended = slices[0], slices[1]
    human_idx = [i for i, d in enumerate(docs) if d.label == 0]
    assert human_idx
    for i in human_idx:
        assert defended.texts[i] == normalize_unicode(clean.texts[i])


def test_unicode_fold_fully_reverses_the_unicode_attacks_on_ascii_text():
    # True **on the fixture**, whose documents contain no characters the fold rewrites.
    # It is NOT a general property — see the next test, which is the one that matters.
    docs = load_smoke()
    for attack in ("homoglyph", "zero_width"):
        slices = build_slices(docs, [build_attack(attack)], defences=["unicode_fold"])
        clean = _slice_named(slices, "clean")
        attacked = _slice_named(slices, f"attack:{attack}")
        defended = _slice_named(slices, f"attack:{attack} +unicode_fold")
        assert attacked.texts != clean.texts, f"{attack} changed nothing"
        assert defended.texts == clean.texts, f"{attack} not fully reversed"


def test_the_defence_is_lossy_on_text_it_was_not_defending():
    """The defence rewrites legitimate documents too, so it has a cost of its own.

    This kills an attractive shortcut. It is tempting to skip the `synonym +unicode_fold`
    cell of an expensive run, reasoning that a defence against Unicode attacks "cannot
    matter" for a purely lexical attack. On RAID it does matter: `unicode_fold` alters 61
    of 1,500 synonym-attacked documents, because 411 of them contain non-ASCII that NFKC
    or the confusables fold rewrites — and under the Unicode attacks the defence restores
    only 1,439 of 1,500 documents to their original form, not all of them.

    So a defence is never free, every cell must actually be measured, and the
    `clean +defence` control exists precisely to price the damage the defence does when
    there was no attack at all.
    """
    ligatures = "Deﬁne the ﬁrst ﬂag"  # NFKC expands these to plain ASCII
    assert normalize_unicode(ligatures) != ligatures

    cyrillic = "Ineﬃcient аnalysis"  # a naturally-occurring Cyrillic 'а'
    assert normalize_unicode(cyrillic) != cyrillic


def test_slices_record_attack_and_defence_separately_from_the_name():
    docs = load_smoke()
    slices = build_slices(docs, [build_attack("homoglyph")], defences=["unicode_fold"])
    s = _slice_named(slices, "attack:homoglyph +unicode_fold")
    assert s.attack == "homoglyph"
    assert s.defence == "unicode_fold"
    assert _slice_named(slices, "clean").attack is None


def _report(detector, slice_name, tpr):
    return Report(
        detector=detector, slice_name=slice_name, n_total=100, n_scored=100, refusal=0.0,
        tpr_at_1pct=tpr, tpr_at_0_1pct=None, overconfidence=None, auroc=None,
    )


def test_recovery_is_the_fraction_of_damage_undone():
    # clean 50%, attacked 20%, defended 44% -> (44-20)/(50-20) = 80%
    reports = [
        _report("d", "clean", 0.50),
        _report("d", "attack:homoglyph", 0.20),
        _report("d", "attack:homoglyph +unicode_fold", 0.44),
    ]
    assert "80%" in recovery_table(reports)


def test_full_recovery_reads_as_100_percent():
    reports = [
        _report("d", "clean", 0.50),
        _report("d", "attack:homoglyph", 0.20),
        _report("d", "attack:homoglyph +unicode_fold", 0.50),
    ]
    assert "100%" in recovery_table(reports)


def test_a_defence_that_makes_things_worse_reads_negative():
    reports = [
        _report("d", "clean", 0.50),
        _report("d", "attack:homoglyph", 0.20),
        _report("d", "attack:homoglyph +unicode_fold", 0.14),
    ]
    assert "-20%" in recovery_table(reports)


def test_recovery_is_na_when_the_attack_did_no_damage():
    # Dividing by ~zero would manufacture a dramatic number out of noise.
    reports = [
        _report("d", "clean", 0.50),
        _report("d", "attack:synonym", 0.50),
        _report("d", "attack:synonym +unicode_fold", 0.50),
    ]
    assert "n/a" in recovery_table(reports)


def test_recovery_table_is_empty_without_defended_slices():
    assert recovery_table([_report("d", "clean", 0.5)]) == "(no defended slices)"


def test_end_to_end_defended_run_on_the_fixture():
    docs = load_smoke()
    reports = run(
        [__import__("detbench.detectors", fromlist=["build"]).build("stylometric")],
        docs,
        ["homoglyph"],
        defences=["unicode_fold"],
    )
    names = {r.slice_name for r in reports}
    assert "attack:homoglyph +unicode_fold" in names
    # Defended slice must match clean exactly, since the defence fully reverses the attack.
    clean = next(r for r in reports if r.slice_name == "clean")
    defended = next(r for r in reports if r.slice_name.endswith("+unicode_fold") and "attack" in r.slice_name)
    assert defended.auroc == pytest.approx(clean.auroc)
