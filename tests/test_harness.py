"""End-to-end harness tests, including that the caveats travel with the numbers."""

from detbench.attacks import build as build_attack
from detbench.data.fixtures import load_smoke
from detbench.detectors import build as build_detector
from detbench.harness import build_slices, render_notes, render_table, run


def test_slices_cover_clean_plus_one_per_attack():
    docs = load_smoke()
    attacks = [build_attack("homoglyph"), build_attack("synonym")]
    slices = build_slices(docs, attacks)
    assert [s.name for s in slices] == ["clean", "attack:homoglyph", "attack:synonym"]


def test_attacks_touch_machine_text_only():
    # Laundering human text would measure something nobody does, and would let a
    # detector's post-attack score improve for the wrong reason.
    docs = load_smoke()
    slices = build_slices(docs, [build_attack("synonym")])
    clean, attacked = slices[0], slices[1]
    for doc, before, after in zip(docs, clean.texts, attacked.texts):
        if doc.label == 0:
            assert before == after


def test_run_produces_a_report_per_detector_slice_pair():
    docs = load_smoke()
    reports = run([build_detector("stylometric")], docs, ["homoglyph", "synonym"])
    assert len(reports) == 3
    assert all(r.n_total == len(docs) for r in reports)


def test_table_leads_with_the_deployment_metric_not_auroc():
    reports = run([build_detector("stylometric")], load_smoke(), [])
    table = render_table(reports)
    assert table.index("TPR@1%FPR") < table.index("AUROC")


def test_small_fixture_cannot_resolve_a_strict_false_positive_rate():
    # Six human documents cannot support a 0.1% claim, and the harness says so rather
    # than printing a number. This is the demo's intended lesson.
    reports = run([build_detector("stylometric")], load_smoke(), [])
    assert all(r.tpr_at_0_1pct is None for r in reports)
    assert "n/a" in render_table(reports)


def test_notes_explain_every_n_a_that_appears():
    reports = run([build_detector("stylometric")], load_smoke(), ["synonym"])
    notes = render_notes(reports)
    assert "1,000 human documents" in notes
    assert "uncalibrated" in notes
    assert "AUROC" in notes
