"""Sensitivity tests.

The behaviours worth pinning are the ones that keep the headline number honest: a shift is
only counted when both sides produced a score, and the ratio refuses to exist without a
signal to measure against.
"""

from detbench.core import RefusalReason, Verdict
from detbench.data.fixtures import load_smoke
from detbench.detectors import build
from detbench.preprocessing import get
from detbench.sensitivity import measure, render_notes, render_table


class _Fixed:
    """Detector returning a preset score per text. Makes displacement exact."""

    name = "fixed"

    def __init__(self, table: dict[str, float], refuse: set[str] | None = None):
        self.table = table
        self.refuse = refuse or set()

    def score_one(self, text: str) -> Verdict:
        if text in self.refuse:
            return Verdict.refuse(self.name, RefusalReason.TOO_SHORT)
        return Verdict(detector=self.name, score=self.table[text])


class _Const:
    """Preprocessor mapping any input to a fixed output."""

    def __init__(self, name: str, out: str):
        self.name = name
        self.out = out

    def apply(self, text: str) -> str:
        return self.out


def test_identity_preprocessing_shows_no_shift():
    det = build("stylometric")
    docs = load_smoke()
    reports = measure(det, [d.text for d in docs], ["raw"], labels=[d.label for d in docs])
    assert reports[0].mean_abs_shift == 0.0
    assert reports[0].max_abs_shift == 0.0
    assert reports[0].flip_rate == 0.0
    assert reports[0].verdict == "negligible"


def test_ratio_is_shift_over_signal_gap():
    # human=0.0, machine=1.0 → gap 1.0; every doc shifts by exactly 0.5 → ratio 0.5.
    det = _Fixed({"h": 0.0, "m": 1.0, "x": 0.5})
    reports = measure(det, ["h", "m"], [_Const("p", "x")], labels=[0, 1])
    r = reports[0]
    assert r.signal_gap == 1.0
    assert r.mean_abs_shift == 0.5
    assert r.sensitivity_ratio == 0.5
    assert r.verdict == "material"


def test_ratio_above_one_is_reported_as_dominating():
    det = _Fixed({"h": 0.0, "m": 0.2, "x": 2.0})
    reports = measure(det, ["h", "m"], [_Const("p", "x")], labels=[0, 1])
    assert reports[0].sensitivity_ratio > 1.0
    assert reports[0].verdict == "DOMINATES"


def test_no_labels_means_no_ratio_and_no_flips():
    # Shift magnitudes are still meaningful; a ratio without a signal is not.
    det = _Fixed({"h": 0.0, "m": 1.0, "x": 0.5})
    reports = measure(det, ["h", "m"], [_Const("p", "x")], labels=None)
    r = reports[0]
    assert r.mean_abs_shift == 0.5
    assert r.signal_gap is None
    assert r.sensitivity_ratio is None
    assert r.flip_rate is None
    assert r.verdict == "unlabelled"


def test_refused_documents_are_excluded_not_counted_as_zero_shift():
    # "x" refuses after preprocessing. Treating that as zero displacement would make an
    # unstable preprocessor look perfectly stable.
    det = _Fixed({"h": 0.0, "m": 1.0, "x": 0.0}, refuse={"x"})
    reports = measure(det, ["h", "m"], [_Const("p", "x")], labels=[0, 1])
    r = reports[0]
    assert r.n_compared == 0
    assert r.mean_abs_shift is None
    assert r.n_refused_raw == 0
    assert r.n_refused_variant == 2


def test_newly_refused_is_surfaced_in_the_row_and_notes():
    det = _Fixed({"h": 0.0, "m": 1.0, "x": 0.0}, refuse={"x"})
    reports = measure(det, ["h", "m"], [_Const("p", "x")], labels=[0, 1])
    assert reports[0].as_row()["newly refused"] == 2
    assert "below the length gate" in render_notes(reports)


def test_flip_rate_counts_documents_crossing_the_reference_line():
    # Threshold is the midpoint of class means = 0.5; both documents swap sides.
    det = _Fixed({"h": 0.0, "m": 1.0, "H": 1.0, "M": 0.0})

    class Swap:
        name = "swap"

        def apply(self, text: str) -> str:
            return {"h": "H", "m": "M"}[text]

    reports = measure(det, ["h", "m"], [Swap()], labels=[0, 1])
    assert reports[0].flip_rate == 1.0


def test_table_and_notes_render_for_the_real_detector():
    det = build("stylometric")
    docs = load_smoke()
    reports = measure(
        det,
        [d.text for d in docs],
        ["strip_code", "prose_only"],
        labels=[d.label for d in docs],
    )
    table = render_table(reports)
    assert "ratio" in table and "verdict" in table
    assert "signal gap" in render_notes(reports)
