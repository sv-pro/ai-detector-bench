"""Stylometric detector tests.

The most important test here is `test_unfitted_detector_reports_no_probability`. Everything
else is mechanics; that one is the project's thesis expressed as an assertion.
"""

import pytest

from detbench.core import RefusalReason
from detbench.data.fixtures import HUMAN, MACHINE
from detbench.detectors import build
from detbench.detectors.stylometric import (
    FEATURE_NAMES,
    StylometricDetector,
    extract_features,
)

LONG_HUMAN = HUMAN[0].text
LONG_MACHINE = MACHINE[0].text


def test_registered_and_buildable():
    det = build("stylometric")
    assert det.name == "stylometric"


def test_extracts_every_declared_feature():
    feats = extract_features(LONG_HUMAN)
    assert set(feats) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in feats.values())


def test_features_are_length_normalised():
    # Doubling a document should leave rate features roughly where they were; a feature
    # that scaled with length would just be re-encoding length, which the refusal gate
    # already handles.
    once = extract_features(LONG_MACHINE)
    twice = extract_features(LONG_MACHINE + " " + LONG_MACHINE)
    assert abs(once["comma_rate"] - twice["comma_rate"]) < 0.02
    assert abs(once["llm_marker_rate"] - twice["llm_marker_rate"]) < 0.02


def test_unfitted_detector_reports_no_probability():
    # A hand-weighted feature sum is not a probability. Printing one would be the
    # precise dishonesty this repository exists to measure in others.
    v = StylometricDetector().score_one(LONG_MACHINE)
    assert not v.refused
    assert v.score is not None
    assert v.p_machine is None
    assert v.meta["calibrated"] is False


def test_refuses_short_text():
    v = StylometricDetector().score_one("Only a handful of words here.")
    assert v.refused and v.reason == RefusalReason.TOO_SHORT


def test_fitting_produces_calibrated_probabilities_with_provenance():
    det = StylometricDetector()
    det.fit(
        [d.text for d in HUMAN],
        [d.text for d in MACHINE],
        fitted_on="smoke-fixture-v1",
    )
    v = det.score_one(LONG_MACHINE)
    assert v.p_machine is not None
    assert 0.0 <= v.p_machine <= 1.0
    # A probability must arrive with the distribution it was calibrated against.
    assert v.meta["fitted_on"] == "smoke-fixture-v1"
    assert v.meta["n_calibration_samples"] == len(HUMAN) + len(MACHINE)


def test_fitting_separates_the_fixture_it_was_fitted_on():
    # In-distribution separation is the easy case and proves only that the fitting loop
    # works. It is deliberately NOT evidence the detector generalises — see
    # docs/METHODOLOGY.md on distribution shift.
    det = StylometricDetector()
    det.fit([d.text for d in HUMAN], [d.text for d in MACHINE], fitted_on="smoke")
    human_p = [det.score_one(d.text).p_machine for d in HUMAN]
    machine_p = [det.score_one(d.text).p_machine for d in MACHINE]
    assert max(human_p) < min(machine_p)


def test_fit_rejects_empty_training_data():
    with pytest.raises(ValueError):
        StylometricDetector().fit([], [], fitted_on="nothing")
