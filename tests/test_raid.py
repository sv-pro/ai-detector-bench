"""RAID loader tests.

All offline — they run against a synthetic CSV with RAID's exact column names, verified
against the real header at `https://dataset.raid-bench.xyz/extra_none.csv` on 2026-08-02:

    id,adv_source_id,source_id,model,decoding,repetition_penalty,attack,domain,title,prompt,generation

The defaults these tests pin are methodology choices, not conveniences: English prose only,
non-adversarial only, balanced classes. Each is something a caller could get silently wrong.
"""

import csv

import pytest

from detbench.data import raid

HEADER = [
    "id", "adv_source_id", "source_id", "model", "decoding", "repetition_penalty",
    "attack", "domain", "title", "prompt", "generation",
]

PROSE = (
    "The committee published its findings after a lengthy review of the available "
    "evidence, and the conclusions were broadly consistent with earlier work in the "
    "field, though several members recorded reservations about the methodology used."
)


def _row(model, domain, attack="none", text=PROSE, decoding="greedy"):
    return {
        "id": "x", "adv_source_id": "x", "source_id": "x",
        "model": model, "decoding": decoding, "repetition_penalty": "no",
        "attack": attack, "domain": domain, "title": "t", "prompt": "p",
        "generation": text,
    }


@pytest.fixture
def csv_path(tmp_path):
    rows = []
    for i in range(8):
        rows.append(_row("human", "news"))
        rows.append(_row("chatgpt", "news"))
        rows.append(_row("gpt4", "wiki"))
    rows.append(_row("human", "code", text="def f(x):\n    return x + 1\n" * 12))
    rows.append(_row("chatgpt", "code", text="def g(y):\n    return y * 2\n" * 12))
    rows.append(_row("human", "german", text="Der Ausschuss " * 40))
    rows.append(_row("chatgpt", "news", attack="homoglyph"))
    rows.append(_row("human", "news", text="too short"))

    p = tmp_path / "extra_none.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    return p


def test_url_and_filename_match_the_published_layout():
    assert raid.url("extra") == "https://dataset.raid-bench.xyz/extra_none.csv"
    assert raid.url("train", adversarial=True) == "https://dataset.raid-bench.xyz/train.csv"
    with pytest.raises(ValueError):
        raid.filename("nope")


def test_unlabelled_test_split_is_refused_with_a_reason():
    # RAID-test withholds labels for the public leaderboard, so no metric can come from it.
    with pytest.raises(ValueError, match="withholds labels"):
        raid.load(split="test")


def test_missing_cache_names_the_command_and_the_download_size():
    with pytest.raises(FileNotFoundError, match="raid-fetch"):
        raid.load(split="extra", path=None, cache_dir=raid.Path("/nonexistent"))


def test_human_rows_are_label_zero_and_generators_are_label_one():
    assert raid.row_to_document(_row("human", "news")).label == 0
    assert raid.row_to_document(_row("chatgpt", "news")).label == 1


def test_provenance_is_preserved_in_source():
    doc = raid.row_to_document(_row("chatgpt", "news", attack="homoglyph"))
    assert "chatgpt" in doc.source and "news" in doc.source
    assert "attack=homoglyph" in doc.source


def test_defaults_exclude_code_and_non_english(csv_path):
    docs = raid.load(path=csv_path, limit_per_class=100)
    for d in docs:
        assert "code" not in d.source
        assert "german" not in d.source


def test_code_domain_is_opt_in(csv_path):
    docs = raid.load(path=csv_path, domains={"code"}, limit_per_class=100, min_chars=10)
    assert docs and all("code" in d.source for d in docs)


def test_defaults_exclude_pre_attacked_rows_only_when_asked(csv_path):
    # `attacks=` is an explicit filter; the adversarial *file* is the other axis.
    clean = raid.load(path=csv_path, attacks={"none"}, limit_per_class=100)
    assert all("attack=" not in d.source for d in clean)
    attacked = raid.load(path=csv_path, attacks={"homoglyph"}, limit_per_class=100)
    assert attacked and all("attack=homoglyph" in d.source for d in attacked)


def test_short_documents_are_dropped_before_the_detector_sees_them(csv_path):
    docs = raid.load(path=csv_path, limit_per_class=100, min_chars=200)
    assert all(len(d.text) >= 200 for d in docs)


def test_classes_are_balanced_by_limit_per_class(csv_path):
    docs = raid.load(path=csv_path, limit_per_class=3)
    assert sum(1 for d in docs if d.label == 0) == 3
    assert sum(1 for d in docs if d.label == 1) == 3


def test_sampling_is_reproducible_under_a_seed(csv_path):
    a = raid.load(path=csv_path, limit_per_class=2, seed=7)
    b = raid.load(path=csv_path, limit_per_class=2, seed=7)
    assert [d.text for d in a] == [d.text for d in b]


def test_unknown_domain_is_rejected_rather_than_silently_empty(csv_path):
    with pytest.raises(ValueError, match="unknown domain"):
        raid.load(path=csv_path, domains={"nonsense"})


def test_empty_match_reports_what_is_actually_in_the_file(tmp_path):
    # An empty result is nearly always a domain mismatch. Returning [] would send the
    # caller hunting through their own filters instead of the data.
    p = tmp_path / "extra_none.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        w.writerows([_row("human", "czech", text="Ceska " * 60)])
    with pytest.raises(ValueError) as exc:
        raid.load(split="extra", path=p, domains={"news"})
    msg = str(exc.value)
    assert "requested domains" in msg and "present in file" in msg
    assert "czech" in msg
    # The `extra` split is the trap this hint exists for.
    assert "--split train" in msg


def test_extra_split_domains_are_recorded_because_the_name_misleads():
    # Verified against the real file: `extra` is NOT more of the same. It holds only the
    # non-English and non-prose domains; the English prose domains are in `train`.
    assert raid.SPLIT_DOMAINS["extra"] == frozenset({"code", "german", "czech"})
    assert not (raid.SPLIT_DOMAINS["extra"] & raid.ENGLISH_PROSE_DOMAINS)


def test_default_split_is_the_one_with_english_prose():
    import inspect

    assert inspect.signature(raid.load).parameters["split"].default == "train"


def test_describe_reports_composition_and_citation(csv_path):
    docs = raid.load(path=csv_path, limit_per_class=5)
    text = raid.describe(docs)
    assert "human" in text and "machine" in text
    assert "Dugan" in text and "MIT" in text


def test_attack_counterparts_map_to_registered_detbench_attacks():
    from detbench.attacks import available

    for raid_name, ours in raid.ATTACK_COUNTERPARTS.items():
        assert raid_name in raid.RAID_ATTACKS
        assert ours in available()
