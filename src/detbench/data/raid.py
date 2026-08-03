"""RAID loader — the corpus that replaces the smoke fixture.

RAID (Dugan et al., ACL 2024) is the reference robustness benchmark for machine-generated
text detection: ~6.2M non-adversarial generations across 11 generators, 11 domains and 4
decoding strategies, plus 11 adversarial variants of each. MIT licensed.

    @inproceedings{dugan-etal-2024-raid,
      title  = {{RAID}: A Shared Benchmark for Robust Evaluation of
                Machine-Generated Text Detectors},
      author = {Dugan, Liam and Hwang, Alyssa and Trhlik, Filip and others},
      booktitle = {Proceedings of ACL},
      year   = {2024},
      url    = {https://arxiv.org/abs/2405.07940}
    }

**Three decisions this module makes on your behalf, each of which is a methodology choice
rather than plumbing.**

1. **Non-adversarial by default.** RAID ships pre-attacked rows, and `detbench` applies its
   own attacks. Loading the adversarial variants and then attacking them again would
   measure a compound transform nobody deploys. Pass `adversarial=True` deliberately — and
   when you do, the overlap is an *opportunity*: RAID's `homoglyph`, `zero_width_space` and
   `synonym` are reference implementations of three attacks this package also implements,
   so the same source documents can be used to check ours against theirs.

2. **English prose only, by default.** RAID includes a `code` domain (the human rows are
   literally Python source) and two non-English domains (`czech`, `german`). Scoring source
   code with a detector whose features are sentence-length variance and discourse markers
   produces a number with no meaning, and the published false-positive concentration on
   non-native English writers makes silently mixing languages worse than useless. Both are
   opt-in via `domains=`.

3. **Balanced by default.** RAID's natural class balance is roughly 1 human document per 11
   machine documents, because each human text seeds generations from every model. Reporting
   a false-positive rate off that ratio would rest on a small human sample while looking
   like a large study, so `load` samples up to `limit_per_class` of each label.

Downloads are streamed to a cache directory and never held in memory whole — the largest
file is 765 MB and the adversarial variants reach 11.8 GB.
"""

from __future__ import annotations

import csv
import os
import random
import sys
import urllib.request
from pathlib import Path
from typing import Iterator

from .fixtures import Document

RAID_BASE = "https://dataset.raid-bench.xyz"
CITATION = "Dugan et al., RAID (ACL 2024), arXiv:2405.07940, MIT licensed"

# Labelled splits carry a `model` column that identifies human rows. RAID-test withholds
# labels for the public leaderboard, so it cannot be used to compute a metric here.
LABELLED_SPLITS = {"train", "extra"}
UNLABELLED_SPLITS = {"test"}
SPLITS = LABELLED_SPLITS | UNLABELLED_SPLITS

# Verified against the real files on 2026-08-02, because the name misleads: **`extra` is
# not "more of the same"**. It holds only the three non-English / non-prose domains —
# `code` (32,200 rows), `german` (68,950) and `czech` (68,775). The eight English prose
# domains are in `train`. Loading `extra` with the default prose filter therefore matches
# nothing, which is correct behaviour and a confusing experience, so `load` says why.
SPLIT_DOMAINS = {
    "extra": frozenset({"code", "german", "czech"}),
    # `train` holds the English prose domains; not enumerated here because it is the
    # default and any filter mismatch is reported from the file itself.
}

# Approximate sizes, stated so a caller knows what a download costs before starting.
APPROX_BYTES = {
    ("train", False): 802_000_000,
    ("extra", False): 257_000_000,
    ("test", False): 81_000_000,
    ("train", True): 11_800_000_000,
    ("extra", True): 3_710_000_000,
    ("test", True): 1_220_000_000,
}

ENGLISH_PROSE_DOMAINS = frozenset(
    {"abstracts", "books", "news", "poetry", "recipes", "reddit", "reviews", "wiki"}
)
CODE_DOMAINS = frozenset({"code"})
NON_ENGLISH_DOMAINS = frozenset({"czech", "german"})
ALL_DOMAINS = ENGLISH_PROSE_DOMAINS | CODE_DOMAINS | NON_ENGLISH_DOMAINS

RAID_ATTACKS = frozenset(
    {
        "none", "homoglyph", "number", "article_deletion", "insert_paragraphs",
        "perplexity_misspelling", "upper_lower", "whitespace", "zero_width_space",
        "synonym", "paraphrase", "alternative_spelling",
    }
)

# RAID attacks with a counterpart in `detbench.attacks`. Same source documents through both
# gives a direct check of our implementations against the reference ones.
ATTACK_COUNTERPARTS = {
    "homoglyph": "homoglyph",
    "zero_width_space": "zero_width",
    "synonym": "synonym",
}

HUMAN_MODEL = "human"


def default_cache_dir() -> Path:
    return Path(os.environ.get("DETBENCH_CACHE", Path.home() / ".cache" / "detbench"))


def filename(split: str, adversarial: bool = False) -> str:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")
    return f"{split}.csv" if adversarial else f"{split}_none.csv"


def url(split: str, adversarial: bool = False) -> str:
    return f"{RAID_BASE}/{filename(split, adversarial)}"


def cache_path(split: str, adversarial: bool = False, cache_dir: Path | None = None) -> Path:
    return (cache_dir or default_cache_dir()) / "raid" / filename(split, adversarial)


def download(
    split: str,
    adversarial: bool = False,
    cache_dir: Path | None = None,
    max_bytes: int | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Stream a RAID split to the cache. Returns the local path.

    `max_bytes` fetches only a prefix via an HTTP range request, which is how you sanity-
    check a pipeline against real data without pulling hundreds of megabytes. A truncated
    prefix ends mid-row, so the final partial line is discarded rather than parsed — a
    half-row would silently become a document with a mangled `generation`.
    """
    dest = cache_path(split, adversarial, cache_dir)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    src = url(split, adversarial)
    req = urllib.request.Request(src)
    if max_bytes is not None:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")

    tmp = dest.with_suffix(dest.suffix + ".part")
    written = 0
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            written += len(chunk)
            # Only animate for a human at a terminal; piped or captured output would
            # otherwise accumulate one line per megabyte.
            if progress and sys.stderr.isatty():
                print(f"\r  {split}: {written / 1e6:.1f} MB", end="", file=sys.stderr)
    if progress:
        print(f"\r  {split}: {written / 1e6:.1f} MB", file=sys.stderr)

    if max_bytes is not None:
        # Drop the trailing partial row.
        data = tmp.read_bytes()
        cut = data.rfind(b"\n")
        if cut > 0:
            tmp.write_bytes(data[: cut + 1])

    tmp.replace(dest)
    return dest


def iter_rows(path: Path) -> Iterator[dict]:
    """Stream CSV rows. Never loads the file whole — these reach 11.8 GB."""
    # RAID generations run to ~75k characters, past csv's default field ceiling.
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        yield from csv.DictReader(fh)


def row_to_document(row: dict) -> Document:
    """One RAID row → a labelled Document, with provenance kept in `source`."""
    model = (row.get("model") or "").strip()
    label = 0 if model == HUMAN_MODEL else 1
    bits = [model or "?", row.get("domain") or "?"]
    if row.get("decoding"):
        bits.append(row["decoding"])
    attack = row.get("attack") or "none"
    if attack != "none":
        bits.append(f"attack={attack}")
    return Document(text=row.get("generation") or "", label=label, source="raid: " + " · ".join(bits))


def domains_in(path: Path) -> dict[str, int]:
    """Domain histogram for a cached file. Used to explain an empty result."""
    counts: dict[str, int] = {}
    for row in iter_rows(path):
        d = row.get("domain") or "?"
        counts[d] = counts.get(d, 0) + 1
    return counts


def load(
    split: str = "train",
    adversarial: bool = False,
    domains: set[str] | frozenset[str] | None = None,
    models: set[str] | None = None,
    attacks: set[str] | None = None,
    limit_per_class: int = 500,
    min_chars: int = 200,
    seed: int = 0,
    cache_dir: Path | None = None,
    path: Path | None = None,
) -> list[Document]:
    """Load a balanced, filtered subset of RAID as `Document`s.

    Reservoir-samples each class while streaming, so the subset is drawn from the whole
    file rather than from whatever happens to sit at the top of it — RAID is grouped by
    source document, and taking the first N rows would return a handful of topics.
    """
    if split in UNLABELLED_SPLITS:
        raise ValueError(
            f"split {split!r} withholds labels for the public leaderboard; "
            f"use one of {sorted(LABELLED_SPLITS)} to compute metrics"
        )
    domains = frozenset(domains) if domains is not None else ENGLISH_PROSE_DOMAINS
    unknown = set(domains) - ALL_DOMAINS
    if unknown:
        raise ValueError(f"unknown domain(s): {sorted(unknown)}")

    src = path or cache_path(split, adversarial, cache_dir)
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found — run `detbench raid-fetch --split {split}` first "
            f"(~{APPROX_BYTES.get((split, adversarial), 0) / 1e6:.0f} MB). {CITATION}"
        )

    rng = random.Random(seed)
    reservoir: dict[int, list[Document]] = {0: [], 1: []}
    seen: dict[int, int] = {0: 0, 1: 0}

    for row in iter_rows(src):
        if (row.get("domain") or "") not in domains:
            continue
        if models is not None and (row.get("model") or "") not in models:
            continue
        if attacks is not None and (row.get("attack") or "none") not in attacks:
            continue
        text = row.get("generation") or ""
        if len(text) < min_chars:
            continue

        doc = row_to_document(row)
        pool, n = reservoir[doc.label], seen[doc.label]
        if len(pool) < limit_per_class:
            pool.append(doc)
        else:
            j = rng.randrange(n + 1)
            if j < limit_per_class:
                pool[j] = doc
        seen[doc.label] = n + 1

    docs = reservoir[0] + reservoir[1]
    if not docs:
        # An empty result is almost always a domain mismatch, and silently returning []
        # sends the caller hunting through their filters. Say what is actually in there.
        present = domains_in(src)
        raise ValueError(
            f"no documents matched in {src.name}.\n"
            f"  requested domains: {sorted(domains)}\n"
            f"  present in file  : {sorted(present)}\n"
            + (
                "  note: the `extra` split holds ONLY code/german/czech. The English "
                "prose domains are in `train` — try --split train.\n"
                if split == "extra"
                else ""
            )
        )
    return docs


def describe(docs: list[Document]) -> str:
    """Composition of a loaded subset. Printed with results so a table is self-describing."""
    n_h = sum(1 for d in docs if d.label == 0)
    n_m = len(docs) - n_h
    doms: dict[str, int] = {}
    models: dict[str, int] = {}
    for d in docs:
        parts = d.source.removeprefix("raid: ").split(" · ")
        if len(parts) >= 2:
            models[parts[0]] = models.get(parts[0], 0) + 1
            doms[parts[1]] = doms.get(parts[1], 0) + 1
    top = ", ".join(f"{k}={v}" for k, v in sorted(doms.items(), key=lambda x: -x[1])[:6])
    return (
        f"RAID subset: {len(docs)} docs ({n_h} human / {n_m} machine), "
        f"{len(models)} generators, domains: {top}\n  {CITATION}"
    )
