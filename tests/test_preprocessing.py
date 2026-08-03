"""Preprocessor tests: each must remove what it claims and keep what it does not claim."""

import pytest

from detbench.preprocessing import (
    available,
    collapse_whitespace,
    get,
    prose_only,
    strip_code,
    strip_front_matter,
    strip_markdown,
    strip_urls,
    unicode_fold,
)

DOC = """---
title: "Example"
pubDate: 'Aug 1 2026'
---

# A heading

Some prose with `inline code` and a [link](https://example.com/page) in it.

```bash
cargo test --workspace
harness gate --world w.yaml
```

> A quoted line.

- a list item
- another item

| col | col |
|---|---|
| a | b |

More prose, **emphasised** and _slanted_, ending here.
"""


def test_registry_exposes_every_variant():
    assert set(available()) >= {
        "raw",
        "strip_code",
        "strip_front_matter",
        "strip_markdown",
        "strip_urls",
        "collapse_whitespace",
        "unicode_fold",
        "prose_only",
    }


def test_raw_is_the_identity():
    assert get("raw").apply(DOC) == DOC


def test_unknown_preprocessor_names_the_alternatives():
    with pytest.raises(KeyError, match="strip_code"):
        get("nope")


def test_strip_front_matter_removes_only_the_header_block():
    out = strip_front_matter(DOC)
    assert "pubDate" not in out
    assert "# A heading" in out
    # A document without front matter must be untouched.
    assert strip_front_matter("no front matter here") == "no front matter here"


def test_strip_code_removes_fenced_and_inline_but_keeps_prose():
    out = strip_code(DOC)
    assert "cargo test" not in out
    assert "inline code" not in out
    assert "Some prose with" in out
    assert "More prose" in out


def test_strip_urls_keeps_the_link_text():
    out = strip_urls(DOC)
    assert "example.com" not in out
    assert "link" in out


def test_strip_markdown_keeps_prose_and_link_text():
    out = strip_markdown(DOC)
    assert "A heading" in out  # heading text survives, the #s do not
    assert not out.lstrip().startswith("#")
    assert "link" in out
    assert "emphasised" in out and "**" not in out
    assert "a list item" in out


def test_collapse_whitespace_is_idempotent():
    once = collapse_whitespace(DOC)
    assert collapse_whitespace(once) == once
    assert "   " not in once


def test_unicode_fold_matches_the_attack_defence():
    from detbench.attacks.lexical import normalize_unicode

    text = "Thе аnаlуsis"  # Cyrillic lookalikes
    assert unicode_fold(text) == normalize_unicode(text)


def test_prose_only_composes_the_individual_steps():
    out = prose_only(DOC)
    for gone in ("pubDate", "cargo test", "example.com", "**", "|---|"):
        assert gone not in out
    assert "More prose" in out


def test_preprocessors_are_deterministic():
    for name in available():
        p = get(name)
        assert p.apply(DOC) == p.apply(DOC)
