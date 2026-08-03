"""Preprocessing variants — the pipeline choices nobody writes down.

This is deliberately **not** in `attacks/`, and the distinction is the point. An attack is
adversarial, applied to machine text only, by someone trying to evade. A preprocessor is a
*defender's* choice, applied uniformly to every document, made once by whoever wired the
tool up and then never mentioned again: strip the code blocks or don't, drop the front
matter or don't, collapse the whitespace or don't.

Nobody reports these choices, because they feel like plumbing rather than methodology. The
first real document this benchmark was pointed at said otherwise: stripping fenced code
from a 2,800-word technical tutorial moved its score by **+0.275**, further than the gap
between several documents whose true labels were opposite. If an undocumented plumbing
decision moves the answer more than the signal does, then two honest people running "the
same detector" on the same text will disagree, and neither will be able to say why.

`sensitivity.py` turns that into a number.

Every preprocessor is pure, deterministic, and order-independent of the others, so a
variant is reproducible from its name alone.
"""

from __future__ import annotations

import re
from typing import Callable, Protocol, runtime_checkable

FRONT_MATTER = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.S)
FENCED_CODE = re.compile(r"```.*?```", re.S)
INDENTED_CODE = re.compile(r"(?m)^(?: {4}|\t).*$")
INLINE_CODE = re.compile(r"`[^`\n]+`")
URL = re.compile(r"https?://\S+|www\.\S+")
MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
MD_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
MD_BLOCKQUOTE = re.compile(r"(?m)^\s{0,3}>\s?")
MD_LIST = re.compile(r"(?m)^\s*(?:[-*+]|\d+\.)\s+")
MD_RULE = re.compile(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
MD_TABLE_ROW = re.compile(r"(?m)^\s*\|.*\|\s*$")
MD_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1", re.S)
WHITESPACE = re.compile(r"[ \t]+")
BLANK_LINES = re.compile(r"\n{3,}")


@runtime_checkable
class Preprocessor(Protocol):
    name: str

    def apply(self, text: str) -> str: ...


class _Fn:
    """Adapter turning a named function into a `Preprocessor`."""

    def __init__(self, name: str, fn: Callable[[str], str], doc: str = ""):
        self.name = name
        self._fn = fn
        self.__doc__ = doc

    def apply(self, text: str) -> str:
        return self._fn(text)


def identity(text: str) -> str:
    return text


def strip_front_matter(text: str) -> str:
    return FRONT_MATTER.sub("", text)


def strip_code(text: str) -> str:
    """Remove fenced, indented, and inline code.

    The highest-impact variant on technical documents, and the one most likely to be
    applied silently — code blocks look obviously like noise, so people drop them without
    recording that they did.
    """
    text = FENCED_CODE.sub(" ", text)
    text = INDENTED_CODE.sub(" ", text)
    return INLINE_CODE.sub(" ", text)


def strip_urls(text: str) -> str:
    return URL.sub(" ", text)


def strip_markdown(text: str) -> str:
    """Reduce markdown to its prose, keeping link and image *text* but dropping targets."""
    text = MD_IMAGE.sub(r"\1", text)
    text = MD_LINK.sub(r"\1", text)
    text = MD_TABLE_ROW.sub(" ", text)
    text = MD_RULE.sub(" ", text)
    text = MD_HEADING.sub("", text)
    text = MD_BLOCKQUOTE.sub("", text)
    text = MD_LIST.sub("", text)
    return MD_EMPHASIS.sub(r"\2", text)


def collapse_whitespace(text: str) -> str:
    text = WHITESPACE.sub(" ", text)
    return BLANK_LINES.sub("\n\n", text).strip()


def unicode_fold(text: str) -> str:
    """NFKC + zero-width strip + confusables fold.

    Shared with the attack defence in `attacks.lexical` rather than reimplemented, so the
    two can never drift into disagreeing about what "normalised" means.
    """
    from .attacks.lexical import normalize_unicode

    return normalize_unicode(text)


def prose_only(text: str) -> str:
    """The composite a real pipeline usually applies: front matter, code, markdown, space.

    Included because sensitivity to a *combination* is what a deployment actually
    experiences — nobody applies exactly one of these.
    """
    return collapse_whitespace(strip_markdown(strip_code(strip_front_matter(text))))


_REGISTRY: dict[str, Preprocessor] = {}


def register(p: Preprocessor) -> None:
    if p.name in _REGISTRY:
        raise ValueError(f"preprocessor already registered: {p.name}")
    _REGISTRY[p.name] = p


def available() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str) -> Preprocessor:
    if name not in _REGISTRY:
        raise KeyError(f"unknown preprocessor {name!r}; available: {', '.join(available())}")
    return _REGISTRY[name]


for _name, _fn, _doc in [
    ("raw", identity, "No transformation. The baseline every variant is compared against."),
    ("strip_front_matter", strip_front_matter, "Drop YAML front matter."),
    ("strip_code", strip_code, "Drop fenced, indented, and inline code."),
    ("strip_urls", strip_urls, "Drop bare URLs."),
    ("strip_markdown", strip_markdown, "Reduce markdown syntax to prose."),
    ("collapse_whitespace", collapse_whitespace, "Collapse runs of spaces and blank lines."),
    ("unicode_fold", unicode_fold, "NFKC, zero-width strip, confusables fold."),
    ("prose_only", prose_only, "The usual composite: front matter + code + markdown + space."),
]:
    register(_Fn(_name, _fn, _doc))
