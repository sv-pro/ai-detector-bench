"""Attack registry. Same lazy-factory shape as the detector registry."""

from __future__ import annotations

from typing import Callable

from .base import Attack, AttackResult, edit_rate

_REGISTRY: dict[str, Callable[..., Attack]] = {}


def register(key: str, factory: Callable[..., Attack]) -> None:
    if key in _REGISTRY:
        raise ValueError(f"attack already registered: {key}")
    _REGISTRY[key] = factory


def available() -> list[str]:
    return sorted(_REGISTRY)


def build(key: str, **kwargs) -> Attack:
    if key not in _REGISTRY:
        raise KeyError(f"unknown attack {key!r}; available: {', '.join(available())}")
    return _REGISTRY[key](**kwargs)


def _homoglyph(**kw):
    from .lexical import HomoglyphAttack

    return HomoglyphAttack(**kw)


def _zero_width(**kw):
    from .lexical import ZeroWidthAttack

    return ZeroWidthAttack(**kw)


def _synonym(**kw):
    from .lexical import SynonymAttack

    return SynonymAttack(**kw)


register("homoglyph", _homoglyph)
register("zero_width", _zero_width)
register("synonym", _synonym)

__all__ = ["Attack", "AttackResult", "edit_rate", "register", "available", "build"]
