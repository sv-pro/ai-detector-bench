"""Detector registry.

Registration is lazy — a factory, not an instance — so that listing the available
detectors never triggers a multi-gigabyte model download. `build` is the only thing that
constructs one.
"""

from __future__ import annotations

from typing import Callable

from ..core import Detector

_REGISTRY: dict[str, Callable[..., Detector]] = {}


def register(key: str, factory: Callable[..., Detector]) -> None:
    if key in _REGISTRY:
        raise ValueError(f"detector already registered: {key}")
    _REGISTRY[key] = factory


def available() -> list[str]:
    return sorted(_REGISTRY)


def build(key: str, **kwargs) -> Detector:
    if key not in _REGISTRY:
        raise KeyError(f"unknown detector {key!r}; available: {', '.join(available())}")
    return _REGISTRY[key](**kwargs)


def _stylometric(**kw):
    from .stylometric import StylometricDetector

    return StylometricDetector(**kw)


def _binoculars(**kw):
    from .binoculars import BinocularsDetector

    return BinocularsDetector(**kw)


def _binoculars_small(**kw):
    from .binoculars import SMALL_PAIR, BinocularsDetector

    kw.setdefault("observer", SMALL_PAIR[0])
    kw.setdefault("performer", SMALL_PAIR[1])
    return BinocularsDetector(**kw)


def _fast_detectgpt(**kw):
    from .fast_detectgpt import FastDetectGPTDetector

    return FastDetectGPTDetector(**kw)


register("stylometric", _stylometric)
register("binoculars", _binoculars)
register("binoculars-small", _binoculars_small)
register("fast-detectgpt", _fast_detectgpt)

__all__ = ["register", "available", "build"]
