"""Corpora. The shipped fixture is a smoke test; real evaluation sets are external."""

from .fixtures import SMOKE_WARNING, Document, load_smoke

__all__ = ["Document", "load_smoke", "SMOKE_WARNING"]
