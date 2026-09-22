from __future__ import annotations

from gradpath.sources.adapters.base import InstitutionAdapter

_REGISTRY: dict[str, type[InstitutionAdapter]] = {}


def register(adapter_cls: type[InstitutionAdapter]) -> type[InstitutionAdapter]:
    _REGISTRY[adapter_cls.slug] = adapter_cls
    return adapter_cls


def get_adapter(slug: str) -> type[InstitutionAdapter] | None:
    return _REGISTRY.get(slug)


def all_adapters() -> list[type[InstitutionAdapter]]:
    return list(_REGISTRY.values())


# Importing the modules is what registers them. A genuine ImportError here
# (a typo, a bad import inside an adapter module) must raise loudly: swallowing
# it would silently drop that institution's adapter and let resolution fall
# through to the weaker generic crawler with no visible symptom.
from gradpath.sources.adapters import gist, kaist, snu  # noqa: F401
