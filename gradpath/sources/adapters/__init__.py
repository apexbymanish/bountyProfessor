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


# Importing the modules is what registers them. These land in Task 11; until
# then this package must still import cleanly (Task 10 ships with no
# adapters registered, which tests/test_email_chain.py's
# test_registered_adapters_satisfy_the_contract documents as an expected,
# cross-task failure).
try:
    from gradpath.sources.adapters import gist, kaist, snu  # noqa: F401
except ImportError:
    pass
