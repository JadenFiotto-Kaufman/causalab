"""``load_model`` caches its bundles (``normalized_cache``, four entries) — a
property no type checker or suite saw go missing once when a helper was
inserted between the cache and the function. Pinned here so the next such
edit fails a test."""

from __future__ import annotations

import pytest

from causalab.neural.engines.nnsight_tracing import loading

pytestmark = pytest.mark.unit


def test_load_model_is_a_four_entry_cache() -> None:
    assert hasattr(loading.load_model, "cache_info")
    assert loading.load_model.cache_info().maxsize == 4


def test_the_helper_beside_it_is_a_plain_function() -> None:
    assert not hasattr(loading.torch_module, "cache_info")
