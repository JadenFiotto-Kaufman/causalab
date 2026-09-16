"""``normalized_cache`` keys on bound arguments: every spelling of one call is
one entry, distinct arguments are distinct entries, and the cache keeps
``lru_cache``'s public surface."""

from __future__ import annotations

import inspect

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from causalab.neural.shared.normalized_cache import (
    CacheInfo,
    NormalizedCache,
    normalized_cache,
)

pytestmark = pytest.mark.unit


def _counted(maxsize: int = 4, **keys):
    calls: list[tuple] = []

    @normalized_cache(maxsize=maxsize, keys=keys or None)
    def build(key: str, revision: str = "main", *, dtype: str = "fp32", options=None):
        """Build one thing."""
        calls.append((key, revision, dtype, options))
        return object()

    return build, calls


class TestSpellings:
    def test_every_spelling_of_one_call_is_one_entry(self) -> None:
        build, calls = _counted()
        first = build("k")
        assert build(key="k") is first
        assert build("k", "main") is first
        assert build("k", revision="main", dtype="fp32") is first
        assert build("k", options=None) is first
        assert len(calls) == 1
        assert build.cache_info() == CacheInfo(hits=4, misses=1, maxsize=4, currsize=1)

    @pytest.mark.parametrize(
        "difference",
        [{"revision": "other"}, {"dtype": "bf16"}, {"options": ("a",)}],
    )
    def test_a_different_bound_value_is_a_different_entry(self, difference) -> None:
        build, calls = _counted()
        first = build("k")
        assert build("k", **difference) is not first
        assert len(calls) == 2

    @settings(max_examples=60, deadline=None)
    @given(
        key=st.text(min_size=1, max_size=4),
        revision=st.sampled_from(["main", "other"]),
        dtype=st.sampled_from(["fp32", "bf16"]),
        spelling=st.lists(
            st.sampled_from(["pos_rev", "kw_key", "kw_rev", "explicit_dtype"])
        ),
    )
    def test_the_key_depends_only_on_the_bound_values(
        self, key, revision, dtype, spelling
    ) -> None:
        build, _ = _counted()
        canonical = build.key_of(key, revision, dtype=dtype)
        args: list = [key]
        kwargs: dict = {"dtype": dtype}
        if "kw_key" in spelling:
            args = []
            kwargs["key"] = key
        if "pos_rev" in spelling and "kw_key" not in spelling:
            args.append(revision)
        else:
            kwargs["revision"] = revision
        if (
            revision == "main"
            and "kw_rev" not in spelling
            and "pos_rev" not in spelling
        ):
            kwargs.pop("revision", None)  # omitted default
        if dtype == "fp32" and "explicit_dtype" not in spelling:
            kwargs.pop("dtype")
        assert build.key_of(*args, **kwargs) == canonical


class TestKeyFunctions:
    def test_a_key_function_canonicalizes_for_the_key_only(self) -> None:
        build, calls = _counted(
            options=lambda o: None if o is None else tuple(sorted(o))
        )
        first = build("k", options=["b", "a"])
        assert build("k", options=["a", "b"]) is first
        assert build("k", options=("b", "a")) is first
        assert calls == [("k", "main", "fp32", ["b", "a"])], (
            "the function saw the caller's own value, not the canonical key"
        )

    def test_an_unhashable_value_without_a_key_function_is_refused_at_the_call(
        self,
    ) -> None:
        build, _ = _counted()
        with pytest.raises(
            TypeError, match="options=\\{'a': 1\\} is not hashable.*keys="
        ):
            build("k", options={"a": 1})

    def test_a_key_function_for_an_unknown_parameter_is_refused_at_decoration(
        self,
    ) -> None:
        def build(key: str) -> str:
            return key

        with pytest.raises(TypeError, match="no parameter named \\['nope'\\]"):
            normalized_cache(maxsize=1, keys={"nope": id})(build)

    def test_a_keyword_variadic_function_is_refused_at_decoration(self) -> None:
        def build(key: str, **extra: object) -> str:
            return key

        with pytest.raises(TypeError, match="\\*\\*extra"):
            normalized_cache(maxsize=1)(build)

    def test_maxsize_below_one_is_refused(self) -> None:
        with pytest.raises(TypeError, match="maxsize"):
            normalized_cache(maxsize=0)(lambda: None)


class TestLifecycle:
    def test_least_recently_used_entry_is_evicted(self) -> None:
        build, calls = _counted(maxsize=2)
        a = build("a")
        b = build("b")
        assert build("a") is a  # a is now the most recent
        c = build("c")  # evicts b
        assert build("a") is a
        assert build("c") is c
        assert build("b") is not b
        assert len(calls) == 4
        assert build.cache_info().currsize == 2

    def test_cache_clear_forgets_entries_and_counters(self) -> None:
        build, calls = _counted()
        first = build("k")
        build("k")
        build.cache_clear()
        assert build.cache_info() == CacheInfo(0, 0, 4, 0)
        assert build("k") is not first
        assert len(calls) == 2

    def test_renewed_is_an_empty_cache_over_the_same_function_and_keys(self) -> None:
        build, _ = _counted(options=lambda o: o if o is None else tuple(o))
        first = build("k", options=["x"])
        fresh = build.renewed()
        assert isinstance(fresh, NormalizedCache)
        assert fresh.cache_info() == CacheInfo(0, 0, 4, 0)
        renewed_first = fresh("k", options=["x"])
        assert renewed_first is not first, "a renewed cache shares no entries"
        assert fresh("k", options=("x",)) is renewed_first, "but the same key functions"
        assert build("k", options=["x"]) is first, "and the original is untouched"

    def test_the_wrapped_function_and_its_metadata_are_kept(self) -> None:
        build, _ = _counted()
        assert build.__wrapped__.__name__ == "build"
        assert build.__name__ == "build"
        assert build.__doc__ == "Build one thing."
        assert list(inspect.signature(build).parameters) == [
            "key",
            "revision",
            "dtype",
            "options",
        ]
