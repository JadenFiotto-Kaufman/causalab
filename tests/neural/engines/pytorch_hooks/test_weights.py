"""The reference engine's weight reader (``weights.py``).

What is held here:

* the reader's result is the stock ``from_pretrained``'s, bit for bit, on
  every tiny family — including the one whose checkpoint transformers has to
  convert (``qwen3.5-moe``: per-expert weights fused, ``language_model``
  prefix dropped, a vision tower and an MTP head to leave unread);
* the key census: exactly the text tower is wanted, nothing of the other
  towers is read;
* the refusals: a parameter the reader did not deliver, a tensor in two
  shards, a wanted tensor no shard carries, a file that is not safetensors;
* the plan and the hand-out as pure properties: a partition is a partition,
  every tensor leaves the prefetcher exactly once whatever the order it is
  asked in, and a shard is read exactly once.
"""

from __future__ import annotations

import dataclasses
import json
import struct
import threading
from pathlib import Path
from typing import Any, Sequence

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks import weights
from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.engines.pytorch_hooks.weights import (
    Prefetch,
    SafetensorsReader,
    Shard,
    TensorHeader,
    checkpoint_files,
    load_pretrained,
    read_header,
    shard_plan,
    wanted_keys,
)
from causalab.protocol.errors import ProtocolError

from tests.neural.engines.pytorch_hooks.conftest import (
    TINY_GPT2,
    TINY_LLAMA,
    TINY_QWEN35_MOE,
)

TINY = (TINY_LLAMA, TINY_GPT2, TINY_QWEN35_MOE)


def _stock(key: str) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        key, dtype=torch.float32, attn_implementation="eager"
    )


def _same_state(a: Any, b: Any) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return set(sa) == set(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def _checkpoint_keys(key: str) -> set[str]:
    files = checkpoint_files(key, "main")
    assert files is not None
    return {name for path in files for name in read_header(path)}


# ---------------------------------------------------------------------------
# The reader's result is the stock loader's
# ---------------------------------------------------------------------------


@pytest.mark.property
class TestMatchesStockLoader:
    @pytest.mark.parametrize("key", TINY)
    def test_state_dict_bit_identical(self, key: str) -> None:
        fast = load_pretrained(
            key, "main", dtype=torch.float32, device="cpu", attn_implementation="eager"
        )
        assert _same_state(_stock(key), fast)
        assert {p.device.type for p in fast.parameters()} == {"cpu"}
        assert {p.dtype for p in fast.parameters()} == {torch.float32}
        # what the stock loader records on the config
        assert fast.config.name_or_path == key

    @pytest.mark.parametrize("key", TINY)
    def test_fastersafetensors_reader_bit_identical(self, key: str) -> None:
        fast = load_pretrained(
            key,
            "main",
            dtype=torch.float32,
            device="cpu",
            attn_implementation="eager",
            reader=weights.FastersafetensorsReader(),
        )
        assert _same_state(_stock(key), fast)

    @pytest.mark.parametrize("key", TINY)
    def test_threaded_reader_bit_identical(self, key: str) -> None:
        """The explicit alternative to the default reader."""
        fast = load_pretrained(
            key,
            "main",
            dtype=torch.float32,
            device="cpu",
            attn_implementation="eager",
            reader=SafetensorsReader(),
        )
        assert _same_state(_stock(key), fast)

    def test_default_reader_is_the_planned_read(self) -> None:
        """The inlined Rust reader is the default; the threaded reader is
        only ever passed explicitly."""
        assert isinstance(weights.default_reader(), weights.FastersafetensorsReader)

    def test_load_model_goes_through_the_reader(self) -> None:
        """``load_model`` is the caller; its bundle carries the same weights."""
        bundle = load_model(TINY_LLAMA)
        assert _same_state(_stock(TINY_LLAMA), bundle.model)
        assert not any(p.requires_grad for p in bundle.model.parameters())

    def test_wanted_keys_is_exactly_the_text_tower(self) -> None:
        """The multimodal fixture: the language tower and the head are wanted,
        the vision tower and the MTP head are not — so they are never read."""
        from transformers import AutoConfig, AutoModelForCausalLM

        keys = _checkpoint_keys(TINY_QWEN35_MOE)
        with torch.device("meta"):
            meta = AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(TINY_QWEN35_MOE), attn_implementation="eager"
            )
        wanted = wanted_keys(meta, keys)
        text = {
            k
            for k in keys
            if k.startswith("model.language_model.") or k == "lm_head.weight"
        }
        assert wanted == text
        others = keys - wanted
        assert others and all(
            k.startswith("model.visual.") or k.startswith("mtp.") for k in others
        )


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _DroppingReader:
    """The stock reader, minus one tensor — a reader that under-delivers."""

    drop: str
    concurrency: int = 2

    def read(
        self, path: Path, keys: Sequence[str], device: torch.device
    ) -> dict[str, torch.Tensor]:
        out = SafetensorsReader().read(path, keys, device)
        out.pop(self.drop, None)
        return out


@pytest.mark.unit
class TestRefusals:
    def test_parameter_the_model_wanted_and_did_not_get(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wanted key kept out of the state dict is a parameter transformers
        would initialize at random — refused by name, nothing returned."""
        original = weights.wanted_keys
        monkeypatch.setattr(
            weights,
            "wanted_keys",
            lambda meta, keys: original(meta, keys) - {"model.norm.weight"},
        )
        with pytest.raises(ProtocolError) as err:
            load_pretrained(
                TINY_LLAMA,
                "main",
                dtype=torch.float32,
                device="cpu",
                attn_implementation="eager",
            )
        assert err.value.code == "P2"
        assert "1 missing" in str(err.value)
        assert "model.norm.weight" in str(err.value)

    def test_reader_that_under_delivers_fails_on_that_key(self) -> None:
        """The prefetcher hands each tensor out once; a tensor the reader never
        produced surfaces as the missing key, not as a silent skip."""
        with pytest.raises(KeyError, match="model.norm.weight"):
            load_pretrained(
                TINY_LLAMA,
                "main",
                dtype=torch.float32,
                device="cpu",
                attn_implementation="eager",
                reader=_DroppingReader(drop="model.norm.weight"),
            )

    def test_wanted_tensor_no_shard_carries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = weights.wanted_keys
        monkeypatch.setattr(
            weights,
            "wanted_keys",
            lambda meta, keys: original(meta, keys) | {"model.phantom.weight"},
        )
        with pytest.raises(ProtocolError, match="model.phantom.weight"):
            load_pretrained(
                TINY_LLAMA,
                "main",
                dtype=torch.float32,
                device="cpu",
                attn_implementation="eager",
            )

    def test_tensor_in_two_shards(self) -> None:
        header = {"w": TensorHeader(dtype="F32", shape=(2,))}
        with pytest.raises(ProtocolError, match="both"):
            shard_plan([(Path("a"), header), (Path("b"), header)], frozenset({"w"}))

    def test_not_a_safetensors_file(self, tmp_path: Path) -> None:
        (tmp_path / "x.safetensors").write_bytes(b"\x01\x02")
        with pytest.raises(ProtocolError, match="truncated header"):
            read_header(tmp_path / "x.safetensors")

    def test_checkpoint_without_safetensors_is_none(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}")
        assert checkpoint_files(str(tmp_path), "main") is None

    def test_index_names_the_shards(self, tmp_path: Path) -> None:
        for name in (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ):
            (tmp_path / name).write_bytes(b"")
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    # transformers' index format: `metadata` beside `weight_map`
                    "metadata": {"total_size": 0},
                    "weight_map": {
                        "a": "model-00002-of-00002.safetensors",
                        "b": "model-00001-of-00002.safetensors",
                        "c": "model-00002-of-00002.safetensors",
                    },
                }
            )
        )
        files = checkpoint_files(str(tmp_path), "main")
        assert files == (
            tmp_path / "model-00001-of-00002.safetensors",
            tmp_path / "model-00002-of-00002.safetensors",
        )

    def test_read_header_reads_only_the_table(self, tmp_path: Path) -> None:
        table = {
            "__metadata__": {"format": "pt"},
            "w": {"dtype": "BF16", "shape": [3, 4], "data_offsets": [0, 24]},
        }
        body = json.dumps(table).encode()
        path = tmp_path / "one.safetensors"
        path.write_bytes(struct.pack("<Q", len(body)) + body + b"\0" * 24)
        assert read_header(path) == {"w": TensorHeader(dtype="BF16", shape=(3, 4))}


# ---------------------------------------------------------------------------
# The plan and the hand-out, as properties
# ---------------------------------------------------------------------------

_names = st.text(alphabet="abcdefgh.0123456789", min_size=1, max_size=8)


@st.composite
def _tables(draw: st.DrawFn) -> list[tuple[Path, dict[str, TensorHeader]]]:
    """Files with pairwise-disjoint key sets, as a sharded checkpoint has."""
    pool = sorted(draw(st.sets(_names, min_size=0, max_size=24)))
    n_files = draw(st.integers(min_value=1, max_value=5))
    tables: list[tuple[Path, dict[str, TensorHeader]]] = []
    for i in range(n_files):
        tables.append((Path(f"shard-{i}"), {}))
    for name in pool:
        owner = draw(st.integers(min_value=0, max_value=n_files - 1))
        tables[owner][1][name] = TensorHeader(dtype="F32", shape=(1,))
    return tables


@dataclasses.dataclass
class _RecordingReader:
    """A reader with no disk: each tensor is its key's index in a fixed
    table, so identity is checkable; every call is counted per file."""

    table: dict[str, int]
    concurrency: int = 3
    calls: dict[Path, int] = dataclasses.field(default_factory=dict)
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    def read(
        self, path: Path, keys: Sequence[str], device: torch.device
    ) -> dict[str, torch.Tensor]:
        with self._lock:
            self.calls[path] = self.calls.get(path, 0) + 1
        return {k: torch.tensor([self.table[k]], device=device) for k in keys}


@dataclasses.dataclass
class _RecordingCheckpointReader:
    """A ``CheckpointReader`` with no disk: one ``read_all`` for everything."""

    table: dict[str, int]
    calls: int = 0

    def read_all(
        self, shards: Sequence[Shard], device: torch.device
    ) -> dict[str, torch.Tensor]:
        self.calls += 1
        return {
            k: torch.tensor([self.table[k]], device=device)
            for shard in shards
            for k in shard.keys
        }


@pytest.mark.property
class TestPlanAndHandOut:
    @given(tables=_tables(), data=st.data())
    @settings(max_examples=60, deadline=None)
    def test_whole_checkpoint_reader_is_asked_once(
        self, tables: list[tuple[Path, dict[str, TensorHeader]]], data: st.DataObject
    ) -> None:
        carried = sorted(name for _, table in tables for name in table)
        shards, _ = shard_plan(tables, frozenset(carried))
        reader = _RecordingCheckpointReader(table={n: i for i, n in enumerate(carried)})
        order = data.draw(st.permutations(carried))
        with Prefetch(shards, reader, torch.device("cpu")) as prefetch:
            seen = {name: int(prefetch.take(name).item()) for name in order}
            assert seen == reader.table
            assert reader.calls == (1 if carried else 0)
            for name in carried:
                with pytest.raises(KeyError):
                    prefetch.take(name)

    @given(tables=_tables(), data=st.data())
    @settings(max_examples=200, deadline=None)
    def test_shard_plan_partitions_the_wanted_keys(
        self, tables: list[tuple[Path, dict[str, TensorHeader]]], data: st.DataObject
    ) -> None:
        carried = {name for _, table in tables for name in table}
        extra = data.draw(st.sets(_names, max_size=4))
        wanted = frozenset(
            data.draw(st.sets(st.sampled_from(sorted(carried) or [""]), max_size=24))
            | extra
        )
        wanted = frozenset(w for w in wanted if w)
        shards, absent = shard_plan(tables, wanted)
        planned = [name for shard in shards for name in shard.keys]
        assert len(planned) == len(set(planned))
        assert set(planned) == wanted & carried
        assert absent == wanted - carried
        assert all(shard.keys for shard in shards)
        assert len({shard.path for shard in shards}) == len(shards)
        for shard in shards:
            header_order = [n for n in shard.headers if n in wanted]
            assert list(shard.keys) == header_order

    @given(tables=_tables(), data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_every_tensor_leaves_exactly_once_in_any_order(
        self, tables: list[tuple[Path, dict[str, TensorHeader]]], data: st.DataObject
    ) -> None:
        carried = sorted(name for _, table in tables for name in table)
        shards, _ = shard_plan(tables, frozenset(carried))
        reader = _RecordingReader(table={name: i for i, name in enumerate(carried)})
        order = data.draw(st.permutations(carried))
        with Prefetch(shards, reader, torch.device("cpu")) as prefetch:
            seen: dict[str, int] = {}
            for name in order:
                seen[name] = int(prefetch.take(name).item())
            assert seen == reader.table
            # a shard is read once, and only shards with a wanted key are read
            assert reader.calls == {shard.path: 1 for shard in shards}
            # nothing is left behind, and asking twice is an error, not a copy
            for name in carried:
                with pytest.raises(KeyError):
                    prefetch.take(name)
        # nothing is read for an empty plan either
        if not carried:
            assert reader.calls == {}

    def test_reads_start_on_first_take_not_on_construction(self) -> None:
        """transformers warms its allocator with a model-sized allocation
        right before the first materialization; a reader that had already
        filled the device would collide with it."""
        header = {"w": TensorHeader(dtype="F32", shape=(1,))}
        shards = (Shard(path=Path("s"), keys=("w",), headers=header),)
        reader = _RecordingReader(table={"w": 7})
        with Prefetch(shards, reader, torch.device("cpu")) as prefetch:
            assert reader.calls == {}
            assert int(prefetch.take("w").item()) == 7
            assert reader.calls == {Path("s"): 1}
