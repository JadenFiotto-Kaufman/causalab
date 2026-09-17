"""The address-table CI canary.

One trace per anchor of each tree resolves **every** table entry — match,
drill, handle, selection — and fails with the op-inventory diff on any
miss. It has to *run a trace*, not parse: recursive ``.source`` drilling
only exists inside one. This is the tripwire for a transformers bump — the
``uv.lock`` revision is the real pin, and their own history shows why
(transformers 5 renamed GPT-2's dropout op and broke nnterp's address) —
and the artifact that travels upstream with ``sources.py`` later.

Plus the upstreaming discipline itself: ``sources.py`` imports nothing from
the rest of ``causalab``, pinned by a subprocess import so this suite's own
imports cannot mask a violation.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from causalab.neural.engines.nnsight_nnterp.executor import NnterpExecutor
from causalab.neural.engines.nnsight_nnterp.landers import (
    Navigation,
    fire_ops,
    present_native,
    routing,
)
from causalab.neural.engines.nnsight_nnterp.sources import (
    ADDRESSES,
    GENERATED_ADDRESSES,
    AddressResolutionError,
    match_op,
)
from causalab.neural.shared.kernels import torch_kernel_path
from causalab.neural.shared.loading import torch_module
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.plan import COMPONENT_RANK
from causalab.protocol.registry import CAPABILITIES
from causalab.protocol.schema import SiteSpec

from tests._helpers import a3b_sweep as sweep
from tests.neural.engines.nnsight_nnterp.conftest import ROWS

pytestmark = pytest.mark.smoke

TEXT = "the quick brown fox jumps"


def _layer_for(bundle, component: str) -> int:
    """A layer carrying the stream ``component`` needs; any layer otherwise."""
    stream = CAPABILITIES[component].stream
    return bundle.streams.index(stream) if stream is not None else 0


@pytest.mark.parametrize("fixture", ["nnterp_qwen", "nnterp_gpt2"])
def test_every_table_entry_resolves_in_one_trace_per_anchor(request, fixture):
    """The canary proper. Navigation reuses the block's own drill,
    memo and presentation — the same code path a document takes — so a
    green canary means real documents resolve, not merely that the strings
    match. Entries are requested in rank order, the discipline the
    executor's schedule enforces; a per-fire entry resolves its trip count
    and its first fire."""
    import nnsight

    bundle = request.getfixturevalue(fixture)
    executor = sweep.make_executor(
        NnterpExecutor,
        sweep.read_doc("block_output", 0),
        bundle,
        rows=ROWS,
        with_cf=False,
    )
    entries = sorted(
        (
            (component, address)
            for (tree, component), address in ADDRESSES.items()
            if tree == executor._tree
        ),
        key=lambda item: COMPONENT_RANK[item[0]],
    )
    assert entries, f"no {executor._tree!r} rows in the table"
    sites = {
        component: resolve_site(
            bundle,
            SiteSpec(component=component, layers=(_layer_for(bundle, component),)),
        )
        for component, _ in entries
    }
    by_anchor: dict[int, list] = {}
    for component, address in entries:
        by_anchor.setdefault(id(sites[component].module), []).append(
            (component, address)
        )
    for anchor_entries in by_anchor.values():
        for component, _ in anchor_entries:
            _ = sites[component].module.source  # instrumented before the forward
    saves: dict[str, object] = {}
    for anchor_entries in by_anchor.values():
        nav = Navigation(bundle.key)
        with torch.no_grad(), torch_kernel_path(torch_module(bundle.model)):
            with bundle.model.trace(TEXT):
                if any(address.expert_rows for _, address in anchor_entries):
                    # the routing table is the anchor's own input, requested
                    # at its entry before anything inside it
                    anchor_site = sites[anchor_entries[0][0]]
                    saves["routing"] = nnsight.save(routing(anchor_site, nav, 1))
                for component, address in anchor_entries:
                    site = sites[component]
                    if address.fires == "per_chunk":
                        value_op, count = fire_ops(site, address, nav)
                        saves[f"{component}:trip"] = nnsight.save(count)
                        saves[component] = nnsight.save(value_op.output)
                        continue
                    saves[component] = nnsight.save(present_native(site, address, nav))
    assert set(saves) >= {component for component, _ in entries}
    for component, value in saves.items():
        if component.endswith(":trip"):
            assert int(value) >= 1, component
        else:
            assert isinstance(value, torch.Tensor) and value.numel() > 0, component
    if executor._tree == "llama_tree":
        assert tuple(saves["routing"].shape[:2]) == (
            1,
            len(bundle.tokenizer(TEXT)["input_ids"]),
        )


def test_every_decode_table_entry_resolves_per_step(nnterp_qwen):
    """The decode half of the canary: each generated-frame address resolves
    inside a ``model.generate`` trace, one value per decode step, with the
    body pinned to its forward by the embedding's input (the op has no
    prefill occurrence, so an unpinned loop would outrun it)."""
    import nnsight

    executor = sweep.make_executor(
        NnterpExecutor,
        sweep.read_doc("block_output", 0),
        nnterp_qwen,
        rows=ROWS,
        with_cf=False,
    )
    entries = [
        (component, address)
        for (tree, component), address in GENERATED_ADDRESSES.items()
        if tree == executor._tree
    ]
    assert entries
    sites = {
        component: resolve_site(
            nnterp_qwen,
            SiteSpec(component=component, layers=(_layer_for(nnterp_qwen, component),)),
        )
        for component, _ in entries
    }
    for site in sites.values():
        _ = site.module.source
    embedding = resolve_site(nnterp_qwen, SiteSpec(component="embeddings")).module
    steps = 3
    with torch.no_grad(), torch_kernel_path(torch_module(nnterp_qwen.model)):
        with nnterp_qwen.model.generate(
            TEXT, max_new_tokens=steps + 1, do_sample=False, eos_token_id=None
        ) as tracer:
            saves = nnsight.save({component: [] for component, _ in entries})
            for _ in tracer.iter[1 : steps + 1]:
                _ = embedding.input
                nav = Navigation(nnterp_qwen.key)
                for component, address in entries:
                    saves[component].append(
                        present_native(sites[component], address, nav)
                    )
            _ = tracer.result.save()
    for component, values in saves.items():
        assert len(values) == steps, component
        assert all(isinstance(v, torch.Tensor) and v.numel() > 0 for v in values)


def test_a_missing_pattern_refuses_with_the_inventory():
    with pytest.raises(AddressResolutionError) as excinfo:
        match_op("nonexistent_op", ["real_op_0", "other_op_1"])
    message = str(excinfo.value)
    assert "real_op_0" in message and "other_op_1" in message


def test_an_ambiguous_pattern_refuses_rather_than_guessing():
    """Two hits, neither a call of the symbol: no basis to choose."""
    lines = {"weights_0": "weights = a + b", "weights_1": "weights = weights * c"}
    with pytest.raises(AddressResolutionError, match="2 ops match"):
        match_op("weights", list(lines), lines.__getitem__)


def test_the_call_op_wins_over_the_assignment():
    """The systematic ambiguity — a variable assigned, then called — resolves
    to the call, which is how 'attention_interface' names one op out of two."""
    lines = {
        "attention_interface_0": "attention_interface: Callable = get_interface(",
        "attention_interface_1": "out, weights = attention_interface(",
    }
    assert (
        match_op("attention_interface", list(lines), lines.__getitem__)
        == "attention_interface_1"
    )


def test_sources_imports_nothing_from_causalab():
    """The upstreaming rule, by construction: the table+matcher module must
    stand alone — moving it to nnterp later is then a file move, not a
    rewrite. Loaded by file path in a subprocess (the package ``__init__``
    would drag the engine in), so a green run means the module's own body
    pulled in nothing of causalab at all."""
    import causalab.neural.engines.nnsight_nnterp.sources as sources

    path = sources.__file__
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('sources', {path!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules['sources'] = module  # dataclasses resolves through it\n"
        "spec.loader.exec_module(module)\n"
        "polluted = [m for m in sys.modules if m.startswith('causalab')]\n"
        "assert not polluted, polluted\n"
        "assert module.ADDRESSES  # and it is the real module, the table loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
