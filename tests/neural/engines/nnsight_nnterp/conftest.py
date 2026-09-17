"""Fixtures for the nnterp engine's parity suite.

Both engines load the same checkpoints in fp32 with eager attention on the
CPU, so parity is like-against-like: any disagreement is an executor bug,
not a kernel or dtype story. The whole directory skips when the ``nnsight``
extra (nnsight + nnterp) is not installed.

Placement needs no guard here: the loader passes the requested device as the
pipeline factory's own ``device`` argument, which is honoured on every
platform (``device_map`` alone is ignored at dispatch on MPS).
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import pytest
import torch

pytest.importorskip("nnsight")
pytest.importorskip("nnterp")

from causalab.neural.engines.nnsight_nnterp.executor import (  # noqa: E402
    NnterpExecutor,
)
from causalab.neural.engines.nnsight_nnterp.loading import (  # noqa: E402
    NnterpBundle,
)
from causalab.neural.engines.nnsight_nnterp.loading import (  # noqa: E402
    load_model as load_nnterp_model,
)
from causalab.neural.engines.pytorch_hooks.executor import (  # noqa: E402
    PointExecutor,
)
from causalab.neural.engines.pytorch_hooks.loading import (  # noqa: E402
    ModelBundle,
)
from causalab.neural.engines.pytorch_hooks.loading import (  # noqa: E402
    load_model as load_hooks_model,
)
from causalab.protocol.schema import parse_document  # noqa: E402
from causalab.protocol.validate import validate_document  # noqa: E402

from tests._helpers import a3b_sweep as sweep  # noqa: E402
from tests._helpers.faithful_server import FaithfulServer  # noqa: E402
from tests.protocol._docs import in_order  # noqa: E402

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_QWEN35_MOE = "tiny-random/qwen3.5-moe"
#: The other registered tree: a tuple-returning mixer under a renamed child
#: (nnterp's ``self_attn`` for ``attn``), a fused ``c_attn``, absolute
#: positions.
TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"
#: A tree no registered family detects, which nnterp standardizes.
TINY_GPT_NEOX = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"

#: Two rows, so a batch axis is exercised rather than assumed away. 📐 Each
#: base/counterfactual pair tokenizes to the same length on both fixtures'
#: tokenizers (7 and 6 on Llama's, 6 and 5 on Qwen's), so a whole-tensor
#: swap (the attention pattern) has an operand of the tap's shape — while
#: the two rows differ, so a batch is padded and a whole-sequence read is
#: ragged.
BASE_TEXTS = ["the cat sat on the mat", "the tall man walked home"]
CF_TEXTS = ["a dog ran in the park", "a young girl ran fast"]
ROWS = [
    {"input": base, "counterfactual_inputs": [cf]}
    for base, cf in zip(BASE_TEXTS, CF_TEXTS)
]

#: The same shape on the tiny GPT-2's near-character tokenizer: 📐 each pair
#: is 10 and 11 tokens long, the rows differ.
GPT2_CF_TEXTS = ["the dog sat on the rug", "the old man walked home"]
GPT2_ROWS = [
    {"input": base, "counterfactual_inputs": [cf]}
    for base, cf in zip(BASE_TEXTS, GPT2_CF_TEXTS)
]


@pytest.fixture(scope="session", autouse=True)
def _pinned_torch_threads():
    """Under xdist, give each worker its share of the cores. torch's intra-op
    pool defaults to every core, so N workers run N × cores threads over
    matmuls a few rows wide and spend the run contending. Both engines of a
    comparison run in one worker under one pin, so exact parity holds at any
    thread count."""
    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "0"))
    if workers:
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 1) // workers)))


#: What two *different formulations* of one tensor agree to in fp32 — the only
#: two comparisons of the suite that are not bit-exact, each stating its cause
#: where it passes this: the reference engine's per-step DeltaNet state (the
#: recurrent kernel) against this engine's per-chunk one (the chunked kernel),
#: and the head applied to a whole frame against the head applied to its
#: gathered last rows (a GEMM blocks by its row count). 📐 measured: 3.4e-08
#: and 6.0e-08 on the fixtures.
FORMULATION_ATOL = 1e-6


def assert_same(a: torch.Tensor, b: torch.Tensor, what: str, *, atol: float = 0.0):
    """:func:`sweep.assert_same`, **exactly**: the two engines run the same
    fp32 eager kernels on the same frames in one process — a prompt forward
    without a KV cache on both sides, a generated one with it on both — so
    they agree bit for bit on every fixture family (tiny Llama, tiny GPT-2,
    ``tiny-random/qwen3.5-moe``), read and written, and any difference is an
    executor bug. ``sweep.ATOL`` stays the band of the suites that share that
    helper."""
    sweep.assert_same(a, b, what, atol=atol)


@pytest.fixture(scope="session")
def hooks_llama() -> ModelBundle:
    return load_hooks_model(TINY_LLAMA)


@pytest.fixture(scope="session")
def nnterp_llama() -> NnterpBundle:
    # eager pinned explicitly: the reference engine loads eager, and parity
    # must compare like against like (§7.3); eager is also what enables
    # nnterp's probability accessor, the pattern write's landing
    return load_nnterp_model(TINY_LLAMA, attn_implementation="eager")


@pytest.fixture(scope="session")
def hooks_qwen() -> ModelBundle:
    return load_hooks_model(TINY_QWEN35_MOE)


@pytest.fixture(scope="session")
def nnterp_qwen() -> NnterpBundle:
    return load_nnterp_model(TINY_QWEN35_MOE, attn_implementation="eager")


@pytest.fixture(scope="session")
def hooks_gpt2() -> ModelBundle:
    return load_hooks_model(TINY_GPT2)


@pytest.fixture(scope="session")
def nnterp_gpt2() -> NnterpBundle:
    return load_nnterp_model(TINY_GPT2, attn_implementation="eager")


@pytest.fixture(scope="session")
def nnterp_llama_default_impl() -> NnterpBundle:
    """No pin, so the checkpoint's own default (sdpa) — what the engine's
    loader gives a real document, and what the on-demand eager switch is
    tested against."""
    return load_nnterp_model(TINY_LLAMA)


@pytest.fixture(scope="session")
def remote_llama() -> NnterpBundle:
    """The production client: the weight-free bundle, parameters on meta."""
    return load_nnterp_model(TINY_LLAMA, remote=True)


@pytest.fixture
def ndif_llama(
    nnterp_llama_default_impl: NnterpBundle, monkeypatch: pytest.MonkeyPatch
) -> FaithfulServer:
    """An in-process NDIF serving the tiny Llama under the checkpoint's own
    attention (sdpa) — a model the client bundle shares nothing with
    (``tests/_helpers/faithful_server.py``)."""
    return FaithfulServer(nnterp_llama_default_impl.model, monkeypatch)


#: Each fixture family: its two bundles' fixture names and the rows whose
#: base/counterfactual pairs tokenize to equal lengths on its tokenizer.
FAMILIES: dict[str, tuple[str, str, list[dict[str, Any]]]] = {
    "llama": ("hooks_llama", "nnterp_llama", ROWS),
    "gpt2": ("hooks_gpt2", "nnterp_gpt2", GPT2_ROWS),
    "qwen": ("hooks_qwen", "nnterp_qwen", ROWS),
}
DENSE = ("llama", "gpt2")


@dataclasses.dataclass(frozen=True)
class Family:
    """One checkpoint through both engines."""

    name: str
    hooks: Any
    nnterp: Any
    rows: list[dict[str, Any]]

    def hooked(self, doc: dict[str, Any], *, with_cf: bool) -> PointExecutor:
        return sweep.make_executor(
            PointExecutor, doc, self.hooks, rows=self.rows, with_cf=with_cf
        )

    def traced(self, doc: dict[str, Any], *, with_cf: bool) -> NnterpExecutor:
        return sweep.make_executor(
            NnterpExecutor, doc, self.nnterp, rows=self.rows, with_cf=with_cf
        )

    def read_both(self, component, layer, *, pos=None, head=None):
        doc = sweep.read_doc(
            component,
            layer,
            pos=sweep.default_pos(component) if pos is None else pos,
            head=head,
        )
        hooked = self.hooked(doc, with_cf=False).read_value("r")
        traced = self.traced(doc, with_cf=False).read_value("r")
        return hooked, traced

    def unpatched_logits(self) -> torch.Tensor:
        """The nnterp engine's clean last-position logits — the
        anti-vacuity reference every write is compared against."""
        return self.traced(sweep.read_doc("lm_head", None), with_cf=False).read_value(
            "r"
        )


def read_doc(
    component: str, layer: int, *, pos: object = -1, extra: dict | None = None
) -> dict[str, Any]:
    """:func:`sweep.read_doc` at ``layer``, plus one saved read per ``extra``
    entry — ``name: (site, pos)`` — so several tensors of one forward are
    read in one trace."""
    doc = sweep.read_doc(component, layer, pos=pos)
    for name, (site, npos) in (extra or {}).items():
        doc["method"]["sites"][f"{name}_site"] = site
        doc["method"]["reads"][name] = {
            "site": f"{name}_site",
            "pos": npos,
            "model": "original",
            "input": "base",
        }
        doc["method"]["save"].append(
            {
                "value": name,
                "model": "original",
                "input": "base",
                "file_path": f"{name}.safetensors",
            }
        )
    return doc


def single_row(bundle: NnterpBundle, doc_raw: dict, text: str) -> NnterpExecutor:
    """A one-example executor — dense, so whole-frame identities can reshape
    a ``pos: "all"`` read (the two-row harness's uneven lengths would make
    it ragged), and free to run a text of any length."""
    doc = parse_document(in_order(doc_raw))
    validate_document(doc, engine_is_local=True)
    return NnterpExecutor(
        doc,
        bundle,
        role_rows={"base": [{"input": text}]},
        role_fields={"base": "input"},
        load_tensors=lambda path: (_ for _ in ()).throw(KeyError(path)),
    )


def _family(request: pytest.FixtureRequest, name: str) -> Family:
    hooks_name, nnterp_name, rows = FAMILIES[name]
    return Family(
        name,
        request.getfixturevalue(hooks_name),
        request.getfixturevalue(nnterp_name),
        rows,
    )


@pytest.fixture
def family(request: pytest.FixtureRequest) -> Family:
    """The family named by the test's (indirect) parameter."""
    return _family(request, request.param)
