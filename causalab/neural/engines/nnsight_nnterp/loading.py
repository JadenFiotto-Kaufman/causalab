"""Model loading for the nnsight + nnterp engine.

:func:`load_model` builds one :class:`nnterp.StandardizedTransformer` per
realization and wraps it in an :class:`NnterpBundle` exposing the surface the
shared site map and executor base consume — ``model`` / ``tokenizer`` /
``info`` / ``adapter`` / ``blocks`` / ``stream_at`` / ``mixer_at`` /
``streams`` — so :func:`causalab.neural.shared.sites.resolve_site` addresses
the standardized envoy tree exactly as it addresses the reference engine's
module tree. The ``module`` a resolved site carries is an *envoy*, whose
``.input`` / ``.output`` the executor reads and assigns.

Placement is explicit: the requested ``device`` is passed as the pipeline
factory's own ``device`` argument, and as ``device_map`` too so the two
spellings agree (nnterp defaults ``device_map`` to ``"auto"``). 📐
``device_map`` alone is not enough on dev nnsight — the factory then lands
the weights on ``cuda:0`` when one is present — and a model that lands
elsewhere than the bundle says would compare one device's numerics against
another's.

Attention runs under the checkpoint's default implementation unless the
caller pins one. The pattern read and the attention interior need the eager
path (the mixer returns its weights, and the scores exist, only there),
which the block switches on around a forward that needs it.

A **remote** bundle (``load_model(..., remote=True)``) keeps the checkpoint
off the client: the forwards run on NDIF, and what is built here is the
structure alone. nnterp is told ``remote=True, dispatch=False,
allow_dispatch=False, check_attn_probs_with_trace=False`` — 📐 its default
attention-probability check otherwise runs a trace *without* ``remote=``,
which dispatches the whole checkpoint locally, and its scan fallback would
fire a real NDIF job at load time. The parameters stay on ``meta``; device
placement and the CPU kernel binding are skipped (no forward runs here).
Everything downstream is structural and works on the meta tree:
``ModelInfo`` comes from the config, the adapter from the module tree,
``resolve_site`` from attribute walks, and the encoding from the tokenizer.
The bundle records ``device="cpu"`` — where the client finishes the slices
the server gathered.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from causalab.neural.engines.nnsight_nnterp.adapter import standard_adapter
from causalab.neural.shared import streams
from causalab.neural.shared.compile_cache import configure as configure_compile_cache
from causalab.neural.shared.kernels import bind_kernel_path
from causalab.neural.shared.loading import TORCH_DTYPES, torch_module
from causalab.neural.shared.normalized_cache import normalized_cache
from causalab.protocol.registry import (
    FamilyAdapter,
    ModelInfo,
    model_info_from_hf_config,
    register_model,
)

__all__ = ["NnterpBundle", "load_model"]


@dataclasses.dataclass(frozen=True)
class NnterpBundle:
    """One loaded standardized model with everything the executor needs.

    ``model`` is the :class:`nnterp.StandardizedTransformer`; ``adapter`` is
    the per-model :class:`~causalab.protocol.registry.FamilyAdapter` over its
    standard tree (:mod:`.adapter`), which the shared resolver prefers over
    family detection (``sites.adapter_of``).
    """

    key: str
    revision: str
    model: Any
    tokenizer: Any
    info: ModelInfo
    adapter: FamilyAdapter
    #: The device the weights were placed on and inputs are sent to.
    device: str
    dtype: str
    quantization: dict[str, Any] | None = None
    #: Whether the weights are elsewhere: a remote bundle's parameters are on
    #: ``meta``, and its executor runs every forward on NDIF.
    remote: bool = False

    @property
    def blocks(self) -> Any:
        """The decoder-layer list of envoys, under the standard name."""
        return self.adapter.blocks_of(self.model)

    def stream_at(self, layer: int) -> str:
        """Which mixer stream ``layer`` carries — the shared table's answer,
        read off the standard tree's two mixer names."""
        return streams.stream_at(
            self.blocks, layer, key=self.key, mixers=self.adapter.mixers
        )

    def mixer_at(self, layer: int) -> Any:
        """The mixer envoy at ``layer``, whichever stream it is."""
        return streams.mixer_at(
            self.blocks, layer, key=self.key, mixers=self.adapter.mixers
        )

    @property
    def streams(self) -> tuple[str, ...]:
        return tuple(self.stream_at(i) for i in range(len(self.blocks)))


@normalized_cache(maxsize=4)
def load_model(
    key: str,
    revision: str = "main",
    *,
    dtype: str = "fp32",
    device: str = "cpu",
    attn_implementation: str | None = None,
    remote: bool = False,
) -> NnterpBundle:
    """Load (and cache) one standardized bundle.

    The tokenizer is set to the engines' single padding convention — left
    padding, ``pad = eos`` when the checkpoint ships none — and the weights
    are frozen in eval mode, as the reference loader prepares them. Four
    bundles stay resident, keyed on the bound arguments
    (:func:`~causalab.neural.shared.normalized_cache.normalized_cache`).

    ``attn_implementation=None`` keeps the checkpoint's default; ``"eager"``
    also enables nnterp's attention-probability accessor.

    ``remote=True`` builds the weight-free bundle of the module docstring:
    ``device`` is then where the client finishes values, and must be the CPU.
    """
    from nnterp import StandardizedTransformer

    if remote and device != "cpu":
        raise ValueError(
            f"a remote bundle holds no weights to place on {device!r}: its "
            'forwards run on NDIF, and the client finishes values on "cpu"'
        )
    if not remote:
        configure_compile_cache(device)
    eager = attn_implementation == "eager"
    placement: dict[str, Any] = (
        {
            "remote": True,
            "dispatch": False,
            "allow_dispatch": False,
            "check_attn_probs_with_trace": False,
        }
        if remote
        else {"device": device, "device_map": device, "dispatch": True}
    )
    model = StandardizedTransformer(
        key,
        revision=revision,
        dtype=TORCH_DTYPES[dtype],
        # nnterp passes attn_implementation="eager" itself when it enables the
        # accessor, and a second spelling of it is a duplicate keyword
        **(
            {}
            if attn_implementation is None or eager
            else {"attn_implementation": attn_implementation}
        ),
        enable_attention_probs=eager,
        check_renaming=True,
        **placement,
    )
    raw = torch_module(model)
    raw.eval()
    # only featurizer/free params ever train (§2.11); freezing the network
    # keeps a training graph from accumulating gradients into model weights
    raw.requires_grad_(False)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    info = model_info_from_hf_config(key, model.config)
    register_model(info)
    # a DeltaNet family's kernel globals follow the device the weights are on
    # (shared/kernels.py); the executor wraps each trace in the torch path too
    if not remote:
        bind_kernel_path(raw, on_cuda=device.startswith("cuda"))
    return NnterpBundle(
        key=key,
        revision=revision,
        model=model,
        tokenizer=tokenizer,
        info=info,
        adapter=standard_adapter(raw),
        device=device,
        dtype=dtype,
        remote=remote,
    )
