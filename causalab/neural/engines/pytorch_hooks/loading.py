"""Model, tokenizer and tensor-bundle loading for the reference engine.

One bundle per load configuration, including attention backend: the HF
causal-LM, its tokenizer configured for the engine's one padding convention
(**left**-padded, ``pad = eos``), and the model's static metadata
registered into the protocol model registry so canonicalization inside a
run needs no pre-registration.

Attention defaults to **eager** for the captured goldens, but callers can
select another Transformers backend. The executor temporarily uses eager
only for forwards that read or edit attention-function interiors.

Two ways in. :func:`load_model` loads, prepares and caches a model the
library owns. :meth:`ModelBundle.from_model` wraps a model the **caller**
owns (spec §9, the ownership contract): it derives the same registry entry
but mutates nothing — where the loader would prepare the model it *refuses*,
naming the one call the caller must make, and only where the numbers would
otherwise differ from a loaded run. A caller bundle never enters the loader's
cache.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Mapping

import torch

from causalab.neural.engines.pytorch_hooks.weights import load_pretrained
from causalab.neural.shared import streams
from causalab.neural.shared.compile_cache import configure as configure_compile_cache
from causalab.neural.shared.kernels import bind_kernel_path
from causalab.neural.shared.normalized_cache import normalized_cache
from causalab.neural.shared.services import BundlePoint, TensorBundle
from causalab.protocol.errors import ProtocolError
from causalab.protocol.registry import (
    FamilyAdapter,
    ModelInfo,
    family_for,
    model_info_from_hf_config,
    register_model,
)

__all__ = ["BundlePoint", "ModelBundle", "TensorBundle", "load_model"]

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


@dataclasses.dataclass(frozen=True)
class ModelBundle:
    """One loaded model with everything the executor needs."""

    key: str
    revision: str
    model: Any
    tokenizer: Any
    info: ModelInfo
    #: The device that was **requested**, not necessarily where the weights
    #: are. On a quantized load the `.to(device)` below is skipped —
    #: bitsandbytes/accelerate place the weights themselves and moving them
    #: afterwards is refused — so this records the ask, and the real placement
    #: is `next(model.parameters()).device`. Kept as the request because it is
    #: what the executor sends inputs to; a disagreement surfaces as a loud
    #: device-mismatch at the first forward, never as quiet wrong numbers.
    device: str
    dtype: str
    quantization: dict[str, Any] | None = None

    @functools.cached_property
    def adapter(self) -> FamilyAdapter:
        """The registered family whose predicate recognizes this model's
        module tree (``registry.family_for``) — detected once per
        bundle, structurally, never off the config. Everything below that
        used to be a ``hasattr`` on the tree routes through it."""
        return family_for(self.model)

    @property
    def is_gpt2_family(self) -> bool:
        return self.adapter.family == "gpt2_tree"

    @property
    def blocks(self) -> Any:
        """The decoder-layer ModuleList, whichever tree this family uses."""
        return self.adapter.blocks_of(self.model)

    def stream_at(self, layer: int) -> str:
        """Which mixer stream ``layer`` actually carries — delegated to the
        shared table (:mod:`causalab.neural.shared.streams`), because the
        per-layer hybrid answer must never diverge between engines."""
        return streams.stream_at(
            self.blocks, layer, key=self.key, mixers=self.adapter.mixers
        )

    def mixer_at(self, layer: int) -> Any:
        """The attention/mixer module at ``layer``, whichever stream it is."""
        return streams.mixer_at(
            self.blocks, layer, key=self.key, mixers=self.adapter.mixers
        )

    @property
    def streams(self) -> tuple[str, ...]:
        """``stream_at`` for every layer — the whole tower's shape at a glance."""
        return tuple(self.stream_at(i) for i in range(len(self.blocks)))

    @classmethod
    def from_model(
        cls,
        model: Any,
        tokenizer: Any,
        *,
        key: str,
        revision: str,
        device: str,
        dtype: str,
        quantization: Mapping[str, Any] | None = None,
    ) -> "ModelBundle":
        """Wrap a model the **caller** owns — the supported way in for a model
        that is already loaded (spec §9, the ownership contract).

        ``info`` is derived exactly as :func:`load_model` derives it
        (:func:`model_info_from_hf_config` + :func:`register_model`), so the
        engine's tap table reads the same registry row either way. ``key``
        and ``revision`` are the caller's *assertion*: nothing here can check
        them against the weights, and the run receipt stamps them as given.

        Nothing about ``model`` or ``tokenizer`` is mutated. :func:`load_model`
        prepares what it loads — ``.eval()``,
        ``.requires_grad_(False)``, left padding with a pad token — and each
        of those changes the numbers or what the hooks see, so an object that
        lacks one is **refused** with the exact call to make rather than
        quietly re-moded, moved or re-configured behind the caller's back
        (:func:`_refuse_unprepared`). The bundle is never inserted into
        :func:`load_model`'s cache.

        One thing this method does reach beyond the model: when
        ``CAUSALAB_COMPILE_CACHE`` is set, the process's compiler cache
        variables are pointed at the shared root here, as :func:`load_model`
        does for a model it loads (``shared/compile_cache.py``) — a
        caller-owned model compiles its kernels on first use like any other.

        The caller's attention backend is accepted unchanged. During execution,
        forwards needing attention-function interiors temporarily use eager;
        the executor restores the caller's backend on every exit path.

        Raises:
            ProtocolError: the model or tokenizer is not prepared the way a
                loaded one is (one refusal, one call named), or ``dtype`` is
                not a precision the engine knows.
        """
        if dtype not in _DTYPES:
            raise ProtocolError(
                "P4",
                f"dtype {dtype!r} is not one of {sorted(_DTYPES)} — the same "
                "closed set a document's model.dtype draws from (§2.1)",
            )
        _refuse_unprepared(
            model, tokenizer, dtype=dtype, quantized=quantization is not None
        )
        info = model_info_from_hf_config(key, model.config)
        register_model(info)
        # a caller-owned model compiles its kernels on first use like a loaded
        # one; the shared cache root applies the same way
        configure_compile_cache(device)
        return cls(
            key=key,
            revision=revision,
            model=model,
            tokenizer=tokenizer,
            info=info,
            device=device,
            dtype=dtype,
            quantization=dict(quantization) if quantization is not None else None,
        )


def _refuse_unprepared(
    model: Any, tokenizer: Any, *, dtype: str, quantized: bool
) -> None:
    """Refuse a caller-owned model the loader would have had to prepare.

    One check per preparation :func:`load_model` makes on a model it owns, in
    the loader's own order, each naming the one expression the caller runs
    instead — fail closed on an object the library does not own, and only
    where a loaded run's numbers would differ:

    * **eval mode** — dropout and other train-time branches change the
      numbers on every forward;
    * **frozen weights** — a train document accumulates gradients into any
      parameter that requires them (§2.11 trains featurizers only);
    * **the padding convention** — the position frame is built for left
      padding, and encoding needs a pad token;
    * **the declared precision** — a weight whose dtype is not the declared
      one produces a run the record would misdescribe (skipped on a quantized
      model, whose weights are integer tensors by construction).
    """
    how = "prepare the model the way load_model does"
    if getattr(model, "training", False):
        raise ProtocolError(
            "P4",
            f"the caller-owned model is in train mode — {how}: call "
            "model.eval() before handing it over",
        )
    if any(p.requires_grad for p in model.parameters()):
        raise ProtocolError(
            "P4",
            "the caller-owned model has parameters that require grad, and a "
            "train document would accumulate gradients into them — "
            f"{how}: call model.requires_grad_(False) before handing it over",
        )
    if getattr(tokenizer, "padding_side", None) != "left":
        raise ProtocolError(
            "P4",
            f"the caller-owned tokenizer pads on the {tokenizer.padding_side!r}, "
            "and the engine's position frame is built for left padding — "
            f'{how}: set tokenizer.padding_side = "left" before handing it over',
        )
    if getattr(tokenizer, "pad_token", None) is None:
        raise ProtocolError(
            "P4",
            "the caller-owned tokenizer has no pad token, and batches are "
            f"padded — {how}: set tokenizer.pad_token = tokenizer.eos_token "
            "before handing it over",
        )
    if not quantized:
        wanted = _DTYPES[dtype]
        found = {p.dtype for p in model.parameters() if p.is_floating_point()} - {
            wanted
        }
        if found:
            raise ProtocolError(
                "P4",
                f"the bundle declares dtype {dtype!r} ({wanted}) but the "
                f"caller-owned model holds parameters in {sorted(map(str, found))}"
                " — the run receipt would describe a precision that did not "
                f"run; declare the model's actual dtype, or call model.to({wanted}) "
                "before handing it over",
            )


def quantization_key(
    quantization: Mapping[str, Any] | None,
) -> tuple[tuple[str, Any], ...] | None:
    """The one canonical form of a document's materialized ``model.quantization``
    block, for the cache key: its fields as pairs sorted by name, ``None`` for
    an unquantized realization. Equal blocks give equal keys whatever their
    order, so a realization is one cache entry (§2.1: quantization is a
    document fact, part of identity)."""
    if quantization is None:
        return None
    return tuple(sorted(quantization.items()))


@normalized_cache(maxsize=4, keys={"quantization": quantization_key})
def load_model(
    key: str,
    revision: str = "main",
    *,
    dtype: str = "fp32",
    device: str = "cpu",
    quantization: Mapping[str, Any] | None = None,
    attn_implementation: str | None = "eager",
) -> ModelBundle:
    """Load (and cache) one model bundle.

    The tokenizer is set to the engine's single padding convention: left
    padding with ``pad = eos`` when the checkpoint ships no pad token — the
    inherited pipeline contract the oracle tests were captured under.

    Four bundles stay resident, keyed on the *bound* arguments
    (:func:`~causalab.neural.shared.normalized_cache.normalized_cache`): an
    omitted default and its explicit spelling, positional and keyword, are one
    entry; revision, precision, device, quantization and the attention
    selection are each part of identity, and ``attn_implementation=None`` is
    distinct from ``"eager"``. ``load_model.cache_clear()`` and
    ``cache_info()`` manage the cache.

    ``quantization`` is the document's materialized ``model.quantization``
    block; :func:`quantization_key` is the one place its cache identity is
    defined. The realization is a document fact, not an engine flag (§2.1).

    ``attn_implementation`` selects a Transformers attention backend. Eager
    remains the reproducible default; ``None`` uses Transformers' default.
    Interior taps temporarily switch a forward to eager and restore this
    selection afterward.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # the compilers a CUDA model will use (Triton, TileLang, Inductor) are
    # pointed at the shared cache root, when one is set, before the first
    # kernel is built (shared/compile_cache.py). The loader is the seam rather
    # than the CLI's entry: the library is imported at least as often as it is
    # run from the command line, and the device is only known once a model is
    # placed — so every path a model can land on calls this, and the setting
    # is process-wide (the last call wins)
    configure_compile_cache(device)

    if quantization is None:
        # weights read straight onto ``device``, several shards at once
        # (``weights.py``); the stock CPU load + ``.to(device)`` is one
        # thread end to end, and measured three times slower
        model = load_pretrained(
            key,
            revision,
            dtype=_DTYPES[dtype],
            device=device,
            attn_implementation=attn_implementation,
        )
    else:
        # bitsandbytes quantizes on the way in and places the weights itself;
        # moving them afterwards is refused, so the requested device is only
        # recorded (``ModelBundle.device``)
        model = AutoModelForCausalLM.from_pretrained(
            key,
            revision=revision,
            dtype=_DTYPES[dtype],
            **(
                {"attn_implementation": attn_implementation}
                if attn_implementation is not None
                else {}
            ),
            quantization_config=_bitsandbytes_config(dict(quantization)),
        )
    model.eval()
    # only featurizer/free params ever train (§2.11); freezing the network
    # keeps training graphs from accumulating gradients into model weights
    model.requires_grad_(False)
    # a DeltaNet family's kernel globals follow the device this load put the
    # weights on: the torch path off CUDA, the installed kernels on it — so a
    # bare forward of the model works wherever it lives (shared/kernels.py)
    bind_kernel_path(model)
    tokenizer = AutoTokenizer.from_pretrained(key, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    info = model_info_from_hf_config(key, model.config)
    register_model(info)
    return ModelBundle(
        key=key,
        revision=revision,
        model=model,
        tokenizer=tokenizer,
        info=info,
        device=device,
        dtype=dtype,
        quantization=dict(quantization) if quantization is not None else None,
    )


def _bitsandbytes_config(quantization: dict[str, Any]) -> Any:
    """Lower a materialized ``model.quantization`` block to a
    ``BitsAndBytesConfig``.

    bitsandbytes is an optional extra: quantization is in the *document*
    vocabulary so that a shared protocol says which realization produced its
    numbers, and a reader without the library still gets a document that
    validates, digests and explains — only ``run`` needs the quantizer, and
    it says so precisely.

    Field mapping: https://huggingface.co/docs/transformers/main_classes/quantization
    """
    method = quantization.get("method", "bitsandbytes")
    if method != "bitsandbytes":
        raise ProtocolError(
            "P4", f"quantization method {method!r} has no reference implementation"
        )
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401 — the config is inert without it
    except ImportError as err:
        raise ProtocolError(
            "P2",
            f"this document declares {quantization.get('scheme')!r} weight "
            "quantization, which the reference engine realizes through "
            f"bitsandbytes — not installed ({err}). Install the extra, or run "
            "the document at its unquantized precision by setting "
            "model.quantization out of it (a different experiment, and its "
            "digest says so).",
        ) from err

    scheme = quantization["scheme"]
    compute_dtype = _DTYPES[quantization.get("compute_dtype", "fp32")]
    if scheme == "int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=float(quantization.get("int8_threshold", 6.0)),
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=scheme,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(quantization.get("double_quant", False)),
    )
