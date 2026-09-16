"""The model-family readout adapter: final normalization, unembedding in a
declared accumulation dtype, and centering — one engine-side service keyed by
the registry entry's ``family``.

**What a readout is.** ``logits = lm_head(final_norm(h))`` — the two module
calls the model itself makes after its last block. Both engines already
serve the two tensors (``ln_final``, ``lm_head`` are module-output taps at the
model root, ``registry.TreeAddress.final_norm`` / ``.lm_head``); what no
landed code declared is the *kind* of norm, its **gain convention**, where its
epsilon lives, and the dtype a downstream analysis should accumulate the
unembedding in. Those are exactly the facts a residual decomposition needs —
``N(v) = g · s · v`` linearises the final norm at the fixed scale ``s`` of the
whole residual, and ``g`` is ``weight`` on a Llama-style RMSNorm but
``1 + weight`` on Qwen3.5-MoE's — facts an analysis used to carry as
architecture branches in its own code. Here they are a
declaration per family, measured against the module's own forward
(:meth:`Readout.certify`) rather than assumed.

**Module application, never weight slicing.** :meth:`Readout.normalize`,
:meth:`Readout.logits` and :meth:`Readout.unembed` *call the modules the
family's tree addresses*. That is the principle ``executor_base`` states for
the head-wise projection: running the projection the model's own module
defines cannot be wrong about its own layout (``nn.Linear`` vs ``Conv1D``,
tied vs untied embeddings). The unembedding in the declared accumulation dtype
is the same module forward with its parameters cast
(``torch.func.functional_call``) — no ``.weight`` is read here or by any
consumer.

**Keyed by ``ModelInfo.family``, not ``FamilyAdapter.family``.** The tree
family is too coarse: ``llama_tree`` detects both Llama and Qwen3.5-MoE, whose
final RMSNorms differ in the one fact the linearisation needs (the gain
convention). The HF ``model_type`` the registry entry records is the right
key — 📐 measured on the tiny fixtures (``gpt2``: ``LayerNorm``, gain
``weight``, epsilon ``eps``; ``llama``: ``LlamaRMSNorm``, ``weight``,
``variance_epsilon``; ``qwen3_5_moe_text``: ``Qwen3_5MoeRMSNorm``,
``one_plus_weight``, ``eps`` — the same class the real Qwen3.6-35B-A3B runs,
with the same ``one_plus_weight`` gain).

**Deliberately outside the protocol layer.** Nothing hashed imports this
module, and no ``SHARED`` member of the shipped scripts' closure may — so a
torch-free load never reaches it. The readout is not document vocabulary
either: centering is invisible to every softmax-based metric (spec §2.9, a
uniform shift is a no-op) and shows only in raw ``token_logit`` values, so a
``center`` field or a ``centered_logits`` component is a schema decision
batched with the next legitimate ``schema.py`` change, not smuggled in here.
Likewise a run receipt's ``execution.readout`` block is a change to
``run.py:execution_record``, batched the same way.

**Refusals, by name.** A family with no declaration, an accumulation dtype
outside ``{fp32, fp64}``, a declared epsilon attribute the module lacks, and a
gain convention the module's forward contradicts are each a ``ValueError``
naming the thing and the fix — the ``Identity.tolerance_for`` precedent, never
a bare ``KeyError`` / ``AttributeError``. Every refusal has a passing twin in
``tests/neural/shared/test_readout.py``.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Any, Literal, Mapping, get_args

import torch
from torch.func import functional_call

from causalab.protocol.registry import walk

__all__ = [
    "ACCUMULATION_DTYPES",
    "CERTIFICATION_ULPS",
    "GAINS",
    "NORMS",
    "READOUT_SPECS",
    "UNIT_ROUNDOFF",
    "AccumulationDtype",
    "Certificate",
    "Gain",
    "GainMismatch",
    "Norm",
    "Readout",
    "ReadoutSpec",
    "readout_spec",
    "register_readout",
    "unit_roundoff",
]

#: The final normalization's kind: a root-mean-square norm (no centering) or a
#: layer norm (centered, and with an additive term the module may carry).
Norm = Literal["rmsnorm", "layernorm"]
NORMS: tuple[Norm, ...] = get_args(Norm)

#: How the norm's ``weight`` enters: Llama-style ``weight · normed`` or the
#: zero-centred ``(1 + weight) · normed`` of Qwen3.5-MoE (transformers PR
#: 29402's family). 📐 Which one a family uses is measured, not assumed:
#: :meth:`Readout.certify` refuses a declaration the module's forward
#: contradicts.
Gain = Literal["weight", "one_plus_weight"]
GAINS: tuple[Gain, ...] = get_args(Gain)

#: The dtype the *reference* unembedding accumulates in. Only the two that
#: never round a bf16/fp16/fp32 forward's values coarser than the forward did:
#: an analysis that attributes rounding residuals to declared terms cannot be
#: run in a dtype that adds its own.
AccumulationDtype = Literal["fp32", "fp64"]
ACCUMULATION_DTYPES: tuple[AccumulationDtype, ...] = get_args(AccumulationDtype)
_TORCH_DTYPES: Mapping[str, torch.dtype] = MappingProxyType(
    {"fp32": torch.float32, "fp64": torch.float64}
)

#: Unit roundoff (half an ulp at 1.0) per dtype a readout may run in — the
#: relative size of one rounding of a forward's value. The three protocol
#: ``native_dtype`` spellings' dtypes plus fp64.
UNIT_ROUNDOFF: Mapping[torch.dtype, float] = MappingProxyType(
    {
        torch.float64: 2.0**-53,
        torch.float32: 2.0**-24,
        torch.float16: 2.0**-11,
        torch.bfloat16: 2.0**-8,
    }
)

#: The certification band, in units of ``unit_roundoff(dtype) · max|norm(x)|``.
#: 📐 The right convention measures ≈ 2 such units on every tiny fixture in
#: fp32 (gpt2 4.3e-7, llama 3.1e-7, qwen3.5-moe 2.1e-7 at |ln_final| 2.4–3.5)
#: and ≈ 0.4 on the real A3B in bf16 (the probe's 0.0463 at |ln_final| 30.1);
#: the wrong one measures ≈ |ln_final| itself (2.2–3.5 on the fixtures, 10.4 on
#: the A3B) — six orders of magnitude apart in fp32, one in bf16 with 8 units
#: as the band.
CERTIFICATION_ULPS = 8


def unit_roundoff(dtype: torch.dtype) -> float:
    """The unit roundoff of ``dtype``, or a refusal naming the dtypes a
    readout is certified in — a tolerance is declared per dtype, never
    interpolated."""
    try:
        return UNIT_ROUNDOFF[dtype]
    except KeyError:
        raise ValueError(
            f"no unit roundoff is declared for {dtype} (declared: "
            f"{[str(d) for d in UNIT_ROUNDOFF]}) — a readout is certified in a "
            "dtype whose rounding it can name"
        ) from None


@dataclasses.dataclass(frozen=True)
class ReadoutSpec:
    """One family's readout declaration: the final norm's kind, its gain
    convention, the attribute its epsilon lives at on the module, and the
    dtype the reference unembedding accumulates in. Every field is a closed
    vocabulary or a module attribute name, refused by name otherwise."""

    norm: Norm
    gain: Gain
    eps_attr: str
    accumulation_dtype: AccumulationDtype

    def __post_init__(self) -> None:
        if self.norm not in NORMS:
            raise ValueError(f"readout norm {self.norm!r} is not in {list(NORMS)}")
        if self.gain not in GAINS:
            raise ValueError(f"readout gain {self.gain!r} is not in {list(GAINS)}")
        if not self.eps_attr.isidentifier():
            raise ValueError(
                f"readout eps_attr {self.eps_attr!r} is not an attribute name"
            )
        if self.accumulation_dtype not in ACCUMULATION_DTYPES:
            raise ValueError(
                f"readout accumulation dtype {self.accumulation_dtype!r} is not in "
                f"{list(ACCUMULATION_DTYPES)} — a reference unembedding accumulates "
                "in a dtype at least as wide as the forward's, so the rounding it "
                "attributes is the model's and not its own"
            )


_READOUT_SPECS: dict[str, ReadoutSpec] = {}

#: The registered declarations, keyed by ``ModelInfo.family`` (the HF
#: ``model_type`` of the entry's text config) — read-only view;
#: :func:`register_readout` is the one way in, from any module.
READOUT_SPECS: Mapping[str, ReadoutSpec] = MappingProxyType(_READOUT_SPECS)


def register_readout(family: str, spec: ReadoutSpec) -> None:
    """Register (or replace) the readout declaration of ``family`` — the
    ``register_family`` precedent: a third-party family declares its readout
    beside its adapter, from its own module."""
    if not isinstance(family, str) or not family.isidentifier():
        raise ValueError(f"readout family {family!r} is not an identifier")
    if not isinstance(spec, ReadoutSpec):
        raise ValueError(
            f"family {family!r}: the readout declaration is not a ReadoutSpec"
        )
    _READOUT_SPECS[family] = spec


def readout_spec(family: str | None, *, key: str | None = None) -> ReadoutSpec:
    """The declaration of ``family``, or a refusal naming the family, the
    declared ones and how to declare a new one. ``key`` names the model in the
    refusal of an entry that records no family at all."""
    where = f" (model {key!r})" if key else ""
    if family is None:
        raise ValueError(
            f"the registry entry{where} records no family (ModelInfo.family), so no "
            f"readout declaration can be looked up (declared: {sorted(_READOUT_SPECS)}) "
            "— set the entry's family and declare its readout with "
            "causalab.neural.shared.readout.register_readout"
        )
    try:
        return _READOUT_SPECS[family]
    except KeyError:
        raise ValueError(
            f"family {family!r}{where} declares no readout (declared: "
            f"{sorted(_READOUT_SPECS)}) — declare its final norm, gain convention, "
            "epsilon attribute and accumulation dtype with "
            "causalab.neural.shared.readout.register_readout"
        ) from None


# 📐 Measured on the tiny fixtures (2026-09-03, transformers 5.16 lock), norm
# module forward against both conventions at the fixed scale — the numbers are
# in tests/neural/shared/test_readout.py's docstring. The accumulation dtype is
# float64 because the residual accounting is exact only there.
register_readout(
    "gpt2",
    ReadoutSpec(
        norm="layernorm", gain="weight", eps_attr="eps", accumulation_dtype="fp64"
    ),
)
register_readout(
    "llama",
    ReadoutSpec(
        norm="rmsnorm",
        gain="weight",
        eps_attr="variance_epsilon",
        accumulation_dtype="fp64",
    ),
)
register_readout(
    "qwen3_5_moe_text",
    ReadoutSpec(
        norm="rmsnorm",
        gain="one_plus_weight",
        eps_attr="eps",
        accumulation_dtype="fp64",
    ),
)


@dataclasses.dataclass(frozen=True)
class Certificate:
    """What :meth:`Readout.certify` measured: the declared gain's gap between
    the module forward and the linearisation at the fixed scale, the band it
    was held to, and every convention's gap beside it — so a test pins the
    separation rather than trusting the verdict."""

    family: str
    dtype: str
    gain: Gain
    gap: float
    tolerance: float
    #: ``max|norm(x)|`` as run — the magnitude the band is relative to
    scale: float
    gaps: Mapping[Gain, float]


class GainMismatch(ValueError):
    """The declared gain convention is not the one the module computes."""

    def __init__(self, message: str, certificate: Certificate) -> None:
        super().__init__(message)
        self.certificate = certificate


@dataclasses.dataclass(frozen=True)
class Readout:
    """One loaded model's readout: the final-norm and head modules the family's
    tree addresses, and the family's declaration.

    :meth:`normalize` and :meth:`logits` are the readout **as run** — the
    modules called as the model calls them, at the model's dtype, so
    ``logits(block_output@last)`` is the engine's ``lm_head`` read bit for bit.
    :meth:`unembed` is the **reference** projection: the head's own forward in
    the declared accumulation dtype. :meth:`fixed_rms_scale`,
    :meth:`linearized_norm` and :meth:`norm_offset` are the linearisation a
    residual decomposition uses (``norm(x) ≈ g · s · x + b`` at the fixed
    ``s`` of the whole residual), and :meth:`certify` holds the declaration to
    the module.
    """

    family: str
    spec: ReadoutSpec
    norm: Any
    head: Any

    @classmethod
    def from_bundle(cls, bundle: Any) -> "Readout":
        """Build from a loaded bundle (either engine's): the family adapter's
        ``tree.final_norm`` / ``tree.lm_head`` walked on the model, and the
        declaration of ``bundle.info.family``. Refuses by name a family with
        no declaration, a tree address the model lacks, and a declared epsilon
        attribute or weight the norm module does not have."""
        info = bundle.info
        adapter = bundle.adapter
        spec = readout_spec(info.family, key=info.key)
        norm = walk(bundle.model, adapter.tree.final_norm)
        if norm is None:
            raise ValueError(
                f"family {adapter.family!r} addresses its final norm at "
                f"{adapter.tree.final_norm!r}, but this model "
                f"({type(bundle.model).__name__}) has no such child"
            )
        head = walk(bundle.model, adapter.tree.lm_head)
        if head is None:
            raise ValueError(
                f"family {adapter.family!r} addresses its head at "
                f"{adapter.tree.lm_head!r}, but this model "
                f"({type(bundle.model).__name__}) has no such child"
            )
        if not hasattr(norm, spec.eps_attr):
            have = sorted(
                a for a in ("eps", "variance_epsilon", "epsilon") if hasattr(norm, a)
            )
            raise ValueError(
                f"family {info.family!r} declares its final norm's epsilon at "
                f"attribute {spec.eps_attr!r}, but the module "
                f"({type(norm).__name__}) has no such attribute (it has {have}) — "
                "the declaration and the loaded module disagree; fix the "
                "declaration (register_readout)"
            )
        if getattr(norm, "weight", None) is None:
            raise ValueError(
                f"family {info.family!r} declares a {spec.gain!r} gain on its final "
                f"norm, but the module ({type(norm).__name__}) has no weight"
            )
        return cls(family=info.family, spec=spec, norm=norm, head=head)

    def at(self, accumulation_dtype: str) -> "Readout":
        """The same readout with another accumulation dtype for the reference
        unembedding — validated as a declaration is, so ``bf16`` is refused by
        name here too."""
        return dataclasses.replace(
            self,
            spec=dataclasses.replace(
                self.spec,
                accumulation_dtype=accumulation_dtype,  # pyright: ignore[reportArgumentType]
            ),
        )

    @property
    def eps(self) -> float:
        """The final norm's epsilon, read at the declared attribute."""
        return float(getattr(self.norm, self.spec.eps_attr))

    @property
    def accumulation_dtype(self) -> torch.dtype:
        return _TORCH_DTYPES[self.spec.accumulation_dtype]

    # --- as run ------------------------------------------------------------ #

    def normalize(self, h: torch.Tensor) -> torch.Tensor:
        """``final_norm(h)`` — the module call, at ``h``'s dtype."""
        return self.norm(h)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """``lm_head(final_norm(h))`` — the readout as the model runs it. On
        the last block's output this is the ``lm_head`` read, bit for bit."""
        return self.head(self.norm(h))

    # --- the reference ----------------------------------------------------- #

    def unembed(self, z: torch.Tensor) -> torch.Tensor:
        """The head's own forward on ``z`` in the declared accumulation dtype,
        cast back to ``z``'s dtype. The module's parameters are cast (and
        moved to ``z``'s device) for the call, never read or sliced: whatever
        layout the head has, its forward knows it. When ``z`` and the head are
        already in the accumulation dtype this is the plain module call."""
        acc = self.accumulation_dtype
        params = dict(self.head.named_parameters())
        if z.dtype == acc and all(p.dtype == acc for p in params.values()):
            return self.head(z)
        tensors: dict[str, torch.Tensor] = {
            name: p.detach().to(device=z.device, dtype=acc)
            for name, p in params.items()
        }
        for name, buffer in self.head.named_buffers():
            tensors[name] = (
                buffer.to(device=z.device, dtype=acc)
                if buffer.is_floating_point()
                else buffer.to(device=z.device)
            )
        out = functional_call(self.head, tensors, (z.to(acc),))
        return out.to(z.dtype)

    @staticmethod
    def center(logits: torch.Tensor) -> torch.Tensor:
        """The centered readout: ``logits − mean over the vocabulary``, per
        position. A uniform shift, so every softmax-based metric is invariant
        to it (spec §2.9); it is visible only to raw logit values, which is
        why it is a Python method here and not yet a document field."""
        return logits - logits.mean(dim=-1, keepdim=True)

    # --- the linearisation ------------------------------------------------- #

    def fixed_rms_scale(self, x: torch.Tensor) -> torch.Tensor:
        """The norm's own scale ``s = rsqrt(mean(x²) + eps)`` over the last
        axis of ``x`` — the **fixed** scale of the whole residual, at which the
        decomposition linearises the norm. For a layer norm the mean is of the
        centered ``x`` (its variance), as the module computes it."""
        if self.spec.norm == "layernorm":
            x = x - x.mean(dim=-1, keepdim=True)
        return torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def linearized_norm(self, v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """``N(v) = g · s · v`` with ``g`` from the declared gain convention
        (``weight`` → ``w``; ``one_plus_weight`` → ``1 + w``), ``v`` centered
        first under a layer norm. Linear in ``v`` at fixed ``s``, which is what
        lets a sum of components be normalised term by term."""
        return self._linearized(v, s, self.spec.gain)

    def norm_offset(self, like: torch.Tensor) -> torch.Tensor:
        """The norm's additive term — a layer norm's bias, zero for a norm
        without one — in ``like``'s dtype and device. Added **once** to a
        decomposition, never per component."""
        bias = getattr(self.norm, "bias", None)
        if bias is None:
            return torch.zeros((), dtype=like.dtype, device=like.device)
        return bias.detach().to(device=like.device, dtype=like.dtype)

    def _linearized(self, v: torch.Tensor, s: torch.Tensor, gain: str) -> torch.Tensor:
        w = self.norm.weight.detach().to(device=v.device, dtype=v.dtype)
        g = w if gain == "weight" else 1.0 + w
        if self.spec.norm == "layernorm":
            v = v - v.mean(dim=-1, keepdim=True)
        return g * (s * v)

    def certify(self, x: torch.Tensor) -> Certificate:
        """Hold the declaration to the module: one forward of the norm module
        on ``x`` against ``linearized_norm(x, fixed_rms_scale(x)) +
        norm_offset`` in float64, for **every** gain convention. The declared
        one must land within :data:`CERTIFICATION_ULPS` units of
        ``x``'s dtype's roundoff at ``max|norm(x)|`` — measured, so a wrong
        declaration is refused naming both conventions and the gap
        (:class:`GainMismatch`), and the returned :class:`Certificate` carries
        every gap for a test to pin the separation."""
        with torch.no_grad():
            as_run = self.normalize(x).detach()
            reference_dtype = torch.float64
            x_ref = x.detach().to(reference_dtype)
            s = self.fixed_rms_scale(x_ref)
            offset = self.norm_offset(x_ref)
            as_run_ref = as_run.to(reference_dtype)
            gaps: dict[Gain, float] = {
                gain: float(
                    (as_run_ref - (self._linearized(x_ref, s, gain) + offset))
                    .abs()
                    .max()
                )
                for gain in GAINS
            }
            scale = float(as_run.abs().max())
        tolerance = CERTIFICATION_ULPS * unit_roundoff(as_run.dtype) * scale
        gap = gaps[self.spec.gain]
        certificate = Certificate(
            family=self.family,
            dtype=str(as_run.dtype).removeprefix("torch."),
            gain=self.spec.gain,
            gap=gap,
            tolerance=tolerance,
            scale=scale,
            gaps=MappingProxyType(gaps),
        )
        if not gap <= tolerance:
            others = ", ".join(
                f"{name!r} measures {value:.3e}"
                for name, value in gaps.items()
                if name != self.spec.gain
            )
            raise GainMismatch(
                f"family {self.family!r} declares its final norm's gain as "
                f"{self.spec.gain!r}, but the module ({type(self.norm).__name__}) "
                f"disagrees: the gap between its forward and g·s·x is {gap:.3e} in "
                f"{certificate.dtype}, over the band {tolerance:.3e} "
                f"({CERTIFICATION_ULPS} units of roundoff at max|norm(x)| = "
                f"{scale:.3e}); {others} — fix the declaration (register_readout)",
                certificate,
            )
        return certificate
