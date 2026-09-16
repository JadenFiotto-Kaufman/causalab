"""Path patching as a compiler into the existing nouns (spec §3.2).

A path-patching experiment (``IOI``) is five kinds of entry written by hand:
a sender site, a receiver site, one restorer site per layer in between, the
reads that harvest each of them, the swap that patches the sender, the
freezes that hold every restorer at its clean value, the read of the receiver
under all of that, the write that injects it into an otherwise clean run, and
the two intervened models that group the writes. The shipped
``configs/protocols/path_patching.json`` is those twenty entries for one
sender, one receiver and two frozen layers. Nothing in them is a choice except
the sender, the receivers, the position, which data role the sender's value
comes from, and **which sites are restored** — and that last choice, the
restoration policy, is invisible in the hand-written form: it *is* which
sites happen to appear in ``writes``.

**A path block is a lowering, not a dialect.** One ``method.path_patching``
object names exactly those choices and compiles, here, to exactly the entries
the author would have written — the shipped document's own names, so that the
shipped document re-expressed as a block compiles **byte-identically** to the
hand-written one (``test_paths.py``). The block never reaches the shape
gate, the checklist, the canonical form or an engine: this stage runs between
``families`` and ``gate`` (:data:`~causalab.protocol.compile.STAGES`), rewrites
the explicit tree, and is gone. Every field it takes names something §2
already has — a ``SiteSpec`` (§2.4), a data role (§2.2), one ``pos`` in the
§2.3 grammar copied verbatim into every emitted read and write — and the two
words that are its own (``restoration`` and its policies, ``receivers``) are
the compiler's input and nothing else's. It grows no positions vocabulary and
takes no list of layers to restore: the freeze layers are *derived* from the
sender and the receivers, which is what keeps it a compiler and not a second
runtime. A sender or receiver site spells its depth as §2.4 does — ``layers``,
the band — and **a path runs between two layers**, so each is the one-layer
band (``L`` or ``[L]``, one digest either way) and a band of several layers is
refused ``P4`` by name (:func:`_site`); every site the block emits is written
in the canonical one-layer form ``"layers": [L]``.

**What it makes explicit that the hand-written form leaves implicit.**

* *Ordered receiver sets*: ``receivers`` is an ordered list, every
  member is injected in **one** intervened model — one forward, one joint
  intervention, never a sum of separate runs — and a one-element set is spelled
  ``receiver`` / ``v_receiver`` / ``inject``, the scalar document's own names,
  so it is the scalar case to the byte.
* *Restoration as a field*: ``attention_only`` restores
  ``attention_output`` at every layer strictly between the sender and the
  farthest receiver; ``attention_and_mlp`` additionally restores
  ``mlp_output`` at those layers **and at the sender's own layer** — the MLP
  of the sender's block is downstream of an attention sender and would
  otherwise carry the patched value forward off-path. That choice is stated
  here, in the spec (§3.2) and in the derived record, not made silently. The
  policy enters the point identity through the write set it generates: two
  blocks differing only in policy lower to different ``writes`` and therefore
  to different canonical bytes and point digests (§7), so a comparison across
  policies fails its ``produced_by`` check rather than being silently made.
* *The restorer boundary*: the canonical form sorts an intervened
  model's write list, and execution installs writes in module order, so the
  inter-layer order of the restorers is nowhere in the document. It is derived
  data — ``(layer, COMPONENT_RANK[component])`` — and the compiled form records
  it as an ordered list.

**The compiled form is saved** (a *derived* artifact, never a canonical
section). :func:`describe_paths` is the record —
the block as written, the policy, the receiver order, the restorer boundary
and the names of every emitted entry — carried as the compiler's eighth output
(:attr:`~causalab.protocol.compile.CompiledProtocol.lowered`) and written into
the run receipt as ``derived``. Its binding to the run's identity is by
reference: every emitted name is a key of the receipt's canonical ``method``,
and the document and point digests in the same receipt are over that canonical
form. Two runs that both say "path patching" are compared field by field
there.

**Refusals** carry the parser's codes and no rule number: a malformed block is
a shape error of the block (``P2``), an unknown key ``P3`` with a suggestion,
an unknown policy ``P4``. What the block cannot say — a receiver at or below
the sender, a ``sweep`` or ``at_once`` wrapper inside the block — is refused
here, first, naming ``method.path_patching.<field>`` (an ``at_once`` inside
the block is caught one stage earlier, by ``families``, as rule 28: it sits
where it has no name identity). So is **two emitted sites at one address**:
rule 8 compares absolute writes by site *name*, and the lowering hands every
site its own name, so a duplicate receiver, a receiver at a site the policy
freezes (it would read the frozen clean value and inject clean for clean) and
a sender inside the MLP its own policy holds clean (the freeze would overwrite
everything it wrote — a path effect of exactly zero) would all *run*, silently;
the block refuses each as ``P2`` naming the sender or ``receivers[i]``
responsible. One name for both intervened models, and an authored table that
is not an object where the block emits into it, are ``P2`` for the same
reason: nothing downstream would say so by name.
What the *lowered* document then gets wrong is the checklist's business (a
generated name colliding with an authored one in another section is rule 3).
``--set`` cannot address a field of the block: overrides are section-rooted
(§1) and the block is not a section, so ``--set path_patching.…`` is refused
as a path that does not exist. That, sweeps into the block, and a numbered
rule are deferred deliberately — each would make this a digest mover.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Mapping

from causalab.protocol.errors import ParseError, suggest
from causalab.protocol.plan import COMPONENT_RANK
from causalab.protocol.schema import COMPONENTS, METHOD_SECTIONS
from causalab.protocol.sweep import AT_ONCE_KEY, SWEEP_KEY, band_label

__all__ = [
    "BLOCK",
    "BLOCK_KEYS",
    "POLICIES",
    "RESTORED_COMPONENTS",
    "describe_paths",
    "expand_paths",
    "has_path_block",
]

#: The one key a path block lives under in the ``method`` group (§3.2).
BLOCK = "path_patching"

#: The restoration policies (§3.2): which sites between the sender and the
#: receivers are held at their clean value in the harvest model.
POLICIES: tuple[str, ...] = ("attention_only", "attention_and_mlp")

#: Per policy, the components restored — in the order the forward computes
#: them, which is also their :data:`~causalab.protocol.plan.COMPONENT_RANK`
#: order — and the site-name prefix each takes (``a10``, ``m10``: the shipped
#: document's spelling for attention, its natural sibling for the MLP).
RESTORED_COMPONENTS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "attention_only": (("attention_output", "a"),),
    "attention_and_mlp": (("attention_output", "a"), ("mlp_output", "m")),
}

#: Per restored component, the residual junction its sub-block reads from. A
#: freeze at ``attention_output@L`` holds clean everything the attention of
#: layer *L* computed after ``block_input``; a freeze at ``mlp_output@L``
#: everything the MLP computed after ``block_mid``. A *write* at the same layer
#: whose component ranks strictly after the junction and up to the frozen
#: component is therefore overwritten by the freeze — which is what
#: :func:`_check_addresses` refuses for the sender.
SUBBLOCK_ENTRY: Mapping[str, str] = {
    "attention_output": "block_input",
    "mlp_output": "block_mid",
}

#: The sub-block each restored component is the output of, as a reader says it.
SUBBLOCK_NAME: Mapping[str, str] = {
    "attention_output": "attention",
    "mlp_output": "MLP",
}

#: Every key a block may carry, in the recommended order.
BLOCK_KEYS: tuple[str, ...] = (
    "sender",
    "source",
    "receivers",
    "pos",
    "restoration",
    "harvest",
    "inject",
)

#: The keys with no default — the choices that *are* the experiment.
REQUIRED_KEYS: tuple[str, ...] = ("sender", "receivers", "pos", "restoration")

#: The keys that name something conventional: the data role the sender's
#: value is read from, and the two intervened models' names — the shipped
#: document's, so a hand-authored ``reads.logits.model`` can name them.
DEFAULTS: Mapping[str, str] = {
    "source": "counterfactual",
    "harvest": "patched",
    "inject": "final",
}

#: The keys of an inline site (§2.4, ``SiteSpec``), passed through as written
#: — except ``layers``, which :func:`_site` normalizes to the canonical
#: one-layer band ``[L]`` (a bare ``L`` is the same site, §2.4).
SITE_KEYS: tuple[str, ...] = ("component", "layers", "head", "expert", "stream")

#: The names the sender lowers to — the shipped document's.
SENDER_SITE = "sender"
SENDER_READ = "v_sender"
SENDER_WRITE = "swap_sender"

#: The path every refusal here names: the block is not a section (§1), so it
#: is spelled with its group, which is also how a reader finds it in the file.
BLOCK_PATH = f"method.{BLOCK}"


def has_path_block(raw: Mapping[str, Any]) -> bool:
    """Whether the document ``raw`` carries a path block — the cheap guard that
    keeps the ``paths`` stage off every document written before §3.2."""
    method = raw.get("method")
    return isinstance(method, Mapping) and BLOCK in method


# --------------------------------------------------------------------------- #
# the block, parsed
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class _Path:
    """One path block after its shape checks: the choices, and nothing derived
    yet — derivations are the methods, so that :func:`expand_paths` and
    :func:`describe_paths` cannot disagree about a name."""

    block: Mapping[str, Any]
    sender: Mapping[str, Any]
    sender_layer: int
    source: str
    receivers: tuple[Mapping[str, Any], ...]
    receiver_layers: tuple[int, ...]
    pos: Any
    restoration: str
    harvest: str
    inject: str

    @property
    def receiver_names(self) -> tuple[str, ...]:
        """``receiver`` for a one-element set — the scalar document's own name,
        which is what makes a one-element set the scalar case to the byte —
        and ``receiver_0``, ``receiver_1``, … for several."""
        if len(self.receivers) == 1:
            return ("receiver",)
        return tuple(f"receiver_{i}" for i in range(len(self.receivers)))

    @property
    def inject_names(self) -> tuple[str, ...]:
        if len(self.receivers) == 1:
            return ("inject",)
        return tuple(f"inject_{i}" for i in range(len(self.receivers)))

    @property
    def restored_layers(self) -> dict[str, list[int]]:
        """Per restored component, the layers held at their clean value.

        ``attention_output``: every layer strictly between the sender and the
        farthest receiver. ``mlp_output`` (``attention_and_mlp`` only): those
        layers **and the sender's own** — the sender's block's MLP reads the
        residual stream *after* the attention sender wrote into it, so it is
        downstream of the sender and off the direct path (§3.2 states this
        choice; the derived record carries it as data)."""
        stop = max(self.receiver_layers)
        out: dict[str, list[int]] = {}
        for component, _prefix in RESTORED_COMPONENTS[self.restoration]:
            start = (
                self.sender_layer
                if component == "mlp_output"
                else (self.sender_layer + 1)
            )
            out[component] = list(range(start, stop))
        return out

    @property
    def restorers(self) -> tuple[tuple[int, str, str], ...]:
        """The restorer boundary: ``(layer, component, site name)`` for every
        restored site, ordered by ``(layer, COMPONENT_RANK[component])`` — the
        order the forward computes them in, which the document's write list
        (sorted in the canonical form, §7) and execution (module order) never
        spell out. The ordered component boundary, as data."""
        prefix = dict(RESTORED_COMPONENTS[self.restoration])
        found = [
            (layer, component, f"{prefix[component]}{layer}")
            for component, layers in self.restored_layers.items()
            for layer in layers
        ]
        return tuple(sorted(found, key=lambda item: (item[0], COMPONENT_RANK[item[1]])))

    @property
    def sites(self) -> dict[str, dict[str, Any]]:
        """Every site the block lowers to, name → site, in the order the
        shipped document writes them: the sender, the receivers, the
        restorers. The one list :func:`_emitted` writes *and*
        :func:`_check_addresses` checks, so the check and the emission cannot
        drift."""
        out: dict[str, dict[str, Any]] = {SENDER_SITE: dict(self.sender)}
        for name, site in zip(self.receiver_names, self.receivers):
            out[name] = dict(site)
        for layer, component, name in self.restorers:
            # the canonical one-layer band (§2.4, `canonical._canon_site`), so
            # the lowered document is the hand-written v3 twin's form to the byte
            out[name] = {"component": component, "layers": [layer]}
        return out

    def freeze_name(self, component: str, layer: int) -> str:
        """``freeze_10`` for the attention restorer (the shipped spelling),
        ``freeze_m10`` for the MLP's."""
        prefix = dict(RESTORED_COMPONENTS[self.restoration])[component]
        return f"freeze_{layer}" if prefix == "a" else f"freeze_{prefix}{layer}"


def _refuse(code: str, message: str, path: str) -> ParseError:
    return ParseError(code, message, path=path)


def _find_key(node: Any, key: str, path: str) -> str | None:
    """The path of the first occurrence of ``key`` as a mapping key anywhere
    in ``node`` — :mod:`families`' walk, for the wrapper refusal."""
    if isinstance(node, Mapping):
        for name, value in node.items():
            here = f"{path}.{name}"
            if name == key:
                return path
            found = _find_key(value, key, here)
            if found is not None:
                return found
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found = _find_key(item, key, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _layer(site: Mapping[str, Any]) -> int:
    """The one layer of a parsed sender or receiver: its ``layers`` is the
    one-member band ``[L]`` once :func:`_site` has normalized it."""
    return site["layers"][0]


def _site(raw: Any, path: str) -> tuple[Mapping[str, Any], int]:
    """An inline site as written (§2.4) with its ``layers`` normalized to the
    canonical one-layer band ``[L]``, and that layer as a number — which the
    lowering needs: the freeze range is derived from it.

    **A path runs between two layers**, so a sender or a receiver is the
    one-layer band, spelled ``L`` or ``[L]`` (one digest either way, §2.4). A
    band of several layers is refused ``P4`` here — where the block's other
    refusals live — rather than lowered to a guess: the freeze range has no
    defined start or stop for it, and a member-wise reading ("one path per
    member") is a *set* of paths, which the author says explicitly with one
    block per layer. The human can reverse this in this one place (a "band
    path" would be the one lowering to N sender swaps or N receiver injects
    at once)."""
    if not isinstance(raw, Mapping):
        raise _refuse("P2", "a site is an object with a 'component'", path)
    for key in raw:
        if key not in SITE_KEYS:
            raise _refuse(
                "P3", f"unknown key {key!r}{suggest(key, SITE_KEYS)}", f"{path}.{key}"
            )
    if "component" not in raw:
        raise _refuse("P2", "a site needs a 'component'", path)
    component = raw["component"]
    if not isinstance(component, str) or component not in COMPONENTS:
        raise _refuse(
            "P4",
            f"unknown component {component!r}{suggest(str(component), COMPONENTS)}",
            f"{path}.component",
        )
    role = path.rsplit(".", 1)[-1].split("[")[0]
    band = raw.get("layers")
    members = [band] if _is_int(band) else band
    if (
        not isinstance(members, list)
        or not members
        or not all(_is_int(member) for member in members)
    ):
        raise _refuse(
            "P2",
            f"a path's {role} needs 'layers' naming one layer, L or [L] — the "
            "freeze layers are derived from it",
            path,
        )
    if len(members) > 1:
        raise _refuse(
            "P4",
            "a path runs between two layers; a band "
            f"{'sender' if role == 'sender' else 'receiver'} "
            f"({band_label(members)}: {len(members)} layers) is not defined — "
            "write one path per layer",
            f"{path}.layers",
        )
    return {**raw, "layers": list(members)}, members[0]


def _word(block: Mapping[str, Any], key: str, path: str) -> str:
    value = block.get(key, DEFAULTS[key])
    if not isinstance(value, str) or not value:
        raise _refuse("P2", f"{key!r} is a name (a non-empty string)", f"{path}.{key}")
    return value


def _address(site: Mapping[str, Any]) -> str:
    """``attention_premix@9 head=9`` — a site as a reader spells its address."""
    scope = " ".join(
        f"{key}={site[key]}" for key in SITE_KEYS[2:] if site.get(key) is not None
    )
    return f"{site['component']}@{_layer(site)}" + (f" {scope}" if scope else "")


def _overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Whether two sites address one tensor, or one addresses part of the
    other: the same component at the same layer, and every scope key
    (``head``, ``expert``, ``stream``) either absent on one side or equal —
    a whole-component site covers each of its heads."""
    if a["component"] != b["component"] or _layer(a) != _layer(b):
        return False
    return all(
        a.get(key) is None or b.get(key) is None or a[key] == b[key]
        for key in SITE_KEYS[2:]
    )


def _frozen_over(site: Mapping[str, Any], component: str, layer: int) -> bool:
    """Whether a freeze at ``component@layer`` overwrites a *write* at
    ``site``: the same layer, and the site's component strictly after the
    sub-block's residual junction (:data:`SUBBLOCK_ENTRY`) and at most the
    frozen component itself in :data:`~causalab.protocol.plan.COMPONENT_RANK`
    — the MLP's input, its activation and its output are all overwritten by a
    freeze of its output; the residual stream beside it is not."""
    if _layer(site) != layer:
        return False
    rank = COMPONENT_RANK[site["component"]]
    return COMPONENT_RANK[SUBBLOCK_ENTRY[component]] < rank <= COMPONENT_RANK[component]


def _check_addresses(path: _Path) -> None:
    """No two emitted sites at one address (§3.2). Rule 8 compares absolute
    writes by site *name*, and the lowering gives every site its own name, so
    without this every collision below would *run*, silently: the sender's
    swap overwritten by the clean freeze of its own layer's MLP (a path effect
    of exactly zero), a receiver reading a frozen clean value and injecting
    clean for clean, two receivers swapping one address twice. Each is ``P2``
    naming the authored key responsible — the sender, or the later
    ``receivers[i]`` — and saying what would have happened. The restorers are
    derived, so they are never the one blamed."""
    receivers = list(zip(path.receiver_names, path.receivers))
    for index, (_name, site) in enumerate(receivers):
        for earlier, (_other_name, other) in enumerate(receivers[:index]):
            if _overlap(other, site):
                raise _refuse(
                    "P2",
                    f"receivers[{index}] repeats receivers[{earlier}] "
                    f"({_address(site)}): two absolute swaps at one address in "
                    "one intervened model (rule 8, which compares by site name "
                    "and would not see it) — a receiver set is a set",
                    f"{BLOCK_PATH}.receivers[{index}]",
                )
    for layer, component, name in path.restorers:
        freeze = path.freeze_name(component, layer)
        if _frozen_over(path.sender, component, layer):
            raise _refuse(
                "P2",
                f"the sender at {_address(path.sender)} is inside the "
                f"{SUBBLOCK_NAME[component]} of layer {layer}, which "
                f"{path.restoration!r} holds at its clean "
                f"value ({name} = {component}@{layer}, {freeze}): the clean "
                "freeze at the same address would overwrite the sender's effect "
                "— the path effect would be exactly zero. Move the sender or "
                "choose a policy that does not restore it",
                f"{BLOCK_PATH}.sender",
            )
        frozen = path.sites[name]
        for index, (_receiver_name, site) in enumerate(receivers):
            if _overlap(site, frozen):
                raise _refuse(
                    "P2",
                    f"receivers[{index}] ({_address(site)}) is a site "
                    f"{path.restoration!r} holds at its clean value ({name}, "
                    f"{freeze}): the receiver would read the frozen clean value "
                    "and inject clean for clean — the path effect through it "
                    "would be exactly zero. Move the receiver off the restored "
                    "boundary or choose a policy that does not restore it",
                    f"{BLOCK_PATH}.receivers[{index}]",
                )


def _parse_block(raw: Any) -> _Path:
    """The shape checks of one block, in the order a reader meets them: the
    object and its keys, no wrapper anywhere inside, the required fields, each
    field's own shape, then what the fields say together — every receiver
    strictly downstream of the sender, the two intervened models two names,
    and no two emitted sites at one address (:func:`_check_addresses`)."""
    path = BLOCK_PATH
    if not isinstance(raw, Mapping):
        raise _refuse("P2", "a path block is an object", path)
    for key in raw:
        if key not in BLOCK_KEYS:
            raise _refuse(
                "P3", f"unknown key {key!r}{suggest(key, BLOCK_KEYS)}", f"{path}.{key}"
            )
    # In the compile, ``families`` runs first and refuses an ``at_once`` inside
    # the block as rule 28 (it sits where it has no name identity), so only the
    # ``sweep`` half of this check is ever reached there; a direct caller of
    # :func:`expand_paths` gets both refusals from here.
    for keyword in (SWEEP_KEY, AT_ONCE_KEY):
        found = _find_key(raw, keyword, path)
        if found is not None:
            raise _refuse(
                "P2",
                f"a {keyword!r} wrapper inside a path block is not in this version: "
                "the freeze set is derived from the sender and receiver layers, so "
                "a wrapped layer would need one lowering per point — write the "
                "path out and wrap the emitted entries instead (§3.2)",
                found,
            )
    for key in REQUIRED_KEYS:
        if key not in raw:
            raise _refuse("P2", f"a path block needs {key!r}", path)
    sender, sender_layer = _site(raw["sender"], f"{path}.sender")
    receivers_raw = raw["receivers"]
    if not isinstance(receivers_raw, list) or not receivers_raw:
        raise _refuse(
            "P2",
            "'receivers' is a non-empty list of sites, in the order they are "
            "injected — one joint pass, so a set and not a sum",
            f"{path}.receivers",
        )
    receivers: list[Mapping[str, Any]] = []
    layers: list[int] = []
    for index, item in enumerate(receivers_raw):
        here = f"{path}.receivers[{index}]"
        site, layer = _site(item, here)
        if layer < sender_layer:
            raise _refuse(
                "P2",
                f"a path runs upstream: sender layer {sender_layer} is above "
                f"receiver layer {layer} — a receiver reads what the sender wrote, "
                "so it sits strictly below it",
                here,
            )
        if layer == sender_layer:
            raise _refuse(
                "P2",
                f"a receiver inside the sender's own layer ({layer}) is not in "
                "this version: the restoration policy freezes whole layers between "
                "sender and receiver, and a same-layer path has none",
                here,
            )
        receivers.append(site)
        layers.append(layer)
    restoration = raw["restoration"]
    if not isinstance(restoration, str):
        raise _refuse(
            "P2", f"'restoration' is one of {list(POLICIES)}", f"{path}.restoration"
        )
    if restoration not in POLICIES:
        raise _refuse(
            "P4",
            f"unknown restoration policy {restoration!r}"
            f"{suggest(restoration, POLICIES)}",
            f"{path}.restoration",
        )
    harvest = _word(raw, "harvest", path)
    inject = _word(raw, "inject", path)
    if harvest == inject:
        raise _refuse(
            "P2",
            f"'harvest' and 'inject' both name {inject!r}: the two intervened "
            "models would collapse into one entry and the harvest model — the "
            "one the receivers are read in — would vanish; give them two names",
            f"{path}.inject",
        )
    parsed = _Path(
        block=raw,
        sender=sender,
        sender_layer=sender_layer,
        source=_word(raw, "source", path),
        receivers=tuple(receivers),
        receiver_layers=tuple(layers),
        pos=raw["pos"],
        restoration=restoration,
        harvest=harvest,
        inject=inject,
    )
    _check_addresses(parsed)
    return parsed


# --------------------------------------------------------------------------- #
# the lowering
# --------------------------------------------------------------------------- #


def _emitted(path: _Path) -> dict[str, dict[str, Any]]:
    """The entries the block denotes, per section, in the order the shipped
    document writes them: the sender, the receivers, the restorers; their
    reads; the swap, the freezes, the injections; the two intervened models.

    Every read and write carries the block's ``pos`` verbatim. The restorer
    reads and the receiver reads are on ``base``: a freeze holds a site at the
    value the *clean* forward gives it, and the receiver is harvested in the
    harvest model on the same input the injection model runs on."""
    pos = copy.deepcopy(path.pos)
    sites: dict[str, Any] = path.sites
    reads: dict[str, Any] = {
        SENDER_READ: {
            "site": SENDER_SITE,
            "pos": pos,
            "model": "original",
            "input": path.source,
        }
    }
    writes: dict[str, Any] = {
        SENDER_WRITE: {
            "site": SENDER_SITE,
            "pos": copy.deepcopy(pos),
            "do": {"swap": SENDER_READ},
        }
    }
    freezes: list[str] = []
    for layer, component, name in path.restorers:
        reads[f"v_{name}"] = {
            "site": name,
            "pos": copy.deepcopy(pos),
            "model": "original",
            "input": "base",
        }
        freeze = path.freeze_name(component, layer)
        writes[freeze] = {
            "site": name,
            "pos": copy.deepcopy(pos),
            "do": {"swap": f"v_{name}"},
        }
        freezes.append(freeze)
    for name, inject in zip(path.receiver_names, path.inject_names):
        reads[f"v_{name}"] = {
            "site": name,
            "pos": copy.deepcopy(pos),
            "model": path.harvest,
            "input": "base",
        }
        writes[inject] = {
            "site": name,
            "pos": copy.deepcopy(pos),
            "do": {"swap": f"v_{name}"},
        }
    models: dict[str, Any] = {
        path.harvest: {"input": "base", "writes": [SENDER_WRITE, *freezes]},
        path.inject: {"input": "base", "writes": list(path.inject_names)},
    }
    return {
        "sites": sites,
        "reads": reads,
        "writes": writes,
        "intervened_models": models,
    }


def _merged(
    method: Mapping[str, Any],
    emitted: Mapping[str, Mapping[str, Any]],
    blame: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    """The method group with the block gone and the emitted entries in.

    An emitted table merges into the authored one of the same name —
    generated entries first, the author's after, which is the shipped
    document's own order — and a name the author also declares in that table
    is refused rather than overwritten (``blame`` maps an emitted
    ``(section, name)`` to the block key that chose the name, where one did:
    ``harvest`` and ``inject``; every other collision is the block's). An
    authored table that is not an object where the block emits into it is
    refused here too, by the table's path — the gate would say so by name,
    but the derived record is built first and would have nothing to point at.
    A table the author did not write is
    inserted where :data:`~causalab.protocol.schema.METHOD_SECTIONS` puts it,
    relative to the sections present, so the lowered document raises no
    order warning the authored one did not (rule 2)."""
    rank = {name: i for i, name in enumerate(METHOD_SECTIONS)}
    unknown = len(METHOD_SECTIONS)
    out: dict[str, Any] = {}
    pending = [section for section in emitted if section not in method]
    for section, table in method.items():
        if section == BLOCK:
            continue
        while pending and rank[pending[0]] < rank.get(section, unknown):
            out[pending[0]] = dict(emitted[pending[0]])
            pending.pop(0)
        if section in emitted:
            if not isinstance(table, Mapping):
                raise _refuse(
                    "P2",
                    f"the path block emits {section} entries "
                    f"{list(emitted[section])}, so 'method.{section}' has to be "
                    f"an object for them to merge into — it is a "
                    f"{type(table).__name__}",
                    f"method.{section}",
                )
            for name in emitted[section]:
                if name in table:
                    raise _refuse(
                        "P2",
                        f"the path block emits {section}.{name!r}, which the "
                        "document also declares — one of the two has to be "
                        "renamed",
                        blame.get((section, name), BLOCK_PATH),
                    )
            out[section] = {**emitted[section], **table}
        else:
            out[section] = table
    for section in pending:
        out[section] = dict(emitted[section])
    return out


def expand_paths(explicit: Mapping[str, Any]) -> dict[str, Any]:
    """The explicit tree with its path block lowered to the entries it denotes
    — the ``paths`` compile stage. A document without a block comes back as it
    was; one with a malformed block is refused with a :class:`ParseError`
    naming ``method.path_patching.<field>``."""
    if not has_path_block(explicit):
        return dict(explicit)
    method = explicit["method"]
    path = _parse_block(method[BLOCK])
    blame = {
        ("intervened_models", path.harvest): f"{BLOCK_PATH}.harvest",
        ("intervened_models", path.inject): f"{BLOCK_PATH}.inject",
    }
    return {
        **explicit,
        "method": _merged(method, _emitted(path), blame),
    }


def describe_paths(
    authored: Mapping[str, Any], explicit: Mapping[str, Any]
) -> dict[str, Any]:
    """The derived record of a lowering (spec §6): ``{}`` for a
    document without a block, else ``{"path_patching": {…}}`` with

    * ``authored`` — the block as it stood in ``authored``, the tree the stage
      lowered (overrides applied and artifact fields resolved: the explicit
      spelling, which for a block is also the authored one, since ``--set``
      cannot reach it);
    * ``restoration``, ``source``, ``pos``, ``harvest``, ``inject`` — the
      choices, with their defaults filled;
    * ``receivers`` — the emitted receiver site names **in authored order**.
      The order is data: it numbers the members (``receiver_0`` is the first
      authored), so it is in the canonical form as names; it is repeated here
      so a reader need not parse them (the intervened model's own write list
      is sorted in the canonical form);
    * ``restorers`` — the restorer boundary, ``[layer, component, site]`` per
      restored site, ordered by ``(layer, COMPONENT_RANK[component])``;
    * ``restored`` — per restored component, its layers: the choice to hold
      ``mlp_output`` at the sender's own layer, legible as data;
    * ``emitted`` — per section, the names of every entry the block lowered
      to, each a key of ``explicit``'s method group.

    Identity binding is by reference: the run receipt carries this beside the
    canonical ``method`` whose keys ``emitted`` names, and the document and
    point digests in the same receipt are over that canonical form."""
    if not has_path_block(authored):
        return {}
    path = _parse_block(authored["method"][BLOCK])
    emitted = _emitted(path)
    lowered = explicit.get("method")
    if not isinstance(lowered, Mapping):
        raise AssertionError("describe_paths runs on the lowered tree")
    for section, table in emitted.items():
        have = lowered.get(section)
        missing = [
            name for name in table if not isinstance(have, Mapping) or name not in have
        ]
        if missing:
            raise AssertionError(
                f"the lowered tree lacks emitted {section} entries {missing}"
            )
    return {
        BLOCK: {
            "authored": copy.deepcopy(dict(path.block)),
            "restoration": path.restoration,
            "source": path.source,
            "pos": copy.deepcopy(path.pos),
            "harvest": path.harvest,
            "inject": path.inject,
            "sender": SENDER_SITE,
            "receivers": list(path.receiver_names),
            "restorers": [list(item) for item in path.restorers],
            "restored": {
                component: list(layers)
                for component, layers in path.restored_layers.items()
            },
            "emitted": {section: list(table) for section, table in emitted.items()},
        }
    }
