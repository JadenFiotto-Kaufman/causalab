"""What every execution engine uses — and none may fork.

The §8 services that are protocol work, not hook work: tokenization and the
batch frame (:mod:`.encoding`), contract layouts (:mod:`.layout`), payload
math (:mod:`.mechanisms`), metric lowering (:mod:`.metrics`), artifact
writing and stamping (:mod:`.outputs`), bundle loading, role resolution and
identity records (:mod:`.services`), the per-layer hybrid stream table
(:mod:`.streams`), the component vocabulary's module taps (:mod:`.sites`),
the loaders' dtype table and envoy unwrap (:mod:`.loading`), and the
on-demand eager-attention switch (:mod:`.attention_backend`). An engine owns *loading and execution*; everything here is
the shared remainder, single-homed so two engines can never disagree about it.
"""
