"""Artifact I/O and plotting primitives.

This package is the single source of truth for code that touches disk:
- ``artifacts``: JSON / safetensors / pickle save/load, metadata, intervention
  and training result writers.
- ``plots``: shared figure rendering helpers.
- ``counterfactuals``: counterfactual dataset save/load.
- ``configs``: runner config save/load for notebook workflows.
- ``pipelines``: LMPipeline and analysis-result loaders.
- ``sae_checkpoints``: readers for *foreign* SAE checkpoints (the sanctioned
  ``torch.load(weights_only=False)`` exception) — vanilla decoder + block frame.
- ``artifact_viewer``: generic, spec-driven HTML viewer that renders a browsable
  page of experiment artifacts from a declarative ``viewer_spec.yaml``.

Dependency rule: ``causalab.io`` is the lowest application layer above
third-party libs. It imports ``causalab.protocol`` and ``causalab.causal`` and
nothing above itself — never ``causalab.workflow``, which is what
``tests/test_architecture_layering.py`` enforces as its invariant 1. (This rule
used to forbid ``causalab.methods``, ``causalab.analyses`` and
``causalab.runner``; all three were deleted in the protocol refactor, so naming
them made the rule read as satisfied while guarding nothing.)
"""
