"""Numerical analysis a workflow ``script`` step can run.

Most modules here are deterministic step scripts with ``main(inputs, outputs)``
addressed from a workflow document as
``{"script": {"module": "causalab.analysis.<name>"}}``. They are ordinary
Python, importable and testable without the workflow layer, which is the point:
a document names them, but nothing about them depends on being named.

| module | what it computes | produces |
|---|---|---|
| ``fit_pca`` | a principal basis over a saved read, by full SVD | tensor + table |
| ``harvest_difference`` | a steering direction as the difference of two harvest means | tensor + table |
| ``head_stats`` | mean and spread of a metric per (layer, head) cell | table |
| ``paired_ttest`` | a two-sided paired t-test of two metric tables | table |
| ``random_mask`` | a size-matched random gate mask from a fitted one, the DBM control | tensor |
| ``subspace_angles`` | principal angles between two saved bases (a fit's span against a PCA block, another fit, another seed) — the identifiable comparison of subspaces | table |
| ``certify_control`` | the three certification legs of a self-swap control, from its saved reads | table |
| ``compare_hypotheses`` | saved top-1 outputs against frozen symbolic predictions, joined by example ID | per-pair comparison table |

These are **not** the retired ``causalab/methods/`` — that was
interventions-as-Python, and interventions are documents now
(``docs/intervention_protocol.md``). What lives here is the numerics that no
intervention vocabulary can express: fits, statistics, and the operands a
later intervention consumes.

``export_dbm.export(manifest_path, register_from_hf=False)`` is a standalone
API for validated DBM apply results. It returns masks, metrics and provenance.

Numerics are imported inside the functions that use them, so listing or
hashing a script costs nothing but stdlib (``tests/test_architecture_layering.py``).
"""

from __future__ import annotations

__all__: list[str] = []
