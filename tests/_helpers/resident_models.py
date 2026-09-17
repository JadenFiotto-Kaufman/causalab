"""Evict every model the engine loaders hold resident.

Both engines load through a four-entry ``normalized_cache``
(``causalab.neural.shared.normalized_cache``; the ``load_model`` of
``causalab.neural.engines.pytorch_hooks.loading`` and of
``causalab.neural.engines.nnsight_nnterp.loading``), so a model a test loaded
— directly, or through a protocol run — stays *alive* after the test ends. The
per-test ``gc.collect()`` + ``torch.cuda.empty_cache()`` in
``tests/conftest.py`` reclaims only *dead* models; nothing it does can
free a cached one. The golden tier runs every module in one process on one
accelerator, so whatever the previous module left cached is subtracted from
the next module's budget — enough for the paper goldens' cached GPT-2 XL and
Llama-3.1-8B to cost ``tests/golden/test_readout_a3b.py`` its ~70 GB
``Qwen/Qwen3.6-35B-A3B`` load. ``tests/golden/conftest.py`` calls this at
every golden module boundary.
"""

from __future__ import annotations

import gc

import torch


def evict_resident_models() -> None:
    """Clear both loader caches, collect the freed graphs, hand VRAM back.

    The imports are deferred so importing this helper stays cheap for the CPU
    tiers and never pulls the nnsight stack into a process that does not use it.
    Order matters: the caches must drop their references *before* the
    collection pass, and the collection must run before ``empty_cache`` has
    anything to return to the driver.
    """
    from causalab.neural.engines.nnsight_nnterp.loading import (
        load_model as nnterp_load_model,
    )
    from causalab.neural.engines.pytorch_hooks.loading import (
        load_model as hooks_load_model,
    )

    hooks_load_model.cache_clear()
    nnterp_load_model.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
