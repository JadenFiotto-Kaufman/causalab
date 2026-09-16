# CUDA graph workflow benchmark

`standard.json` runs the production Qwen3.6-35B-A3B bf16 workflow: locate,
DAS, DBM, random-subspace controls, mean harvest, and mean ablation. The included
months dataset is one group-disjoint split table for `natural_domains_arithmetic`
with `domain_type=months`: seed 0, a 60/40 train/test split over 64 unique
inputs, 39 training pairs and 25 test pairs (the committed bytes predate the
current builder and are the benchmark's fixed input; nothing sits beside a
table). Training settings come from the production method files.

The method is a paired timing. With model weights already cached, the same
workflow runs on a CUDA host once eager and once with CUDA graphs, each in a
fresh process, and the two are alternated (eager/graphs/graphs/eager for two
repeats) so neither order is favoured:

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 uv run causalab run benchmarks/cuda_graphs/standard.json \
  --device cuda --engine pytorch_hooks --data-root benchmarks/cuda_graphs/data --out out/eager-1
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 uv run causalab run benchmarks/cuda_graphs/standard.json \
  --device cuda --engine pytorch_hooks --data-root benchmarks/cuda_graphs/data --out out/graphs-1 --cuda-graphs
```

Use a fresh output directory per run. Wall time is the whole process:
startup, model loading, capture, training, evaluation, saving, and teardown.
Record optimizer update counts and capture/fallback counters beside it. Compare
the saved artifacts of an eager and a graphs run outside the timer; inspect
reported differences before claiming scientific parity. Execution metadata can
differ even when scientific values agree.

Cohort counters distinguish training replays/fallbacks and evaluation graph/eager
windows. Row counters separate active examples, remainder padding, and slots
whose members have stopped. `cohort_staging_host_nanoseconds` measures slot-copy
enqueue time on the host, not GPU elapsed time or first-use preparation.
`cohort_*_captured_pool_bytes` sums pool sizes at capture; it is not simultaneous
peak memory. Evaluation time includes scoring and capture when one is built.
Pass `--fit-rows N` to compare the bounded cohort path. If batching changes
early stopping, compare a separate study with fixed update counts as well.

For compilation policies and cache configuration, see the
[CUDA graph usage guide](../../docs/cuda_graphs.md#compilation-caches).
Keep study-specific sweep variants and generated results outside the repository.
