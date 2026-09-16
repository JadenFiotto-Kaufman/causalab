"""Complete pairs with explicit family and split membership."""

import random

FAMILIES = ["broad", "narrow_b", "narrow_null", "narrow_output"]
SPLITS = ["train", "validation", "test"]
DATASET_ROLES = {
    f"{f}_{s}": {"family": f, "split": s} for f in FAMILIES for s in SPLITS
}


def _pairs(n, seed, offset, family):
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        a, b, da, db = [rng.randrange(3) for _ in range(4)]
        target = (da + b) % 3
        alternative = {
            "narrow_b": (a + db) % 3,
            "narrow_null": (a + b) % 3,
            "narrow_output": (da + db) % 3,
        }.get(family)
        if target == alternative:
            continue
        context = offset + len(rows) * 2
        rows.append(
            {
                "input": {"a": a, "b": b, "context": context},
                "counterfactual_inputs": [{"a": da, "b": db, "context": context + 1}],
            }
        )
    return rows


def make_datasets(model, n, seed):
    return {
        name: _pairs(
            n if role["split"] == "train" else max(1, n // 2),
            seed + i,
            i * 1_000_000,
            role["family"],
        )
        for i, (name, role) in enumerate(DATASET_ROLES.items())
    }


def random_pairs(model, n, seed):
    return _pairs(n, seed, 100_000_000, "broad")
