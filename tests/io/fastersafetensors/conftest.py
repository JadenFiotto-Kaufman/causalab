from __future__ import annotations

import pytest
import torch


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA device")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)
