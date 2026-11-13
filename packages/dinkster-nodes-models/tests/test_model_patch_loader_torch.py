"""CPU behavior coverage for ModelPatchLoader checkpoint dispatch."""

from __future__ import annotations

import pytest
from dinkster_models.model_patch_loader import (
    ModelPatchLoader,
)


def test_loader_rejects_unbound_asset() -> None:
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        ModelPatchLoader.execute(name=object())
