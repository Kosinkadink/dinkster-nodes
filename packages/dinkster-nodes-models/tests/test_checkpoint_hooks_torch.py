"""Checkpoint hook behavior with actual CPU patchers and controlled loader I/O."""

from __future__ import annotations

import pytest
from dinkster_models.hook_checkpoint import (
    CreateHookModelAsLoraModelOnly,
)


def test_checkpoint_hook_rejects_unbound_filename() -> None:
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        CreateHookModelAsLoraModelOnly.execute(ckpt_name="unbound.ckpt", strength_model=1.0)
