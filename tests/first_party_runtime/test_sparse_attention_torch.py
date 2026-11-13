"""CPU seams for BlockSparseAttention model patching."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_FLOAT,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_models.sparse_attention import (
    SparseAttnPatch,
    make_attention_override,
)


class _Sampling:
    def percent_to_sigma(self, percent: float) -> float:
        return 1.0 - percent


class _Model:
    def __init__(self) -> None:
        self.model_options = {"transformer_options": {}}
        self.callbacks: dict[tuple[Any, str], Any] = {}

    def clone(self) -> _Model:
        result = _Model()
        result.model_options = {
            "transformer_options": self.model_options["transformer_options"].copy()
        }
        return result

    def get_model_object(self, name: str) -> Any:
        return _Sampling() if name == "model_sampling" else object()

    def add_callback_with_key(self, _kind: Any, key: str, callback: Any) -> None:
        self.callbacks[_kind, key] = callback


class ManifestModel(_Model):
    def __init__(self) -> None:
        super().__init__()
        self.previous_calls = 0
        self.model_options["transformer_options"]["optimized_attention_override"] = self.previous

    def clone(self) -> ManifestModel:
        clone = ManifestModel()
        clone.model_options = {
            "transformer_options": self.model_options["transformer_options"].copy()
        }
        return clone

    def previous(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.previous_calls += 1
        return func(*args, **kwargs)


class ManifestSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sparse-attention-fixture.source",
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(MODEL=ManifestModel())


class ManifestReader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sparse-attention-fixture.reader",
            inputs=(
                InputSpec("source", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("patched", TypeExpr.concrete(MODEL_TYPE)),
            ),
            outputs=(OutputSpec("verified", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, source: ManifestModel, patched: ManifestModel) -> dict[str, object]:
        source_hook = source.model_options["transformer_options"]["optimized_attention_override"]
        assert getattr(source_hook, "__self__", None) is source
        assert not source.callbacks
        prepare, cleanup = patched.callbacks.values()
        replacement_calls: list[str] = []

        def replacement(func: Any, *args: Any, **kwargs: Any) -> Any:
            replacement_calls.append("replacement")
            return func(*args, **kwargs)

        options = {"transformer_options": {"optimized_attention_override": replacement}}
        prepare(patched, 0, options)
        installed = options["transformer_options"]["optimized_attention_override"]
        prepare(patched, 0, options)
        assert options["transformer_options"]["optimized_attention_override"] is installed
        q = torch.full((1, 5, 128), 1.0, dtype=torch.float32)
        k = torch.full((1, 5, 128), 2.0, dtype=torch.float32)
        v = torch.full((1, 5, 128), 3.0, dtype=torch.float32)
        result = installed(lambda q, k, v, *_args, **_kwargs: q + 2 * k + 3 * v, q, k, v, 1)
        torch.testing.assert_close(result, torch.full_like(q, 14.0))
        assert replacement_calls == ["replacement"]
        patch = next(
            value.cell_contents
            for value in cleanup.__closure__ or ()
            if isinstance(value.cell_contents, SparseAttnPatch)
        )
        patch.pooled[(1,)] = (q, q)
        patch.vsa_plans[(1,)] = {}
        cleanup(patched)
        assert not patch.pooled and not patch.vsa_plans and patch.vsa_rope is None
        options["transformer_options"]["optimized_attention_override"] = replacement
        prepare(patched, 1, options)
        reinstalled = options["transformer_options"]["optimized_attention_override"]
        torch.testing.assert_close(
            reinstalled(lambda q, k, v, *_args, **_kwargs: q + 2 * k + 3 * v, q, k, v, 1),
            torch.full_like(q, 14.0),
        )
        assert replacement_calls == ["replacement", "replacement"]
        return cls.outputs(verified=1.0)


MANIFEST_NODES = (ManifestSource, ManifestReader)


def _patch(*, minimum: int = 0) -> SparseAttnPatch:
    return SparseAttnPatch(1.3, 0.0, False, 1.0, 0.0, minimum, {2}, "off", 0, False)


def test_dense_fallback_preserves_previous_override_and_all_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dinkster_models.sparse_attention._runtime", lambda: (torch, object()))
    patch = _patch(minimum=8)
    calls: list[str] = []

    def base(*_args: Any, **_kwargs: Any) -> str:
        calls.append("base")
        return "base"

    def previous(func: Any, *args: Any, **kwargs: Any) -> str:
        calls.append("previous")
        return func(*args, **kwargs)

    override = make_attention_override(patch, previous)
    q = torch.zeros((1, 4, 128))
    for options in ({}, {"block_index": 2}, {"sigmas": torch.tensor([2.0])}):
        assert override(base, q, q, q, 1, transformer_options=options) == "base"
    assert override(base, q, q, q, 1, mask=torch.zeros((1, 1))) == "base"
    assert calls == ["previous", "base"] * 4
    assert patch.dense_reason({"sigmas": torch.tensor([1.0])}, 8, 0) is None
    assert patch.dense_reason({"sigmas": torch.tensor([0.0])}, 8, 0) is None
