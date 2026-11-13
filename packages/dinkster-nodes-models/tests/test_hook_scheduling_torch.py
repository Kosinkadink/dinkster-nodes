"""CPU behavior for native hook scheduling and combination nodes."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_models.hook_scheduling import (
    CombineHooks2,
    CombineHooks4,
    CombineHooks8,
    CreateHookKeyframe,
    CreateHookKeyframesInterpolated,
)


def _hooks() -> Any:
    from dinkster_inference import hooks

    return hooks


def _group(strength: float) -> Any:
    hooks = _hooks()
    group = hooks.HookGroup()
    group.add(hooks.WeightHook(strength_model=strength))
    return group


def test_create_keyframes_preserves_source_and_interpolation_endpoints() -> None:
    initial = CreateHookKeyframe.execute(strength_mult=-1.25, start_percent=0.75)["HOOK_KF"]
    result = CreateHookKeyframesInterpolated.execute(
        strength_start=0.25,
        strength_end=3.25,
        interpolation="ease_in",
        start_percent=0.1,
        end_percent=0.7,
        keyframes_count=3,
        prev_hook_kf=initial,
    )["HOOK_KF"]

    assert [(keyframe.strength, keyframe.start_percent) for keyframe in initial.keyframes] == [
        (-1.25, 0.75)
    ]
    assert result is not initial
    assert [float(keyframe.strength) for keyframe in result.keyframes] == [0.25, 1.0, 3.25, -1.25]
    torch.testing.assert_close(
        torch.tensor([keyframe.start_percent for keyframe in result.keyframes]),
        torch.tensor([0.1, 0.4, 0.7, 0.75]),
    )
    assert [keyframe.guarantee_steps for keyframe in result.keyframes] == [1, 0, 0, 1]


@pytest.mark.parametrize(
    ("node", "count"),
    ((CombineHooks2, 2), (CombineHooks4, 4), (CombineHooks8, 8)),
)
def test_combine_hooks_preserves_singleton_and_clones_multi_group_inputs(
    node: Any, count: int
) -> None:
    groups = [_group(float(index + 1)) for index in range(count)]
    inputs = {
        f"hooks_{letter}": group for letter, group in zip("ABCDEFGH"[:count], groups, strict=True)
    }
    combined = node.execute(**inputs)["HOOKS"]

    assert len(combined.hooks) == count
    assert [hook.strength_model for hook in combined.hooks] == [
        float(index + 1) for index in range(count)
    ]
    assert all(
        result is not source.hooks[0] for result, source in zip(combined.hooks, groups, strict=True)
    )
    assert CombineHooks2.execute(hooks_A=groups[0], hooks_B=None)["HOOKS"] is groups[0]
    assert CombineHooks2.execute(hooks_A=None, hooks_B=None)["HOOKS"] is None
