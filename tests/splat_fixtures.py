"""Deterministic splat inputs for pack I/O tests."""

from typing import Any

import numpy


def splat_value(
    batch: int = 1, gaussians: int = 5, coefficients: int = 4, counts: list[int] | None = None
) -> dict[str, Any]:
    rng = numpy.random.default_rng(7)
    rotations = rng.normal(0.0, 1.0, (batch, gaussians, 4)).astype(numpy.float32)
    rotations /= numpy.linalg.norm(rotations, axis=2, keepdims=True)
    value: dict[str, Any] = {
        "positions": rng.uniform(-0.5, 0.5, (batch, gaussians, 3)).astype(numpy.float32),
        "scales": rng.uniform(0.005, 0.05, (batch, gaussians, 3)).astype(numpy.float32),
        "rotations": rotations,
        "opacities": rng.uniform(0.05, 0.95, (batch, gaussians, 1)).astype(numpy.float32),
        "sh": rng.normal(0.0, 1.0, (batch, gaussians, coefficients, 3)).astype(numpy.float32),
    }
    if counts is not None:
        value["counts"] = numpy.asarray(counts, numpy.int64)
    return value
