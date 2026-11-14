"""CPU behavior for native sampler factory nodes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    initialize_inference_process,
)
from dinkster_sampling.sampler_factories import (
    KSamplerSelect,
    SamplerDPMAdaptative,
    SamplerDPMPP_2M_SDE,
    SamplerDPMPP_2S_Ancestral,
    SamplerDPMPP_3M_SDE,
    SamplerDPMPP_SDE,
    SamplerER_SDE,
    SamplerEulerAncestral,
    SamplerEulerAncestralCFGPP,
    SamplerLMS,
    SamplerSASolver,
    SamplerSEEDS2,
)


class ModelSampling:
    def percent_to_sigma(self, percent: float) -> float:
        return 3.0 - percent * 2.0


class Model:
    def get_model_object(self, name: str) -> ModelSampling:
        assert name == "model_sampling"
        return ModelSampling()


def _sampler(node: type[Any], **inputs: Any) -> Any:
    return node.execute(**inputs)["SAMPLER"]


def test_sampler_factories_construct_pinned_runtime_samplers_with_exact_options() -> None:
    cases = (
        (KSamplerSelect, {"sampler_name": "euler"}, "sample_euler", {}),
        (
            SamplerEulerAncestral,
            {"eta": 0.375, "s_noise": 1.25},
            "sample_euler_ancestral",
            {"eta": 0.375, "s_noise": 1.25},
        ),
        (
            SamplerEulerAncestralCFGPP,
            {"eta": 0.625, "s_noise": 2.5},
            "sample_euler_ancestral_cfg_pp",
            {"eta": 0.625, "s_noise": 2.5},
        ),
        (SamplerLMS, {"order": 7}, "sample_lms", {"order": 7}),
        (
            SamplerDPMPP_3M_SDE,
            {"eta": 0.4, "s_noise": 1.3, "noise_device": "cpu"},
            "sample_dpmpp_3m_sde",
            {"eta": 0.4, "s_noise": 1.3},
        ),
        (
            SamplerDPMPP_2M_SDE,
            {"solver_type": "heun", "eta": 0.4, "s_noise": 1.3, "noise_device": "cpu"},
            "sample_dpmpp_2m_sde",
            {"eta": 0.4, "s_noise": 1.3, "solver_type": "heun"},
        ),
        (
            SamplerDPMPP_SDE,
            {"eta": 0.4, "s_noise": 1.3, "r": 0.27, "noise_device": "cpu"},
            "sample_dpmpp_sde",
            {"eta": 0.4, "s_noise": 1.3, "r": 0.27},
        ),
        (
            SamplerDPMPP_2S_Ancestral,
            {"eta": 0.4, "s_noise": 1.3},
            "sample_dpmpp_2s_ancestral",
            {"eta": 0.4, "s_noise": 1.3},
        ),
        (
            SamplerDPMAdaptative,
            {
                "order": 2,
                "rtol": 0.04,
                "atol": 0.005,
                "h_init": 0.07,
                "pcoeff": 0.2,
                "icoeff": 0.8,
                "dcoeff": 0.1,
                "accept_safety": 0.75,
                "eta": 0.3,
                "s_noise": 1.1,
            },
            "dpm_adaptive_function",
            {
                "order": 2,
                "rtol": 0.04,
                "atol": 0.005,
                "h_init": 0.07,
                "pcoeff": 0.2,
                "icoeff": 0.8,
                "dcoeff": 0.1,
                "accept_safety": 0.75,
                "eta": 0.3,
                "s_noise": 1.1,
            },
        ),
        (
            SamplerER_SDE,
            {"solver_type": "ER-SDE", "max_stage": 2, "eta": 0.5, "s_noise": 1.3},
            "sample_er_sde",
            {"s_noise": 1.3, "max_stage": 2},
        ),
        (
            SamplerSASolver,
            {
                "model": Model(),
                "eta": 0.75,
                "sde_start_percent": 0.25,
                "sde_end_percent": 0.75,
                "s_noise": 1.4,
                "predictor_order": 2,
                "corrector_order": 3,
                "use_pece": True,
                "simple_order_2": True,
            },
            "sample_sa_solver",
            {
                "s_noise": 1.4,
                "predictor_order": 2,
                "corrector_order": 3,
                "use_pece": True,
                "simple_order_2": True,
            },
        ),
        (
            SamplerSEEDS2,
            {"solver_type": "phi_2", "eta": 0.4, "s_noise": 1.3, "r": 0.27},
            "sample_seeds_2",
            {"eta": 0.4, "s_noise": 1.3, "r": 0.27, "solver_type": "phi_2"},
        ),
    )
    for node, inputs, function_name, expected_options in cases:
        sampler = _sampler(node, **inputs)
        assert callable(sampler.sampler_function)
        assert sampler.sampler_function.__name__ == function_name
        for name, value in expected_options.items():
            assert sampler.extra_options[name] == value


def test_er_sde_and_sa_solver_closures_preserve_solver_boundaries() -> None:
    er_sde = _sampler(SamplerER_SDE, solver_type="ER-SDE", max_stage=2, eta=0.5, s_noise=1.3)
    reverse = _sampler(
        SamplerER_SDE, solver_type="Reverse-time SDE", max_stage=2, eta=0.5, s_noise=1.3
    )
    ode = _sampler(SamplerER_SDE, solver_type="ER-SDE", max_stage=2, eta=0.0, s_noise=1.3)
    values = torch.tensor((0.25, 2.0), dtype=torch.float32)
    torch.testing.assert_close(
        er_sde.extra_options["noise_scaler"](values),
        values * ((values**0.3).exp() + 10.0) ** 0.5,
    )
    torch.testing.assert_close(reverse.extra_options["noise_scaler"](values), values**1.5)
    assert ode.extra_options["s_noise"] == 0.0
    assert ode.extra_options["noise_scaler"](values) is values

    solver = _sampler(
        SamplerSASolver,
        model=Model(),
        eta=0.75,
        sde_start_percent=0.25,
        sde_end_percent=0.75,
        s_noise=1.4,
        predictor_order=2,
        corrector_order=3,
        use_pece=True,
        simple_order_2=True,
    )
    tau = solver.extra_options["tau_func"]
    assert tau(2.5) == 0.75
    assert tau(1.5) == 0.75
    assert tau(0.5) == 0.0

    explicit_ode = _sampler(SamplerER_SDE, solver_type="ODE", max_stage=1, eta=0.9, s_noise=2.0)
    assert explicit_ode.extra_options["s_noise"] == 0.0
    assert explicit_ode.extra_options["noise_scaler"](values) is values


def test_sde_noise_device_gpu_names_are_selected_without_gpu_execution() -> None:
    assert (
        _sampler(
            SamplerDPMPP_3M_SDE, eta=0.4, s_noise=1.3, noise_device="gpu"
        ).sampler_function.__name__
        == "sample_dpmpp_3m_sde_gpu"
    )
    assert (
        _sampler(
            SamplerDPMPP_2M_SDE,
            solver_type="heun",
            eta=0.4,
            s_noise=1.3,
            noise_device="gpu",
        ).sampler_function.__name__
        == "sample_dpmpp_2m_sde_gpu"
    )
    assert (
        _sampler(
            SamplerDPMPP_SDE, eta=0.4, s_noise=1.3, r=0.27, noise_device="gpu"
        ).sampler_function.__name__
        == "sample_dpmpp_sde_gpu"
    )


def test_euler_ancestral_factory_runs_with_reordered_runtime_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_inference_process()
    from dinkster_inference import samplers

    monkeypatch.setattr(samplers, "SAMPLER_NAMES", tuple(reversed(samplers.SAMPLER_NAMES)))
    sampler = _sampler(SamplerEulerAncestral, eta=0.375, s_noise=1.25)
    noise = torch.tensor([[[[0.2, -0.4], [1.1, -0.8]]]], dtype=torch.float32)

    class SamplingModel:
        inner_model = SimpleNamespace(inner_model=SimpleNamespace(model_sampling=object()))

        def __call__(
            self, value: torch.Tensor, _sigma: torch.Tensor, **_kwargs: Any
        ) -> torch.Tensor:
            return torch.full_like(value, 0.25)

    result = sampler.sampler_function(
        SamplingModel(),
        noise,
        torch.tensor((1.5, 0.75)),
        extra_args={"seed": 739},
        disable=True,
        **sampler.extra_options,
    )
    sigma_up = 0.375 * (0.75**2 * (1.5**2 - 0.75**2) / 1.5**2) ** 0.5
    sigma_down = (0.75**2 - sigma_up**2) ** 0.5
    ancestral_noise = torch.randn_like(noise, generator=torch.Generator().manual_seed(740))
    expected = noise * (sigma_down / 1.5) + 0.25 * (1 - sigma_down / 1.5)
    torch.testing.assert_close(result, expected + ancestral_noise * 1.25 * sigma_up)
