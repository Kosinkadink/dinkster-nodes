"""CPU behavior for Comfy model, CLIP, and VAE device routing."""
# pyright: standard

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any

import pytest
from dinkster_api.v1 import (
    CLIP_TYPE,
    CORE_STRING,
    MODEL_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_models.device_routing import (
    MultiGPUWorkUnits,
    SelectCLIPDevice,
    SelectModelDevice,
    SelectVAEDevice,
)

from dinkster.compose import PackSpec, ServingComposer, _installed_pack_spec

DEVICE_ROUTING_NODES = (MultiGPUWorkUnits, SelectModelDevice, SelectCLIPDevice, SelectVAEDevice)

_MODEL = TypeExpr.concrete(MODEL_TYPE)
_FLOAT32 = object()


class Model:
    pass


class Patcher:
    def __init__(
        self, load_device: object, offload_device: object, *, dynamic: bool = False
    ) -> None:
        self.model = Model()
        self.load_device = load_device
        self.offload_device = offload_device
        self.dynamic = dynamic
        self.additional: dict[str, list[Patcher]] = {}
        self.compute_dtype: object | None = None
        self.deepclone_calls = 0

    def clone(self, disable_dynamic: bool = False) -> Patcher:
        cloned = copy.copy(self)
        cloned.model = copy.copy(self.model)
        cloned.additional = {key: list(value) for key, value in self.additional.items()}
        cloned.dynamic = self.dynamic and not disable_dynamic
        return cloned

    def deepclone_multigpu(self, new_load_device: object) -> Patcher:
        self.deepclone_calls += 1
        routed = self.clone()
        routed.load_device = new_load_device
        return routed

    def is_dynamic(self) -> bool:
        return self.dynamic

    def model_dtype(self) -> object:
        return _FLOAT32

    def set_model_compute_dtype(self, dtype: object) -> None:
        self.compute_dtype = dtype

    def get_additional_models_with_key(self, key: str) -> list[Patcher]:
        return self.additional.get(key, [])

    def set_additional_models(self, key: str, models: list[Patcher]) -> None:
        self.additional[key] = models

    def match_multigpu_clones(self) -> None:
        return None


class VAE:
    def __init__(self, patcher: Patcher, device: object) -> None:
        self.patcher = patcher
        self.device = device
        self.first_stage_model = patcher.model


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="device-routing-fixture.source",
            outputs=(
                OutputSpec("model", _MODEL),
                OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        import torch
        from dinkster_inference.model_patcher import ModelPatcher
        from dinkster_inference.sd import CLIP, VAE

        def patcher() -> Any:
            model = torch.nn.Linear(2, 1)
            model.weight = torch.nn.Parameter(torch.tensor([[2.0, -3.0]]))
            model.bias = torch.nn.Parameter(torch.tensor([0.25]))
            return ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))

        clip = CLIP(no_init=True)
        clip.patcher = patcher()
        clip.cond_stage_model = clip.patcher.model
        clip.tokenizer = None
        clip.layer_idx = None
        clip.tokenizer_options = {}
        clip.use_clip_schedule = False
        clip.apply_hooks_to_conds = None
        vae = VAE.__new__(VAE)
        vae.patcher = patcher()
        vae.device = torch.device("cpu")
        vae.first_stage_model = vae.patcher.model
        return dict(cls.outputs(model=patcher(), clip=clip, vae=vae))


class Consumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="device-routing-fixture.consumer",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(type_id))
                for name, type_id in (
                    ("model", MODEL_TYPE),
                    ("original_model", MODEL_TYPE),
                    ("clip", CLIP_TYPE),
                    ("original_clip", CLIP_TYPE),
                    ("vae", VAE_TYPE),
                    ("original_vae", VAE_TYPE),
                )
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        original_model: Any,
        clip: Any,
        original_clip: Any,
        vae: Any,
        original_vae: Any,
    ) -> dict[str, object]:
        import torch
        from dinkster_inference.model_patcher import ModelPatcher
        from dinkster_inference.sd import CLIP, VAE

        assert type(model) is ModelPatcher
        assert type(clip) is CLIP
        assert type(vae) is VAE
        assert model is not original_model
        assert clip is not original_clip and clip.patcher is not original_clip.patcher
        assert vae is not original_vae and vae.patcher is not original_vae.patcher
        assert vae.first_stage_model is vae.patcher.model
        assert vae.device.type == original_vae.device.type == "cpu"
        for routed, source in (
            (model, original_model),
            (clip.patcher, original_clip.patcher),
            (vae.patcher, original_vae.patcher),
        ):
            assert routed.load_device.type == routed.offload_device.type == "cpu"
            assert source.load_device.type == source.offload_device.type == "cpu"
            assert torch.equal(source.model.weight, torch.tensor([[2.0, -3.0]]))
            assert routed.model(torch.tensor([[1.5, -2.0]])).tolist() == [[9.25]]
        assert model.get_additional_models_with_key("multigpu") == []
        return dict(cls.outputs(result="routed"))


@pytest.mark.parametrize("in_process", (True, False))
def test_all_device_nodes_run_twice_through_both_production_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        module = "device_routing_fixture_" + str(int(in_process))
        (tmp_path / f"{module}.py").write_text(
            "from test_device_routing_torch import Consumer, Source\n"
            "NODES = (Source, Consumer)\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "\ndef register_types(registry):\n"
            "    register_inference_types(registry)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            "[pack]\n"
            'name = "device-routing-fixture"\n'
            'publisher = "dinkster"\n'
            'namespaces = ["device-routing-fixture"]\n\n'
            "[pack.entry]\n"
            f'nodes = "{module}:NODES"\n'
            f'types = "{module}:register_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": str(Path(__file__).parent),
            }
        )
        try:
            models = _installed_pack_spec(
                "dinkster-models", "dinkster_models", in_process=in_process
            )
            manifests = (Path(models.manifest), fixture)
            worker_group = "device-routing" if not in_process else None
            await composer.add_pack(
                PackSpec(
                    manifest=models.manifest,
                    packs=models.packs,
                    trust_reserved=True,
                    in_process=in_process,
                    worker_group=worker_group,
                    group_manifests=manifests if worker_group else (),
                )
            )
            await composer.add_pack(
                PackSpec(
                    manifest=fixture,
                    in_process=in_process,
                    worker_group=worker_group,
                    group_manifests=manifests if worker_group else (),
                )
            )
            graph = Graph(
                nodes={
                    "source": GraphNode("device-routing-fixture.source"),
                    "split": GraphNode(
                        "dinkster.models.multi_gpu_work_units",
                        {"model": Link("source", "model"), "max_gpus": 3},
                    ),
                    "route": GraphNode(
                        "dinkster.models.select_model_device",
                        {"model": Link("split", "MODEL"), "device": "cpu"},
                    ),
                    "clip": GraphNode(
                        "dinkster.models.select_clip_device",
                        {"clip": Link("source", "clip"), "device": "cpu"},
                    ),
                    "vae": GraphNode(
                        "dinkster.models.select_vae_device",
                        {"vae": Link("source", "vae"), "device": "default"},
                    ),
                    "consumer": GraphNode(
                        "device-routing-fixture.consumer",
                        {
                            "model": Link("route", "MODEL"),
                            "original_model": Link("source", "model"),
                            "clip": Link("clip", "CLIP"),
                            "original_clip": Link("source", "clip"),
                            "vae": Link("vae", "VAE"),
                            "original_vae": Link("source", "vae"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            first = await engine.run(graph, ["consumer"])
            assert first.outputs["consumer"]["result"].resolve() == "routed"
            second = await engine.run(graph, ["consumer"])
            assert second.outputs["consumer"]["result"].resolve() == "routed"
            assert set(second.cached) == set(graph.nodes)
        finally:
            await composer.close()

    asyncio.run(scenario())
