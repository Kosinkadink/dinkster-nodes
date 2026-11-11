"""Public node packs compose, reload and execute under their declared namespaces."""

from __future__ import annotations

from pathlib import Path

import pytest
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral
from dinkster_workers import load_manifest

from dinkster.compose import CompositionError, PackSpec, ServingComposer
from tests.async_scenarios import run_scenario

TESTS_DIR = Path(__file__).parent


@pytest.mark.parametrize("in_process", (False, True))
def test_pack_namespace_compose_reload_execute_and_reject_second_owner(
    tmp_path: Path, in_process: bool
) -> None:
    from dinkster_utilities import NODES

    claims = tuple(node.schema().node_type for node in NODES)
    manifest = TESTS_DIR.parent / "packages/dinkster-nodes-utilities/dinkster-pack.toml"
    assert load_manifest(manifest).namespaces == ("dinkster.utilities",)
    other_manifest = tmp_path / "dinkster-pack.toml"
    other_manifest.write_text(
        '[pack]\nname = "other-owner"\npublisher = "dinkster"\n'
        'namespaces = ["dinkster.utilities"]\n'
        '[pack.entry]\nnodes = "other_owner_fixture:NODES"\n'
    )
    (tmp_path / "other_owner_fixture.py").write_text(
        "from dinkster_utilities.primitive import PrimitiveString\nNODES = (PrimitiveString,)\n"
    )

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(
                PackSpec(manifest=manifest, in_process=in_process, trust_reserved=True)
            )
            assert set(composer.composition.schemas) == set(claims)
            assert set(composer.composition.node_packs.values()) == {"dinkster-utilities"}
            assert not composer.composition.lazy_choices
            if in_process:
                with pytest.raises(CompositionError, match="cannot reload in-process pack"):
                    await composer.reload_pack("dinkster-utilities")
            else:
                await composer.reload_pack("dinkster-utilities")
            assert set(composer.composition.schemas) == set(claims)
            assert not composer.composition.lazy_choices
            await composer.add_pack(
                PackSpec(
                    manifest=TESTS_DIR.parent / "packages/dinkster-nodes-text/dinkster-pack.toml",
                    in_process=in_process,
                    trust_reserved=True,
                )
            )
            combined_claims = set(composer.composition.schemas)
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "text": GraphNode(
                        "dinkster.utilities.primitive_string", {"value": "a\U0001f600"}
                    ),
                    "length": GraphNode(
                        "dinkster.text.string_length", {"string": Link("text", "STRING")}
                    ),
                    "false": GraphNode("dinkster.utilities.primitive_boolean", {"value": False}),
                    "not": GraphNode(
                        "dinkster.utilities.comfy_not_node", {"value": Link("length", "length")}
                    ),
                    "and": GraphNode(
                        "dinkster.utilities.comfy_and_node",
                        {
                            "values.value0": Link("length", "length"),
                            "values.value1": Link("false", "BOOLEAN"),
                        },
                    ),
                    "or": GraphNode(
                        "dinkster.utilities.comfy_or_node",
                        {
                            "values.value0": Link("false", "BOOLEAN"),
                            "values.value1": Link("length", "length"),
                        },
                    ),
                    "combo_default": GraphNode("dinkster.utilities.custom_combo", {"choice": ""}),
                    "combo_index": GraphNode("dinkster.utilities.primitive_int", {"value": 205}),
                    "combo": GraphNode(
                        "dinkster.utilities.custom_combo",
                        {
                            "choice": "outside-menu",
                            "index": Link("combo_index", "INT"),
                            "2": TypedLiteral("STRING", "unused sparse option"),
                            "205": Link("text", "STRING"),
                            "label": TypedLiteral("STRING", "arbitrary extra name"),
                            **{str(i): TypedLiteral("INT", i) for i in range(300, 405)},
                        },
                    ),
                }
            )
            targets = ["length", "not", "and", "or", "combo_default", "combo"]
            result = await engine.run(graph, targets)
            assert result.outputs["length"]["length"].resolve() == 2
            assert result.outputs["not"]["BOOLEAN"].resolve() is False
            assert result.outputs["and"]["BOOLEAN"].resolve() is False
            assert result.outputs["or"]["BOOLEAN"].resolve() is True
            assert result.outputs["combo_default"]["STRING"].resolve() == ""
            assert result.outputs["combo_default"]["INDEX"].resolve() == 0
            assert result.outputs["combo"]["STRING"].resolve() == "outside-menu"
            assert result.outputs["combo"]["INDEX"].resolve() == 205
            repeated = await engine.run(graph, targets)
            assert repeated.executed == ()
            assert "combo" in repeated.cached
            assert repeated.outputs["combo"]["INDEX"].resolve() == 205
            changed = Graph(
                nodes={
                    **graph.nodes,
                    "combo_index": GraphNode("dinkster.utilities.primitive_int", {"value": -7}),
                }
            )
            changed_result = await engine.run(changed, ["combo"])
            assert changed_result.outputs["combo"]["STRING"].resolve() == "outside-menu"
            assert changed_result.outputs["combo"]["INDEX"].resolve() == -7
            assert "combo" in changed_result.executed
            with pytest.raises(
                CompositionError, match="declared by both 'dinkster-utilities' and 'other-owner'"
            ):
                await composer.add_pack(PackSpec(manifest=other_manifest, trust_reserved=True))
            assert set(composer.composition.schemas) == combined_claims
        finally:
            await composer.close()

    run_scenario(scenario())


def test_installed_pack_digest_matches_bundled_artifact(tmp_path: Path) -> None:
    from dinkster_registry import artifact_digest, unpack_artifact
    from dinkster_workers.pack_wheel import build_pack_archive

    from dinkster import compose

    source = Path(__file__).parent.parent / "packages/dinkster-nodes-image"
    output = tmp_path / "wheel"
    build_pack_archive(source, output)
    archive = next(output.glob("*.whl"))
    unpack_artifact(archive, tmp_path, artifact_digest(archive.read_bytes()))
    bundled = tmp_path / "dinkster_image"

    source_digest = compose._installed_pack_digest(
        source / "dinkster-pack.toml", source / "src/dinkster_image"
    )
    bundled_digest = compose._installed_pack_digest(bundled / "dinkster-pack.toml", None)

    assert source_digest == bundled_digest
