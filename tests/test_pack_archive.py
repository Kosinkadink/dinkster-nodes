from __future__ import annotations

import sys
from pathlib import Path

import pytest
from blake3 import blake3
from dinkster_workers.pack_wheel import build_pack_archive

from scripts.build_official_packs import build as build_official_packs
from tests.async_scenarios import run_scenario


@pytest.fixture(scope="module")
def official_payloads(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ...]:
    # Consumers only read these wheels and unpack into their own test directories.
    return build_official_packs(
        Path(__file__).resolve().parents[1], tmp_path_factory.mktemp("official-payloads")
    )


def test_official_publication_declarations_match_archive_admission(
    tmp_path: Path, official_payloads: tuple[Path, ...]
) -> None:
    from dinkster_registry import Lockfile, unpack_artifact
    from dinkster_registry.archive_admission import run_ingest_integrity, run_manifest_admission

    from scripts.build_official_packs import library_metadata

    first = official_payloads
    catalog = Lockfile.from_record_json(first[-1].read_text())
    assert len(catalog.packs) == 37
    for pack in catalog.packs:
        package_name = pack.pack.replace("-", "_")
        archive = first[-1].parent / f"{package_name}-{pack.version}-py3-none-any.whl"
        data = archive.read_bytes()
        if pack.pack == "dinkster-pack-helpers":
            assert library_metadata(archive) == (
                "dinkster-pack-helpers",
                "0.0.3",
                (
                    "dinkster-api<0.1,>=0.0.2.dev0",
                    "dinkster-inference",
                    "numpy>=1.26",
                    "pillow>=10",
                    "torch",
                ),
            )
            assert f"blake3:{blake3(data).hexdigest()}" == pack.artifact_digest
            continue
        assert not run_ingest_integrity(data, len(data), pack.artifact_digest).blocked
        source = (
            unpack_artifact(archive, tmp_path / "sources" / pack.pack, pack.artifact_digest)
            / package_name
        )
        accepted = run_manifest_admission(
            source, pack.pack, pack.version, publisher="dinkster", engine_version="0.0.4.dev123"
        )
        assert not accepted.blocked, accepted.findings
        for engine in ("0.0.1", "0.1"):
            refused = run_manifest_admission(
                source, pack.pack, pack.version, publisher="dinkster", engine_version=engine
            )
            assert refused.blocked
            assert refused.findings[0].code == "manifest.invalid"


def test_official_sources_prepare_catalogs_and_execute_existing_node_ids(
    tmp_path: Path, official_payloads: tuple[Path, ...]
) -> None:
    import json
    from dataclasses import replace

    from dinkster_graph import graph_from_wire
    from dinkster_registry import Lockfile, artifact_digest, unpack_artifact
    from dinkster_workers import load_manifest
    from dinkster_workers.catalog import read_catalog
    from dinkster_workers.doctor import prepare_catalog
    from dinkster_workers.pack_wheel import pack_manifest_path

    from dinkster.compose import CompositionError, PackSpec, ServingComposer
    from dinkster.packs import pack_info_from_manifest

    saved = tmp_path / "workflow.json"
    original = (
        b'{"nodes":{"length":{"nodeType":"dinkster.text.string_length",'
        b'"inputs":{"string":"source packs"}},'
        b'"image":{"nodeType":"dinkster.image.empty_image",'
        b'"inputs":{"width":4,"height":5,"batch_size":2}},'
        b'"size":{"nodeType":"dinkster.image.get_image_size",'
        b'"inputs":{"image":{"$link":{"node":"image","output":"IMAGE"}}}}}}'
    )
    saved.write_bytes(original)
    payloads = official_payloads
    lock = Lockfile.from_record_json(payloads[-1].read_text())
    specs = []
    for entry in lock.packs:
        if entry.pack == "dinkster-pack-helpers":
            assert entry.claims == ()
            continue
        filename = f"{entry.pack.replace('-', '_')}-{entry.version}-py3-none-any.whl"
        source = unpack_artifact(
            payloads[-1].parent / filename,
            tmp_path / "sources" / entry.pack,
            entry.artifact_digest,
        )
        manifest = load_manifest(pack_manifest_path(source))
        assert read_catalog(manifest) is None
        report = prepare_catalog(manifest.path, environment={"PYTHONPATH": str(source)})
        assert report.ok, report
        assert read_catalog(manifest) is not None
        repacked = tmp_path / "repacked" / filename
        build_pack_archive(manifest.root, repacked)
        assert artifact_digest(repacked.read_bytes()) == entry.artifact_digest
        info = replace(
            pack_info_from_manifest(manifest),
            version=entry.version,
            artifact_digest=entry.artifact_digest,
            source=entry.source,
            publisher=entry.publisher,
        )
        specs.append(
            PackSpec(
                manifest=manifest.path,
                python=sys.executable,
                env={
                    "PYTHONPATH": str(source),
                    "DINKSTER_REMOTE_CATALOG_BASE": "",
                    "DINKSTER_REMOTE_GATEWAY_BASE": "",
                },
                packs={entry.pack: info},
                trust_reserved=True,
                require_catalog=True,
            )
        )

    async def execute() -> None:
        composer = ServingComposer()
        try:
            ordered = composer.order_pack_entries(specs)
            composer.validate_catalogs(ordered)
            for spec in ordered:
                await composer.add_pack(spec)
            composition = composer.composition
            assert composition.node_packs["dinkster.text.string_length"] == "dinkster-text"
            assert composition.node_packs["dinkster.image.empty_image"] == "dinkster-image"
            assert composition.packs["dinkster-text"].version == "0.0.3"
            assert composition.packs["dinkster-image"].version == "0.0.7"
            result = await composition.make_engine(lambda _event: None).run(
                graph_from_wire(json.loads(saved.read_bytes())), ["length", "size"]
            )
            assert result.diagnostics == ()
            assert result.outputs["length"]["length"].resolve() == 12
            assert result.outputs["size"]["width"].resolve() == 4
            assert result.outputs["size"]["height"].resolve() == 5
            assert result.outputs["size"]["batch_size"].resolve() == 2
            assert saved.read_bytes() == original
            source = tmp_path / "sources" / "dinkster-text" / "dinkster_text"
            (source / "changed.txt").write_text("source drift")
            with pytest.raises(CompositionError, match="missing or stale"):
                composer.validate_catalogs(ordered)
        finally:
            await composer.close()

    run_scenario(execute())
