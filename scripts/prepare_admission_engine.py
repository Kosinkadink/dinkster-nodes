"""Provision a verified published engine on its selected managed base."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from dinkster_store import object_path
from dinkster_supervisor.environment_artifacts import (
    EnvironmentCatalog,
    acquire,
    validate_dependency_base,
    validate_manifest,
)
from dinkster_supervisor.environment_constraints import requirement_hashes
from dinkster_supervisor.environment_generations import prepare_engine_interpreter
from dinkster_supervisor.environment_state import (
    Channel,
    content_digest,
    interpreter,
    writer_lock,
)
from dinkster_workers.pack_lock import GenerationPackages


def prepare_engine(
    release: Path,
    output: Path,
    store: Path,
    base_manifest: Path,
    *,
    engine_identity: str,
) -> Path:
    """Use published artifacts and the supervisor's verified generation constructor."""
    engine_data = (release / "release-manifest.json").read_bytes()
    engine: dict = validate_manifest(engine_data, engine_identity, None)
    if engine["repository"] != "Kosinkadink/Dinkster":
        raise ValueError("admission requires a published Dinkster engine")
    base_data = base_manifest.read_bytes()
    raw = json.loads(base_data)
    variant = raw["variant"]
    base: dict = validate_manifest(base_data, raw["id"], variant)
    validate_dependency_base(engine.get("dependencies", {}).get(variant), base)
    if engine["workerProtocol"] != base["workerProtocol"]:
        raise ValueError("engine and selected base worker protocols differ")
    for row in engine["artifacts"]:
        artifact = release / row["name"]
        if artifact.stat().st_size != row["size"] or content_digest(artifact) != (
            "sha256:" + row["sha256"]
        ):
            raise ValueError(f"release artifact differs from manifest: {artifact.name}")
    output.mkdir()
    prepared = output / "engine"
    shutil.copytree(release, prepared)
    document = {
        "format": "dinkster.channel/1",
        "engine": {
            "version": engine["version"],
            "release": engine_identity,
            "locations": [
                {"type": "mirror", "path": f"engine/{engine_identity[7:]}/release-manifest.json"}
            ],
        },
        "base": {
            variant: {
                "id": base["id"],
                "locations": [
                    {"type": "mirror", "path": f"base/{variant}/{base['id'][7:]}/manifest.json"}
                ],
            }
        },
        "minimumDesktopVersion": "0.0.0",
    }
    base_source = (
        f"https://releases.dinkster.org/base/{variant}/{base.get('label', base['id'][7:])}"
    )
    catalog = EnvironmentCatalog(
        "main",
        document,
        Channel.parse(document),
        engine,
        (str(prepared),),
        {variant: base},
        {variant: (base_source,)},
        engine_data,
    )
    store.mkdir(parents=True, exist_ok=True)
    with writer_lock(store):
        effective = acquire(catalog, variant, store)
        python = prepare_engine_interpreter(
            output / "generation", store=store, base=base["id"], constraints=effective
        )
    wheelhouse = output / "wheelhouse"
    wheelhouse.mkdir()
    sources = [prepared, object_path(store / "bases", base["id"]) / "wheels"]
    sources.extend(
        object_path(store / "wheels", "sha256:" + identity)
        for identity in requirement_hashes(effective.read_text())
    )
    for source in sources:
        for wheel in source.glob("*.whl"):
            link = wheelhouse / wheel.name
            if not link.exists():
                link.symlink_to(wheel.resolve())
    (output / "base-manifest.json").write_bytes(base_data)
    facts = GenerationPackages.inspect(python)
    (output / "runtime.json").write_text(
        json.dumps(
            {
                "commit": engine["commit"],
                "engineVersion": engine["version"],
                "engineRelease": engine_identity,
                "baseId": base["id"],
                "variant": variant,
                "python": str(python),
                "constraints": str(output / "generation/constraints.txt"),
                "constraintsSha256": content_digest(output / "generation/constraints.txt"),
                "packages": dict(facts.versions),
                "pythonVersion": facts.python_version,
            }
        )
        + "\n",
        encoding="ascii",
    )
    return python


def read_runtime(output: Path, engine_identity: str) -> tuple[Path, str, Path, str]:
    """Bind the interpreter and installed facts to the selected published engine."""
    runtime = json.loads((output / "runtime.json").read_bytes())
    if runtime["engineRelease"] != engine_identity:
        raise ValueError("admission runtime does not match the selected engine")
    engine: dict = validate_manifest(
        (output / "engine/release-manifest.json").read_bytes(), engine_identity, None
    )
    if (
        engine["repository"] != "Kosinkadink/Dinkster"
        or engine["commit"] != runtime["commit"]
        or engine["version"] != runtime["engineVersion"]
    ):
        raise ValueError("admission runtime engine identity differs")
    variant = runtime["variant"]
    base_id = runtime["baseId"]
    base: dict = validate_manifest((output / "base-manifest.json").read_bytes(), base_id, variant)
    validate_dependency_base(engine.get("dependencies", {}).get(variant), base)
    if engine["workerProtocol"] != base["workerProtocol"]:
        raise ValueError("engine and selected base worker protocols differ")
    python = interpreter(output / "generation")
    constraints = output / "generation/constraints.txt"
    if str(python) != runtime["python"] or str(constraints) != runtime["constraints"]:
        raise ValueError("admission runtime paths differ from its generation")
    if content_digest(constraints) != runtime["constraintsSha256"]:
        raise ValueError("admission runtime constraints changed")
    facts = GenerationPackages.inspect(python)
    if (
        dict(facts.versions) != runtime["packages"]
        or facts.python_version != runtime["pythonVersion"]
    ):
        raise ValueError("admission runtime installed packages changed")
    return python, base_id, constraints, variant


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--engine-identity", required=True)
    args = parser.parse_args()
    print(
        prepare_engine(
            args.release.resolve(),
            args.output.resolve(),
            args.store.resolve(),
            args.base_manifest.resolve(),
            engine_identity=args.engine_identity,
        )
    )
