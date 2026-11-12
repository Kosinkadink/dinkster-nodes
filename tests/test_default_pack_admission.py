from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest


@pytest.mark.parametrize("candidate_mode", ["build", "native"])
@pytest.mark.parametrize(
    "problem",
    [
        None,
        "engine-version",
        "engine-release",
        "engine-commit",
        "source-commit",
        "artifact",
        "lock",
        "missing-outer",
        "security",
    ],
)
def test_admission_prepared_sources_bind_pack_source_and_published_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str | None, candidate_mode: str
) -> None:
    from dinkster_registry import LockedPack, Lockfile, artifact_digest
    from dinkster_registry.archive_admission import StageResult
    from dinkster_store import pack_archive_path
    from dinkster_workers.pack_lock import LockedDistribution, LockedWheel, PackLock
    from dinkster_workers.pack_wheel import build_pack_archive

    from scripts import check_default_pack_admission as admission

    if sys.platform != "win32":
        temporary = tmp_path / "physical-temporary"
        temporary.mkdir()
        alias = tmp_path / "temporary-parent"
        alias.symlink_to(temporary, target_is_directory=True)
        monkeypatch.setattr(admission.tempfile, "tempdir", str(alias))

    identity = "sha256:" + "e" * 64
    runtime = tmp_path / "runtime"
    (runtime / "engine").mkdir(parents=True)
    (runtime / "engine/release-manifest.json").write_text(
        json.dumps({"version": "0.2", "commit": "c" * 40})
    )
    constraints = runtime / "constraints.txt"
    constraints.touch()
    python = Path(sys.executable)

    def verified_runtime(output, engine_identity):
        assert output == runtime
        assert engine_identity == identity
        return python, "sha256:" + "f" * 64, constraints, "linux-cpu"

    monkeypatch.setattr(admission, "read_runtime", verified_runtime)
    original = subprocess.check_output

    def selected_minimums(command, **kwargs):
        if command[0] == str(python):
            assert command[1:3] == ["-I", "-c"]
            return json.dumps({"candidate": "1.0", "unchanged": "1.0"})
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", selected_minimums)
    monkeypatch.setattr(admission, "publication_selection", lambda *args: ("candidate",))
    monkeypatch.setattr(admission.StaticPackIndex, "catalog", lambda *args, **kwargs: [])
    archives = {}
    entries = {}
    for name, version in (("candidate", "7.2"), ("unchanged", "1.3")):
        source = tmp_path / name
        (source / name).mkdir(parents=True)
        (source / name / "__init__.py").write_text("NODES = []\n")
        (source / "dinkster-pack.toml").write_text(
            f'[pack]\nname = "{name}"\npublisher = "dinkster"\n'
            f'version = "{version}"\nengine = ">=0.2,<1"\n'
            f'[pack.entry]\nnodes = "{name}:NODES"\n'
            '[pack.contracts]\nhost = "dinkster-pack-host/1"\napi = "dinkster-api/v1"\n'
        )
        if problem == "security" and name == "candidate":
            (source / "bad.txt").write_text("dinkster-known-bad-test-payload")
        archive = tmp_path / f"{name}-{version}-py3-none-any.whl"
        build_pack_archive(source, archive)
        archives[name] = archive.read_bytes()
        entries[name] = LockedPack(name, version, artifact_digest(archives[name]), "dinkster", ())

    def dependency_lock(name):
        payload = archives[name]
        entry = entries[name]
        return PackLock(
            name,
            "linux-cpu",
            ">=3.12",
            (),
            (),
            (
                LockedDistribution(
                    name,
                    entry.version,
                    (
                        LockedWheel(
                            f"{name}-{entry.version}-py3-none-any.whl",
                            len(payload),
                            hashlib.sha256(payload).hexdigest(),
                            "https://registry.example/pack.whl",
                        ),
                    ),
                ),
            ),
        )

    def build_candidate(root, output, *, packs, **kwargs):
        assert packs == ("candidate",)
        assert kwargs["python"] == python
        assert kwargs["constraints"] == constraints
        assert kwargs["find_links"] == (runtime / "wheelhouse",)
        output.mkdir()
        (output / "candidate-7.2-py3-none-any.whl").write_bytes(archives["candidate"])
        head = original(["git", "rev-parse", "HEAD"], text=True).strip()
        (output / "official-packs.lock").write_text(
            Lockfile.of(
                [replace(entries["candidate"], source=f"git:Kosinkadink/dinkster-nodes@{head}")]
            ).record_json()
        )
        dependency_lock("candidate").write(output / "candidate-7.2-linux-cpu.pack-lock.json")

    def registry_defaults(registry, minimums, artifacts, *, engine_version):
        assert minimums == {"unchanged": "1.0"}
        entry = entries["unchanged"]
        pack_archive_path(artifacts, entry.artifact_digest).write_bytes(archives["unchanged"])
        return Lockfile.of([entry]), {}

    observed = []

    def selected_tests(root, catalog, artifacts, **kwargs):
        assert kwargs["generation_python"] == python
        assert kwargs["test_packs"] == ("candidate",)
        assert set(kwargs["locks"]) == {"candidate", "unchanged"}
        observed.extend((pack.pack, pack.version) for pack in catalog.packs)

    monkeypatch.setattr(admission, "build", build_candidate)
    monkeypatch.setattr(admission, "prepare_registry_sources", registry_defaults)
    monkeypatch.setattr(
        admission,
        "read_registry_pack_locks",
        lambda config, catalog, **kwargs: tuple(
            dependency_lock(pack.pack) for pack in catalog.packs
        ),
    )
    monkeypatch.setattr(admission, "run_locked_probe", lambda *args, **kwargs: StageResult())
    monkeypatch.setattr(admission, "run_selected_tests", selected_tests)
    candidate = tmp_path / "native-candidates"
    if candidate_mode == "native":
        build_candidate(
            None,
            candidate,
            packs=("candidate",),
            python=python,
            constraints=constraints,
            find_links=(runtime / "wheelhouse",),
        )
        context = {
            "engineVersion": "0.2",
            "engineRelease": identity,
            "engineCommit": "c" * 40,
            "packSourceCommit": original(["git", "rev-parse", "HEAD"], text=True).strip(),
            "baseId": "sha256:" + "f" * 64,
            "variant": "linux-cpu",
        }
        for field in context:
            (candidate / "engine.json").write_text(json.dumps({**context, field: "wrong"}))
            with pytest.raises(ValueError, match="candidate locks do not match"):
                admission.check_defaults(
                    admission.NamedRegistry("", "https://registry.invalid"),
                    engine_version="0.2",
                    engine_identity=identity,
                    runtime=runtime,
                    candidate_sources=candidate,
                )
        (candidate / "engine.json").write_text(json.dumps(context))
        monkeypatch.setattr(
            admission, "build", lambda *args, **kwargs: pytest.fail("rebuilt sources")
        )
        if problem is None:
            for name, replacement, message in (
                (
                    "official-packs.lock",
                    Lockfile.of([replace(entries["candidate"], source="git:wrong")])
                    .record_json()
                    .encode(),
                    "candidate sources do not match",
                ),
                (
                    "candidate-7.2-py3-none-any.whl",
                    b"changed source wheel",
                    "candidate source digest mismatch",
                ),
            ):
                path = candidate / name
                original_bytes = path.read_bytes()
                path.write_bytes(replacement)
                with pytest.raises(ValueError, match=message):
                    admission.check_defaults(
                        admission.NamedRegistry("", "https://registry.invalid"),
                        engine_version="0.2",
                        engine_identity=identity,
                        runtime=runtime,
                        candidate_sources=candidate,
                    )
                path.write_bytes(original_bytes)
            path = candidate / "candidate-7.2-linux-cpu.pack-lock.json"
            original_bytes = path.read_bytes()
            original_lock = dependency_lock("candidate")
            replace(
                original_lock,
                packages=(
                    replace(
                        original_lock.packages[0],
                        wheels=(replace(original_lock.packages[0].wheels[0], sha256="f" * 64),),
                    ),
                ),
            ).write(path)
            with pytest.raises(ValueError, match="locked outer wheel differs from the archive"):
                admission.check_defaults(
                    admission.NamedRegistry("", "https://registry.invalid"),
                    engine_version="0.2",
                    engine_identity=identity,
                    runtime=runtime,
                    candidate_sources=candidate,
                )
            path.write_bytes(original_bytes)
        replace(dependency_lock("candidate"), packages=()).write(
            candidate / "candidate-7.2-linux-cpu.pack-lock.json"
        )
    source_bytes = (
        {path.name: path.read_bytes() for path in candidate.iterdir()}
        if candidate_mode == "native"
        else {}
    )
    prepared = tmp_path / "prepared"
    result = admission.check_defaults(
        admission.NamedRegistry("", "https://registry.invalid"),
        engine_version="0.2",
        engine_identity=identity,
        runtime=runtime,
        candidate_base="b" * 40 if candidate_mode == "build" else None,
        candidate_sources=candidate if candidate_mode == "native" else None,
        prepare_output=prepared,
    )
    if candidate_mode == "native":
        assert {path.name: path.read_bytes() for path in candidate.iterdir()} == source_bytes
    groups = result["groups"]
    assert isinstance(groups, list)
    assert sorted(name for group in groups for name in group) == [
        "candidate",
        "unchanged",
    ]
    assert not observed
    metadata = json.loads((prepared / "engine.json").read_text())
    assert metadata["engineCommit"] == "c" * 40
    assert metadata["packSourceCommit"] != metadata["engineCommit"]
    bound = PackLock.read(prepared / "pack-locks/candidate.json")
    outer = next(item for item in bound.packages if item.name == "candidate")
    assert outer.version == "7.2"
    assert [(wheel.filename, wheel.size, wheel.sha256) for wheel in outer.wheels] == [
        (
            "candidate-7.2-py3-none-any.whl",
            len(archives["candidate"]),
            hashlib.sha256(archives["candidate"]).hexdigest(),
        )
    ]
    transferred = tmp_path / "transferred"
    shutil.copytree(prepared, transferred)
    fields = {
        "engine-version": "engineVersion",
        "engine-release": "engineRelease",
        "engine-commit": "engineCommit",
        "source-commit": "packSourceCommit",
    }
    if problem in fields:
        metadata[fields[problem]] = "wrong"
        (transferred / "engine.json").write_text(json.dumps(metadata))
    elif problem == "artifact":
        pack_archive_path(
            transferred / "artifacts", entries["candidate"].artifact_digest
        ).write_bytes(b"corrupt wheel")
    elif problem == "lock":
        lock = dependency_lock("unchanged")
        lock.write(transferred / "pack-locks/candidate.json")
    elif problem == "missing-outer":
        replace(bound, packages=()).write(transferred / "pack-locks/candidate.json")
    if problem is None:
        report = admission.check_defaults(
            admission.NamedRegistry("", "https://registry.invalid"),
            engine_version="0.2",
            engine_identity=identity,
            runtime=runtime,
            prepared_sources=transferred,
            test_packs=("candidate",),
        )
        assert report["packs"] == [
            {
                "pack": "candidate",
                "version": "7.2",
                "artifactDigest": entries["candidate"].artifact_digest,
            }
        ]
        assert sorted(observed) == [("candidate", "7.2"), ("unchanged", "1.3")]
    else:
        message = {
            "artifact": "locked outer wheel differs from the archive",
            "lock": "dependency lock identity differs",
            "missing-outer": "dependency lock omits the outer wheel",
            "security": "admission refused",
        }.get(problem, "prepared sources do not match the pack source and engine release")
        with pytest.raises(ValueError, match=message):
            admission.check_defaults(
                admission.NamedRegistry("", "https://registry.invalid"),
                engine_version="0.2",
                engine_identity=identity,
                runtime=runtime,
                prepared_sources=transferred,
                test_packs=("candidate",),
            )
        assert not observed
