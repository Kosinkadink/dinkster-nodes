from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def published_generation(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = Path(__file__).resolve().parents[1]
    work = tmp_path_factory.mktemp("published-generation")
    generation = work / "generation"
    (generation / "bin").mkdir(parents=True)
    python = generation / "bin/python"
    shutil.copy2(Path(sys.executable).resolve(), python)
    base = Path(sys.base_prefix)
    lib = generation / "lib"
    lib.mkdir()
    for source in (base / "lib").iterdir():
        if not source.is_dir():
            (lib / source.name).symlink_to(source)
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    stdlib = lib / version
    stdlib.mkdir()
    for source in (base / "lib" / version).iterdir():
        if source.name != "site-packages":
            (stdlib / source.name).symlink_to(source)
    requirements = work / "pylock.toml"
    subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--only-group",
            "dev",
            "--no-emit-workspace",
            "--format",
            "pylock.toml",
            "--output-file",
            str(requirements),
        ],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "--break-system-packages",
            "--require-hashes",
            "--offline",
            "--only-binary",
            ":all:",
            "-r",
            str(requirements),
        ],
        check=True,
    )
    # Managed interpreters can expose engine dependencies through site initialization.
    # Inspect the interpreter's effective metadata, not only its local site directory.
    site = Path(
        subprocess.check_output(
            [str(python), "-I", "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"],
            text=True,
        ).strip()
    )
    inherited = generation / "engine-site"
    site.rename(inherited)
    site.mkdir()
    (site / "engine.pth").write_text(str(inherited) + "\n", encoding="ascii")
    return python


@pytest.mark.skipif(sys.platform != "linux", reason="selected archive tests run in Linux admission")
@pytest.mark.parametrize("selected_value,dependency_version", [(17, None), (29, "2"), (41, "1")])
def test_selected_candidate_uses_published_engine_not_checkout_or_default_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published_generation: Path,
    selected_value: int,
    dependency_version: str | None,
) -> None:
    from dinkster_registry import LockedPack, Lockfile, artifact_digest
    from dinkster_workers.pack_lock import (
        GenerationPackages,
        LockedDistribution,
        LockedWheel,
        PackLock,
    )
    from dinkster_workers.pack_wheel import build_pack_archive
    from dinkster_workers.provision import ProvisionError

    from scripts import run_selected_pack_tests as selected
    from tests.registry_fixtures import _wheel

    temporary = tmp_path / "physical-temporary"
    temporary.mkdir()
    alias = tmp_path / "temporary-parent"
    alias.symlink_to(temporary, target_is_directory=True)
    monkeypatch.setattr(selected.tempfile, "tempdir", str(alias))

    root = Path(__file__).resolve().parents[1]
    tools = selected.selected_test_tools
    shared = _wheel(tmp_path, "shared-test-dependency", dependency_version or "2", "shared", "")
    tool_shared = _wheel(tmp_path, "shared-test-dependency", "3", "shared", "")
    helper = _wheel(
        tmp_path,
        "selected-test-helper",
        "1",
        "test_helper",
        "",
        requires=("shared-test-dependency>=2",),
    )

    def test_tools(fixture, facts, *, variant):
        lock = tools(root, facts, variant=variant)
        if dependency_version is None:
            return lock
        return replace(
            lock,
            requirements=(*lock.requirements, "selected-test-helper"),
            packages=(
                *lock.packages,
                LockedDistribution("selected-test-helper", "1", (helper,)),
                LockedDistribution("shared-test-dependency", "3", (tool_shared,)),
            ),
        )

    monkeypatch.setattr(selected, "selected_test_tools", test_tools)
    checkout = tmp_path / "checkout"
    tests = checkout / "tests"
    tests.mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("")
    (checkout / "conftest.py").write_text(
        "import pytest\n@pytest.fixture\ndef selected_context():\n    return 'selected-fixture'\n"
    )
    poison = checkout / "packages/dinkster-nodes-text/src/dinkster_text"
    poison.mkdir(parents=True)
    (poison / "__init__.py").write_text("raise AssertionError('checkout pack imported')\n")
    (poison.parent.parent / "pyproject.toml").write_text('[project]\nname = "dinkster-text"\n')
    engine_poison = checkout / "src/dinkster"
    engine_poison.mkdir(parents=True)
    (engine_poison / "__init__.py").write_text("raise AssertionError('checkout engine imported')\n")
    (tests / "test_save_text.py").write_text(
        "import importlib.metadata\nfrom pathlib import Path\n"
        "import dinkster\nimport dinkster_text\n"
        "from dinkster.packs import first_party_pack_metadata\n"
        "def test_selected_bytes(selected_context):\n"
        "    assert selected_context == 'selected-fixture'\n"
        "    assert first_party_pack_metadata(Path(dinkster.__file__).resolve().parents[2])"
        "['dinkster-text'] == ('dinkster_text', 'dinkster-text')\n"
        "    assert sum(d.metadata['Name'] == 'dinkster-text' "
        "for d in importlib.metadata.distributions()) == 1\n"
        f"    assert dinkster_text.VALUE == {selected_value}\n"
        + (
            "    assert importlib.metadata.version('shared-test-dependency') == '2'\n"
            if dependency_version is not None
            else ""
        )
        + "    assert importlib.metadata.version('dinkster-text') == '1.2.3'\n"
        "    assert not Path(dinkster.__file__).is_relative_to(Path.cwd())\n"
        "    assert not (Path.cwd() / 'src/dinkster').exists()\n"
        "    assert not (Path.cwd() / '.git').exists()\n"
        "    assert Path(dinkster_text.__file__).is_relative_to(Path.cwd() / 'packages')\n",
        encoding="ascii",
    )
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    source = tmp_path / "source"
    (source / "dinkster_text").mkdir(parents=True)
    (source / "dinkster_text/__init__.py").write_text(f"NODES = []\nVALUE = {selected_value}\n")
    (source / "dinkster-pack.toml").write_text(
        '[pack]\nname = "dinkster-text"\npublisher = "dinkster"\nversion = "1.2.3"\n'
        'engine = ">=0.0.2.dev0,<2"\nrequires = ["packaging>=24"]\n'
        '[pack.entry]\nnodes = "dinkster_text:NODES"\n'
        '[pack.contracts]\nhost = "dinkster-pack-host/1"\napi = "dinkster-api/v1"\n'
    )
    archive = tmp_path / "dinkster_text-1.2.3-py3-none-any.whl"
    build_pack_archive(source, archive)
    payload = archive.read_bytes()
    digest = artifact_digest(payload)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    archive.rename(artifacts / f"{digest.removeprefix('blake3:')}.zip")
    catalog = Lockfile.of([LockedPack("dinkster-text", "1.2.3", digest, "dinkster", ())])
    lock = PackLock(
        "dinkster-text",
        "linux-cpu",
        ">=3.12",
        ("packaging>=24",),
        (),
        (
            LockedDistribution(
                "packaging",
                GenerationPackages.inspect(published_generation).versions["packaging"],
                (),
            ),
            LockedDistribution(
                "dinkster-text",
                "1.2.3",
                (
                    LockedWheel(
                        archive.name,
                        len(payload),
                        hashlib.sha256(payload).hexdigest(),
                        archive.as_uri(),
                    ),
                ),
            ),
        ),
    )
    if dependency_version is not None:
        lock = replace(
            lock,
            requirements=(*lock.requirements, "shared-test-dependency"),
            packages=(
                *lock.packages,
                LockedDistribution("shared-test-dependency", dependency_version, (shared,)),
            ),
        )
    expected = (
        pytest.raises(ProvisionError, match="shared-test-dependency")
        if dependency_version == "1"
        else nullcontext()
    )
    with expected:
        selected.run_selected_tests(
            checkout,
            catalog,
            artifacts,
            generation_python=published_generation,
            locks={"dinkster-text": lock},
            variant="linux-cpu",
        )
