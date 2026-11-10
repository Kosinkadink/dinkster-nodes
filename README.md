# dinkster-nodes

First-party Dinkster node packs and their shared helper library. Engine and
extension API dependencies come from published wheels, not an engine checkout.
The development workspace records their exact artifact URLs and hashes in
`pyproject.toml` and `uv.lock`. Pack manifests retain their own engine requirements
and versions.

## Development

Install uv, then run:

```sh
uv sync --locked --all-packages
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
CUDA_VISIBLE_DEVICES="" DINKSTER_ACCELERATOR=cpu uv run --no-sync pytest tests
```

Run each package's tests in a separate pytest process, for example:

```sh
CUDA_VISIBLE_DEVICES="" DINKSTER_ACCELERATOR=cpu uv run --no-sync pytest packages/dinkster-nodes-audio/tests
```

Package test directories reuse module names and contain package-local fixtures;
collecting every package in one pytest process creates import collisions.
The workspace selects CPU PyTorch wheels on Linux and Windows. Installed engine
dependencies are ordinary wheels; local editable installs are limited to packs
and the helper library in `packages/`.

There is no production publication workflow in this repository. Building or
testing a pack does not submit it to the public registry.
