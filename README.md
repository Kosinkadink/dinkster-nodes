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
uv run --no-sync pyright
uv run --no-sync pyright --pythonplatform Windows
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

Hosted pull-request and main checks use the same locked workspace. Root and
package tests each run in four disjoint file shards, with each package in its
own pytest process. Shader tests use the CPU software renderer.

There is no production publication workflow in this repository. Building or
testing a pack does not submit it to the public registry.

## Registry publisher credentials

First-party packs will use the same self-service credentials as third-party
packs. The `dinkster` publisher is already bound to its Dinkster account; the
first-party publication and credential cutover remain pending on
[the cutover issue](https://github.com/Kosinkadink/comfy-vibe-station/issues/698).
These instructions do not enable publication or complete that cutover.

Open [publisher accounts](https://id.dinkster.org/registry), sign in with the
Dinkster account or organization bound to the reserved `dinkster` publisher,
select that account or organization and choose **Refresh tokens**. The reserved
name is not available for a public claim; do not repeat its initial binding.

When the publication workflow is ready, choose an expiry at most 90 days away,
optionally restrict the token to one pack, then choose **Create token**. Save
the one-time value privately as the workflow's `DINKSTER_REGISTRY_TOKEN` Actions
secret. Never commit it or print it in logs or command arguments. Refreshing
the page cannot recover the value.

For cutover or rotation, install the replacement credential, confirm a
publication through the normal workflow, then revoke the previous credential
on the same page and remove its obsolete Actions secret. Record the first-party
publication and retirement on the cutover issue. Do not revoke the existing
credential before that confirmation or publish manually to demonstrate cutover.
Expired and revoked tokens cannot publish. Publisher registration does not
change admission, scanning or the official publisher tier.
