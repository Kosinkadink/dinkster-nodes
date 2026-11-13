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

Building or testing a pack does not submit it to the public registry.
Pull requests, merge groups and forks cannot publish.

A main push affecting pack sources or publication inputs publishes
their declared versions only after all checks, native dependency locks, selected
wheel admission and isolated publisher/scanner checks pass. The publisher uses
the same source wheels and native locks those jobs checked. It compares the
selected set with existing public versions without credentials before submitting
anything: changed bytes or install locks require a version bump. Identical public
versions are not resubmitted. Submissions are serialized without cancellation,
and the job waits for public activation rather than treating a queue receipt as
acceptance. The credential is available only to the submission step.

This repository is the only publisher of these packs. The engine does not
publish them; the two repositories must never publish the same pack name
concurrently.

## Registry publisher credentials

First-party packs use the same self-service credentials as third-party packs.
The `dinkster` publisher is already bound to its Dinkster account. Credential
cutover requires a confirmed publication through the normal workflow before
the previous credential is revoked.

Open [publisher accounts](https://id.dinkster.org/registry), sign in with the
Dinkster account or organization bound to the reserved `dinkster` publisher,
select that account or organization and choose **Refresh tokens**. The reserved
name is not available for a public claim; do not repeat its initial binding.

To rotate its credential, choose an expiry at most 90 days away,
optionally restrict the token to one pack, then choose **Create token**. Save
the one-time value privately as the workflow's `DINKSTER_REGISTRY_TOKEN` Actions
secret. Never commit it or print it in logs or command arguments. Refreshing
the page cannot recover the value.

For cutover or rotation, install the replacement credential, confirm a
publication through the normal workflow, then revoke the previous credential
on the same page and remove its obsolete Actions secret. Maintainers record the
confirmed first-party publication, token revocation and Actions-secret removal
on issue 698 of the private `Kosinkadink/comfy-vibe-station` tracker before
declaring cutover complete. Do not revoke the existing credential before that
confirmation or publish manually to demonstrate cutover.
Expired and revoked tokens cannot publish. Publisher registration does not
change admission, scanning or the official publisher tier.
