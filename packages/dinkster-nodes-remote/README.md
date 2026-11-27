# dinkster-nodes-remote

Default-installed client pack for catalog-driven remote nodes. It loads the
current `NodeSchema` catalog, exposes valid `dinkster.remote.*` nodes, uploads
asset inputs, submits authenticated jobs, forwards progress and previews, and
verifies downloaded output digests before adding them to Dinkster's asset vault.

Public catalogs are cached under the pack scratch directory and bound to their
catalog endpoint. Cache files are separated by expected environment and endpoints,
so switching between production and QA preserves each public catalog independently.
Authenticated catalogs are kept only in memory, never persisted,
and discarded when the bearer changes. Startup continues with the matching public
cache, or an empty remote pack, if discovery is unavailable. Each catalog request
selects the client's supported schema wire and reads the current bearer file.
Signed-in discovery uses the gateway's verified project context without decoding
JWT authority in the client. Policy-filtered schema changes trigger an atomic
worker reload even when the public catalog epoch is unchanged. Malformed entries
are skipped without hiding valid entries.

The engine supplies resolved addresses from its service table and the expected
`DINKSTER_REMOTE_ENVIRONMENT`. Before catalog discovery, the pack checks the
unauthenticated gateway environment document. A mismatch or unavailable document
disables remote nodes, including cached nodes; a subsequent valid document permits
recovery. QA invocation errors explicitly name QA services.

`dinkster-serve` accepts:

```text
--remote-catalog-base URL
--remote-gateway-base URL
--remote-auth-token-file PATH
--remote-catalog-poll-interval SECONDS
```

The equivalent environment variables are `DINKSTER_REMOTE_CATALOG_BASE`,
`DINKSTER_REMOTE_GATEWAY_BASE`, `DINKSTER_REMOTE_AUTH_TOKEN_FILE`, and
`DINKSTER_REMOTE_CATALOG_POLL_INTERVAL`. The gateway base defaults to the catalog
base. No network request is made when the catalog base is unset. Job polling
defaults to one second for image nodes and five seconds for
video nodes; `DINKSTER_REMOTE_IMAGE_POLL_INTERVAL` and
`DINKSTER_REMOTE_VIDEO_POLL_INTERVAL` override those intervals. The bearer token
file is read for each invocation so session rotation does not require
restarting the pack. With the host-supplied `DINKSTER_REMOTE_RUN_TOKEN_DIRECTORY`,
catalog discovery is anonymous and execution reads only the token named by the
SHA-256 of the host invocation's job reference. Project-run tokens are reread
before uploads, submissions, retries, polling and cancellation. Missing run
context or credentials never falls back to an install-wide token. Without a
token, schemas remain visible and invocation fails with a sign-in error.

`DINKSTER_REMOTE_PARTNER_MODE` optionally selects `fake` or `real` through the
job submission's `X-Dinkster-Partner-Mode` header. When unset, no override is sent:
QA gateway defaults to fake and production defaults to real. Production refuses
explicit fake jobs. `DINKSTER_REMOTE_FAKE_OUTCOME` optionally selects `success`,
`fail`, `timeout`, or `slow` through `X-Dinkster-Fake-Outcome`; it requires effective
fake mode (explicit fake, or an unset mode under QA). Invalid values fail pack
configuration. Both controls are read at process startup, not from node inputs,
and are sent only on job submissions, including retries. They do not change the
catalog, project or saved workflow and never grant real-partner permission.

## Upstream compatibility

- `GeminiDecisionChoice`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_gemini.py).
- `GeminiDecisionPickImage`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_gemini.py).
- `GeminiDecisionScore`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_gemini.py).
- `GeminiDecisionYesNo`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_gemini.py).
- `GeminiImage2Node`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_gemini.py).
- `GeminiImageNode`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_gemini.py).
- `GeminiInputFiles`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_gemini.py).
- `GeminiNanoBanana2`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_gemini.py).
- `GeminiNanoBanana2V2`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_gemini.py).
- `GeminiNode`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_gemini.py).
- `GeminiNodeV2`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_gemini.py).
- `GeminiNodeV3`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_gemini.py).
- `GeminiVideoOmni`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_gemini.py).
- `GeminiVideoOmniV2`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_gemini.py).
- `HeyGenAvatar6IdentityToVideoNode`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_heygen.py).
- `HeyGenAvatar6ReferenceToVideoNode`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_heygen.py).
- `HeyGenAvatarVideoNode`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_heygen.py).
- `HeyGenCreateAvatarNode`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_heygen.py).
- `HeyGenDecisionChoice`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_heygen.py).
- `HeyGenDecisionScore`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_heygen.py).
- `HeyGenDecisionYesNo`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_heygen.py).
- `HeyGenImageToVideoNode`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_heygen.py).
- `HeyGenReferenceToVideoNode`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_heygen.py).
- `HeyGenTalkingPhotoNode`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_heygen.py).
- `HeyGenTextToSpeechNode`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_heygen.py).
- `HeyGenVideoTranslateNode`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_heygen.py).
- `HeyGenVoiceCloneNode`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_heygen.py).
- `HeyGenVoiceCloneTextToSpeechNode`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_heygen.py).
- `HeyGenVoiceTextToSpeechNode`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_heygen.py).
- `TypeSafeJevChoice`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_typesafe.py).
- `TypeSafeJevPickImage`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_typesafe.py).
- `TypeSafeJevScore`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_typesafe.py).
- `TypeSafeJevYesNo`: Partner API node needs the remote gateway catalog and credential contract. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4c2b826d74a1827f5cc4a706608fc4653ed12972/comfy_api_nodes/nodes_typesafe.py).
