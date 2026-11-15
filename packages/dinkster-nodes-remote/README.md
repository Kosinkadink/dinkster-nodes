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
restarting the pack. Without a token, schemas remain visible and invocation
fails with a sign-in error.
