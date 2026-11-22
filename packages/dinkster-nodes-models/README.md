
## Upstream compatibility

- `BlockSparseAttention`: Update the existing native replacement; do not register a duplicate node. Update the existing native schema and execute method in packages/dinkster-nodes-models/src/dinkster_models/sparse_attention.py; preserve its node ID and replacement mapping. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/4bff04177c25cf1f1448ee0e1209f175059a9b35/comfy_extras/nodes_sparse_attention.py).
