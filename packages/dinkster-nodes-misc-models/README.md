
## Upstream compatibility

- `SenseNovaConditioning`: V3 schema has execution or presentation metadata requiring review. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/5df2ce62f1408572c18a12a3aefee326cd2305a5/comfy_extras/nodes_sensenova.py).
- `SenseNovaInterleaveDecode`: V3 schema has execution or presentation metadata requiring review. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/5df2ce62f1408572c18a12a3aefee326cd2305a5/comfy_extras/nodes_sensenova.py).
- `SenseNovaSamplingOptions`: Update the existing native replacement; do not register a duplicate node. Update the existing native schema and execute method in packages/dinkster-nodes-misc-models/src/dinkster_misc_models/sensenova_options.py; preserve its node ID and replacement mapping. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/5df2ce62f1408572c18a12a3aefee326cd2305a5/comfy_extras/nodes_sensenova.py).
