
## Upstream compatibility

- `OpticalFlowLoader`: Update the existing native replacement; do not register a duplicate node. Update the existing native schema and execute method in packages/dinkster-nodes-video/src/dinkster_video/optical_flow.py; preserve its node ID and replacement mapping. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/c844cb672186f187bf9455e2808821417a10d296/comfy_extras/nodes_void.py).
- `VideoDeflicker`: V3 schema has execution or presentation metadata requiring review. No identity replacement exists; add the node to this pack. Review related schemas before folding it into a consolidated node. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/c844cb672186f187bf9455e2808821417a10d296/comfy_extras/nodes_video_deflicker.py).
