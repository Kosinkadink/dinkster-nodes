
## Upstream compatibility

- `TextGenerate`: Update the existing native replacement; do not register a duplicate node. Update the existing native schema and execute method in packages/dinkster-nodes-llm/src/dinkster_llm/text_generation.py; preserve its node ID and replacement mapping. [Upstream source](https://github.com/Comfy-Org/ComfyUI/blob/ef507153a6b96c22e0ba98ceec824682c61d70fb/comfy_extras/nodes_textgen.py).
