"""First-party text manipulation and mounted text output."""

from dinkster_api.v1 import MOUNT_TYPE, TypeRegistry, inference_nodes, register_mount_type

from .save_text import SaveText
from .text import TEXT_NODES

NODES = (*TEXT_NODES, SaveText)


def register_types(registry: TypeRegistry) -> None:
    if MOUNT_TYPE not in registry:
        register_mount_type(registry)


__all__ = ["NODES", "register_types"]

inference_nodes(NODES)
