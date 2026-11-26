"""First-party ComfyUI-origin utilities nodes."""

from dinkster_api.v1 import (
    COLOR_TYPE,
    TypeRegistry,
    inference_nodes,
    register_box_types,
)

from .color_resolution import (
    ColorToRGBInt,
    ResolutionSelector,
)
from .curve_editor import (
    CurveEditor,
    SampleCurve,
)
from .flow import HasValue, OnlyIf, OrElse
from .image_crop import (
    PrimitiveBoundingBox,
)
from .list_utilities import (
    CreateList,
    GetItemFromList,
)
from .logic import (
    ComfyAndNode,
    ComfyNotNode,
    ComfyOrNode,
    CustomCombo,
)
from .loop_lists import ListElement, MakeList
from .loops import (
    EndLoop,
    LoopIteration,
    LoopProgress,
    LoopResult,
    StartLoop,
)
from .math import (
    ComfyMathExpression,
)
from .primitive import (
    Primitive,
    PrimitiveBoolean,
    PrimitiveChoice,
    PrimitiveFloat,
    PrimitiveInt,
    PrimitiveString,
)
from .switch import (
    Switch,
)
from .utilities import (
    ComfyNumberConvert,
    PreviewAny,
    SeedNode,
)

NODES = (
    ColorToRGBInt,
    ResolutionSelector,
    CurveEditor,
    SampleCurve,
    PrimitiveBoundingBox,
    CreateList,
    GetItemFromList,
    ComfyNotNode,
    ComfyAndNode,
    ComfyOrNode,
    CustomCombo,
    StartLoop,
    LoopIteration,
    LoopProgress,
    LoopResult,
    EndLoop,
    ComfyMathExpression,
    PrimitiveString,
    PrimitiveInt,
    PrimitiveFloat,
    PrimitiveBoolean,
    PrimitiveChoice,
    Primitive,
    Switch,
    ComfyNumberConvert,
    SeedNode,
    PreviewAny,
    MakeList,
    ListElement,
    OrElse,
    HasValue,
    OnlyIf,
)


def register_types(registry: TypeRegistry) -> None:
    register_box_types(registry)
    if COLOR_TYPE not in registry:
        registry.register(COLOR_TYPE)


__all__ = ["NODES", "register_types"]

inference_nodes(NODES)
