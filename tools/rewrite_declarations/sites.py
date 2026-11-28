"""Where in the pack source each declaration object was built.

``record()`` wraps the constructors of the old declaration classes so each
object remembers the first frame outside the engine that built it: the line
the rewrite edits. Call it before any pack module is imported.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CLASSES = (
    "InputSpec",
    "OutputSpec",
    "InputFamilySpec",
    "DynamicComboSpec",
    "DynamicComboOption",
    "NodeSchema",
)
_SITES: dict[int, tuple[str, int]] = {}
_KEEP: list[Any] = []


def record() -> None:
    from dinkster_schema import model

    if "dinkster_nodes_3d" in sys.modules or any(m.startswith("dinkster_3d") for m in sys.modules):
        raise RuntimeError("record() must run before pack modules are imported")
    for name in CLASSES:
        cls = getattr(model, name)
        original = cls.__init__

        def init(self: Any, *args: Any, __original: Any = original, **kwargs: Any) -> None:
            __original(self, *args, **kwargs)
            frame = sys._getframe(1)
            while frame is not None:
                path = Path(frame.f_code.co_filename)
                if path.is_relative_to(ROOT) and "tools" not in path.relative_to(ROOT).parts[:1]:
                    _SITES[id(self)] = (str(path.relative_to(ROOT)), frame.f_lineno)
                    _KEEP.append(self)
                    return
                frame = frame.f_back

        cls.__init__ = init


def site(obj: Any) -> str:
    found = _SITES.get(id(obj))
    return f"{found[0]}:{found[1]}" if found else ""
