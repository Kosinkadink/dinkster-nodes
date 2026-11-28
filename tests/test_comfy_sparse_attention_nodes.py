"""Cold wire contract for native sparse-attention workflow inputs."""

from typing import Any, cast

from dinkster_api.v1 import schema_from_wire, schema_to_wire
from dinkster_models import NODES
from dinkster_models.sparse_attention import BlockSparseAttention

from tests.body_rows import wire_entry


def test_sparse_attention_wire_keeps_method_before_schedule_and_single_line_blocks() -> None:
    assert BlockSparseAttention in NODES
    assert BlockSparseAttention.schema().node_type == "dinkster.models.block_sparse_attention"
    wire = schema_to_wire(BlockSparseAttention.schema())
    body = cast(list[dict[str, Any]], wire["body"])
    inputs = [item for item in body if item["kind"] not in ("output", "section")]
    (advanced,) = (item for item in body if item["kind"] == "section")
    assert [item["id"] for item in inputs] == [
        "model",
        "selection",
        "start_percent",
        "end_percent",
    ]
    assert {key: value for key, value in advanced.items() if key != "rows"} == {
        "kind": "section",
        "id": "advanced",
        "label": "Advanced",
        "collapsed": True,
    }
    assert [item["id"] for item in advanced["rows"]] == [
        "dense_blocks",
        "min_tokens",
        "extra_tokens",
        "sink_conditioning",
        "verbose",
    ]
    assert inputs[1]["kind"] == "choice"
    assert wire_entry(wire, "dense_blocks")["widget"] == {"type": "STRING", "multiline": False}
    assert schema_to_wire(schema_from_wire(wire)) == wire
