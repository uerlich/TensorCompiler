
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

def _contiguous_strides_bytes(shape, elem_B):
    strides = [0]*len(shape)
    stride = elem_B
    for i in range(len(shape)-1, -1, -1):
        strides[i] = stride
        stride *= int(shape[i])
    return strides

def emit_laneization_xlsx_from_json(json_path: str|Path, out_xlsx: str|Path, default_elem_bits: int = 8) -> str:
    jp = Path(json_path)
    cfg = json.loads(jp.read_text())
    axes = cfg.get("tensor", {}).get("axes", [])
    shape = cfg.get("tensor", {}).get("shape", [])
    if not axes or not shape or len(axes) != len(shape):
        raise ValueError("JSON must include tensor.axes and tensor.shape of the same length")
    elem_bits = int(cfg.get("tensor", {}).get("bits", default_elem_bits))
    elem_B = max(1, elem_bits // 8)
    strides = _contiguous_strides_bytes(shape, elem_B)
    df = pd.DataFrame({"dim": axes, "size": [int(s) for s in shape], "stride_B": strides})
    out_xlsx = Path(out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx) as w:
        df.to_excel(w, index=False, sheet_name="Laneization")
    return str(out_xlsx)
