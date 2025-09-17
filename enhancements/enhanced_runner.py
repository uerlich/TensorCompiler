from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import json
import numpy as np

from tensor_offline_tool.tool.main import compile_and_run as original_compile_and_run
from tensorviz_dashboard_unified_v21 import evaluate_pipeline, load_config_bom_safe
from .laneization import emit_laneization_xlsx_from_json
from .beat_like_laneization import render_beats_like_laneization

def enhanced_compile_and_run(json_path: str|Path,
                             out_dir: str|Path,
                             pixel_bits: int = 8,
                             zero_mode: str = "idle_lanes",
                             viz: str = "gif",
                             gap_budget_B: Optional[int] = None):
    json_path = Path(json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config_bom_safe(json_path)
    states = evaluate_pipeline(cfg)
    final_state = states[-1]
    final_mapping = np.array(final_state['mapping'])
    final_shape = final_state['shape']
    input_axes = cfg.get("tensor", {}).get("axes", [])


    workbook_path = out_dir / f"{json_path.stem}_laneization.xlsx"
    emit_laneization_xlsx_from_json(json_path, workbook_path, default_elem_bits=pixel_bits)

    original_compile_and_run(
        str(workbook_path),
        pixel_bits,
        zero_mode,
        viz,
        gap_budget_B,
        final_mapping=final_mapping,
        final_shape=final_shape,
        input_axes=input_axes
    )

    beats_gif_path = out_dir / f"{json_path.stem}_beats.gif"
    render_beats_like_laneization(
        workbook_path,
        pixel_bits,
        zero_mode,
        viz,
        gap_budget_B,
        beats_gif_path,
        final_mapping=final_mapping,
        final_shape=final_shape,
        input_axes=input_axes
    )
    
    return {"workbook": str(workbook_path), "beats_gif": str(beats_gif_path)}