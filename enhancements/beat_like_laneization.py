from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from tensor_offline_tool.tool.core.constants import ATOM_B, COMMIT_B, BUS_B, LANES
from tensor_offline_tool.tool.core.io_utils import read_workbook, MappingSpec
from tensor_offline_tool.tool.core.bank_safety import choose_laneization
from tensor_offline_tool.tool.core.simulator import SimConfig, simulate
from tensor_offline_tool.tool.vis.visualize import render_visualization

def _deepcopy_frame(fr: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a deep copy of a frame dictionary, handling numpy arrays."""
    out = {}
    for k, v in fr.items():
        if isinstance(v, np.ndarray): out[k] = v.copy()
        elif isinstance(v, list): out[k] = list(v)
        elif isinstance(v, dict): out[k] = v.copy()
        else: out[k] = v
    return out

def expand_commits_to_beats(frames: List[Dict[str,Any]], elem_B: int = 1, beat_B:int=BUS_B):
    """
    Expands commit-level frames into a granular beat-level timeline, correctly modeling
    the temporal separation of drains and the incremental nature of buffer fills.
    """
    if not frames: return []

    current_beat_state = _deepcopy_frame(frames[0])
    beats_out: List[Dict[str,Any]] = [current_beat_state]
    running_beat_cycle = 0

    for i in range(1, len(frames)):
        commit_frame = frames[i]
        
        # Handle the final non-committed frame from the simulator
        if not commit_frame.get('committed', False) and commit_frame.get('cycle', 0) > running_beat_cycle:
             final_frame = _deepcopy_frame(commit_frame)
             final_frame['cycle'] = running_beat_cycle + 1
             beats_out.append(final_frame)
             continue
        
        irm_pre = current_beat_state.get("input_mem_read_mask", np.array([]))
        irm_post = commit_frame.get("input_mem_read_mask", np.array([]))
        
        changed_elements = np.where((~irm_pre.astype(bool)) & irm_post.astype(bool))[0]
        elements_per_beat = max(1, beat_B // elem_B)
        total_beats = (len(changed_elements) + elements_per_beat - 1) // elements_per_beat
        atoms_per_beat = max(1, beat_B // ATOM_B)

        # Handle drain-only cycle (no new elements read)
        if total_beats == 0:
            running_beat_cycle += 1
            beat_frame = _deepcopy_frame(current_beat_state)
            beat_frame.update({
                'cycle': running_beat_cycle,
                'output_mem_content': commit_frame['output_mem_content'],
                'bufA_atoms': commit_frame['bufA_atoms'], 'bufB_atoms': commit_frame['bufB_atoms'],
                'committed': True, 'banks': [],
                'l1_bus_utilization': commit_frame.get('l1_bus_utilization', 0.0),
                'l2_bus_utilization': 0.0
            })
            beats_out.append(beat_frame)
            current_beat_state = _deepcopy_frame(beat_frame)
            continue

        # Generate frames for each beat in the fill sequence
        for b in range(total_beats):
            running_beat_cycle += 1
            beat_frame = _deepcopy_frame(current_beat_state)
            beat_frame['cycle'] = running_beat_cycle

            if b == 0: # First beat also handles the atomic drain event
                beat_frame.update({
                    'output_mem_content': commit_frame['output_mem_content'],
                    'committed': True, 'banks': commit_frame['banks'],
                    'l1_bus_utilization': commit_frame['l1_bus_utilization'],
                    'l2_bus_utilization': 1.0,
                    'roleA': commit_frame['roleA'], 'roleB': commit_frame['roleB']
                })
                drained_buffer = 'A' if commit_frame.get('roleA') == 'DRAIN' else 'B'
                if drained_buffer == 'A': beat_frame['bufA_atoms'] = [-1] * len(beat_frame['bufA_atoms'])
                else: beat_frame['bufB_atoms'] = [-1] * len(beat_frame['bufB_atoms'])
            else: # Subsequent beats are purely for filling
                beat_frame.update({'committed': False, 'banks': [], 'l1_bus_utilization': 0.0, 'l2_bus_utilization': 1.0})

            # Incrementally update L2 read mask
            elements_to_turn_on = changed_elements[:elements_per_beat * (b + 1)]
            beat_frame["input_mem_read_mask"][elements_to_turn_on] = True
            
            # Incrementally fill the correct buffer
            start_atom_idx = b * atoms_per_beat
            end_atom_idx = (b + 1) * atoms_per_beat
            filling_buffer = 'A' if commit_frame.get('roleA') == 'FILL' else 'B'
            
            # *** FIX: Corrected f-string to include underscore in key name ***
            final_buf_state = commit_frame[f'buf{filling_buffer}_atoms']
            target_buf = beat_frame[f'buf{filling_buffer}_atoms']
            
            for k in range(start_atom_idx, min(end_atom_idx, len(target_buf))):
                if k < len(final_buf_state):
                    target_buf[k] = final_buf_state[k]

            beats_out.append(beat_frame)
            current_beat_state = _deepcopy_frame(beat_frame)
            
    return beats_out

def render_beats_like_laneization(workbook_path: str|Path, pixel_bits: int, zero_mode: str, viz: str, gap_budget_B: int|None, out_gif: str|Path, final_mapping: np.ndarray, final_shape: tuple, input_axes: Optional[List[str]] = None, title: str = "Laneization (beat-level)") -> str:
    mapping_spec: MappingSpec = read_workbook(str(workbook_path), pixel_bits=pixel_bits)
    candidates = mapping_spec.dims
    gap_budget_B_int = None if gap_budget_B is None else int(gap_budget_B)
    choice = choose_laneization(candidates, gap_budget_B=gap_budget_B_int)
    
    elem_B = max(1, pixel_bits // 8)
    cfg = SimConfig(beta_eff_B=choice.beta_eff_B, lifts_B=[l*ATOM_B for l in choice.lifts], zero_mode=zero_mode, pixel_B=elem_B)

    out_prefix = str(Path(out_gif).with_suffix(""))
    sim_out = simulate(mapping_spec, choice, cfg, out_prefix, final_mapping=final_mapping, final_shape=final_shape, input_axes=input_axes)
    if sim_out and sim_out.frames:
        beat_frames = expand_commits_to_beats(sim_out.frames, elem_B=elem_B, beat_B=BUS_B)
        render_visualization(str(out_gif), beat_frames, title=title, tensor_info=sim_out.tensor_info)
    return str(out_gif)