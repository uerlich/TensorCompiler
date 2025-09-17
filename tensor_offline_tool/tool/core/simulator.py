from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import math

from .constants import LANES, BANKS, ATOM_B, COMMIT_B, BUF_B, BUS_B
from .io_utils import MappingSpec, repack_elements_to_atoms

@dataclass
class SimConfig:
    beta_eff_B: int
    lifts_B: List[int]
    zero_mode: str
    pixel_B: int

@dataclass
class SimOutputs:
    trace_path: str
    frames: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    tensor_info: Dict[str, Any]


def banks_for_commit(base_atom_mod32: int, delta_b: int, lifts_atoms: List[int], lanes_active: int) -> List[int]:
    """Return compiled bank indices for a commit using β_eff (via Δb) and per-lane lifts.
    All args expressed in atom units (16 B); returns a list of length lanes_active.
    """
    banks = []
    for lane in range(lanes_active):
        lift = lifts_atoms[lane] if lane < len(lifts_atoms) else 0
        banks.append((base_atom_mod32 + delta_b*lane + lift) % BANKS)
    return banks

def bank_index(addr_B: int) -> int:
    return (addr_B // ATOM_B) % BANKS

def simulate(mapping_spec: MappingSpec, lane_choice, cfg: SimConfig, out_prefix: str,
             final_mapping: np.ndarray, final_shape: tuple, input_axes: Optional[List[str]] = None) -> Optional[SimOutputs]:
    total_elems = int(np.prod(mapping_spec.input_shape)) if mapping_spec.input_shape else 0
    elem_B = cfg.pixel_B if cfg.pixel_B > 0 else 1

    num_atoms, elems_per_atom_counts = repack_elements_to_atoms(total_elems, elem_B, cfg.zero_mode)
    atoms_per_commit = COMMIT_B // ATOM_B
    buf_atoms = BUF_B // ATOM_B
    total_atoms = int(num_atoms)
    if total_atoms <= 0: return None

    commit_ranges: List[Tuple[int, int]] = []
    cur = 0
    while cur < total_atoms:
        buf_boundary = ((cur // buf_atoms) + 1) * buf_atoms
        end = cur + atoms_per_commit
        if end > buf_boundary:
            cur = buf_boundary
            continue
        commit_ranges.append((cur, min(end, total_atoms)))
        cur = end

    frames: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    
    output_total_elems = int(np.prod(final_shape)) if final_shape else 0
    out_mem_elements = np.full(output_total_elems, -1, dtype=np.int64)
    bufA_atoms = [-1] * buf_atoms
    bufB_atoms = [-1] * buf_atoms
    delta_b = cfg.beta_eff_B // ATOM_B
    lifts_atoms = [lb // ATOM_B for lb in cfg.lifts_B]
    elems_per_atom = ATOM_B // elem_B if elem_B > 0 else 1

    frames.append({
        'cycle': 0, 'banks': [], 'delta_b': delta_b, 'lifts_atoms': lifts_atoms,
        'input_mem_read_mask': np.zeros(total_elems, dtype=bool),
        'bufA_atoms': bufA_atoms.copy(), 'bufB_atoms': bufB_atoms.copy(),
        'output_mem_content': out_mem_elements.copy(), 'committed': False,
        'l1_bus_utilization': 0.0, 'l2_bus_utilization': 0.0,
        'roleA': 'IDLE', 'roleB': 'IDLE'
    })

    for commit_idx, (start_atom, end_atom) in enumerate(commit_ranges):
        cycle = commit_idx + 1
        fill_buf = 'A' if commit_idx % 2 == 0 else 'B'
        drain_buf = 'B' if commit_idx % 2 == 0 else 'A'

        l1_util = 0.0
        if commit_idx > 0:
            prev_s, prev_e = commit_ranges[commit_idx - 1]
            l1_util = (prev_e - prev_s) / atoms_per_commit
            
            start_elem = prev_s * elems_per_atom
            end_elem = prev_e * elems_per_atom
            for out_elem_idx in range(start_elem, end_elem):
                if out_elem_idx < output_total_elems:
                    out_coords = np.unravel_index(out_elem_idx, final_shape)
                    if len(out_coords) == len(final_mapping.shape) and all(c < s for c, s in zip(out_coords, final_mapping.shape)):
                        source_flat_id = final_mapping[out_coords]
                        if source_flat_id >= 0:
                           out_mem_elements[out_elem_idx] = source_flat_id
            
            if drain_buf == 'A':
                bufA_atoms = [-1] * buf_atoms
            else:
                bufB_atoms = [-1] * buf_atoms
        
        num_atoms_in_commit = end_atom - start_atom
        l2_util = num_atoms_in_commit / atoms_per_commit
        
        active_fill_buf_list = bufA_atoms if fill_buf == 'A' else bufB_atoms
        for k in range(num_atoms_in_commit):
            if k < len(active_fill_buf_list):
                active_fill_buf_list[k] = (k % LANES)

                # Compute banks from compiled lane law: base + Δb·lane + lift_lane (all in atoms)
        lanes_active = min(LANES, num_atoms_in_commit)

        base_atom_mod32 = start_atom % BANKS
        delta_b_atoms = delta_b  # already in atoms
        # Fast-path: if gcd(Δb,32) ≤ 2 and all lifts are zero, natural residues are distinct
        use_fast = (delta_b_atoms != 0 and math.gcd(delta_b_atoms, BANKS) <= 2 and all(x == 0 for x in lifts_atoms))
        if cfg.beta_eff_B == 0:
            # single-lane fallback region: only one lane writes per commit
            lanes_active = min(1, lanes_active)
            banks = banks_for_commit(base_atom_mod32, 0, lifts_atoms, lanes_active)
        else:
            if use_fast:
                banks = [ (base_atom_mod32 + delta_b_atoms*lane) % BANKS for lane in range(lanes_active) ]
            else:
                banks = banks_for_commit(base_atom_mod32, delta_b_atoms, lifts_atoms, lanes_active)

        # Assert pairwise distinctness among active lanes
        if len(set(banks)) != len(banks):
            raise AssertionError(f"[bank-safety] collision in commit {commit_idx}: base={base_atom_mod32}, Δb={delta_b_atoms}, lifts={lifts_atoms[:lanes_active]}, banks={banks}")
        trace_rows.append({
            'commit_idx': commit_idx, 'buf': fill_buf, 'addr_atom_start': start_atom, 'addr_atom_end': end_atom - 1,
            'banks': '|'.join(map(str, banks)), 'delta_b': delta_b
        })
        
        read_mask_elements = frames[-1]['input_mem_read_mask'].copy()
        start_elem_read = start_atom * elems_per_atom
        end_elem_read = end_atom * elems_per_atom
        if total_elems > 0:
            read_mask_elements[start_elem_read:end_elem_read] = True

        frames.append({
            'cycle': cycle, 'banks': banks, 'delta_b': delta_b, 'lifts_atoms': lifts_atoms,
            'input_mem_read_mask': read_mask_elements,
            'bufA_atoms': bufA_atoms.copy(), 'bufB_atoms': bufB_atoms.copy(),
            'output_mem_content': out_mem_elements.copy(), 'committed': True,
            'l1_bus_utilization': l1_util, 'l2_bus_utilization': l2_util,
            'roleA': ('FILL' if fill_buf == 'A' else 'DRAIN'),
            'roleB': ('FILL' if fill_buf == 'B' else 'DRAIN')
        })

    if commit_ranges:
        final_cycle = len(commit_ranges) + 1
        final_drain_buf = 'A' if (len(commit_ranges) - 1) % 2 == 0 else 'B'
        
        last_s, last_e = commit_ranges[-1]
        l1_util_final = (last_e - last_s) / atoms_per_commit

        start_elem = last_s * elems_per_atom
        end_elem = last_e * elems_per_atom
        for out_elem_idx in range(start_elem, end_elem):
            if out_elem_idx < output_total_elems:
                out_coords = np.unravel_index(out_elem_idx, final_shape)
                if len(out_coords) == len(final_mapping.shape) and all(c < s for c, s in zip(out_coords, final_mapping.shape)):
                    source_flat_id = final_mapping[out_coords]
                    if source_flat_id >= 0:
                        out_mem_elements[out_elem_idx] = source_flat_id
        
        if final_drain_buf == 'A': bufA_atoms = [-1] * buf_atoms
        else: bufB_atoms = [-1] * buf_atoms

        frames.append({
            'cycle': final_cycle, 'banks': [], 'delta_b': delta_b, 'lifts_atoms': lifts_atoms,
            'input_mem_read_mask': frames[-1]['input_mem_read_mask'].copy(),
            'bufA_atoms': bufA_atoms.copy(), 'bufB_atoms': bufB_atoms.copy(),
            'output_mem_content': out_mem_elements.copy(), 'committed': True,
            'l1_bus_utilization': l1_util_final, 'l2_bus_utilization': 0.0,
            'roleA': 'IDLE', 'roleB': 'IDLE'
        })

    trace_csv = f"{out_prefix}_trace.csv"
    pd.DataFrame(trace_rows).to_csv(trace_csv, index=False)

    metrics = {'commits': len(commit_ranges), 'atoms': total_atoms}
    tensor_info = {
        'input_shape': mapping_spec.input_shape, 'output_shape': final_shape, 'pixel_B': elem_B,
        'num_atoms': num_atoms, 'elems_per_atom': elems_per_atom_counts, 'input_axes': input_axes or [],
        'total_input_elements': total_elems, 'total_output_elements': output_total_elems
    }
    return SimOutputs(trace_path=trace_csv, frames=frames, metrics=metrics, tensor_info=tensor_info)