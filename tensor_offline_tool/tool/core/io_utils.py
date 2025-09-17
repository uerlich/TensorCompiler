from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from .constants import ATOM_B, LANES, BANKS

@dataclass
class MappingSpec:
    dims: List[Tuple[str,int,int]]               # (name, size, stride_bytes)
    pixel_bytes: int
    input_shape: Tuple[int, ...] = field(default_factory=lambda: (1, 256, 256))
    # output_shape is determined by the pipeline, not known at workbook read time.
    absidx: Optional[pd.DataFrame] = None
    vecidx: Optional[pd.DataFrame] = None

def read_workbook(path: str, pixel_bits: int) -> MappingSpec:
    """
    Attempts to read sheets:
      - 'Laneization' with columns [dim, size, stride_B]
      - 'AbsIdx Mapping' (optional)
      - 'VecIdx Mapping' (optional)
    Falls back to a synthetic default if the sheet is missing.
    """
    p = Path(path)
    xl = pd.ExcelFile(p)
    dims: List[Tuple[str,int,int]] = []
    if 'Laneization' in xl.sheet_names:
        df = xl.parse('Laneization')
        for _, r in df.iterrows():
            dims.append((str(r.get('dim','D')), int(r.get('size',256)), int(r.get('stride_B', ATOM_B))))
    else:
        dims = [('W', 256, ATOM_B), ('H', 256, ATOM_B*256)]

    absidx = xl.parse('AbsIdx Mapping') if 'AbsIdx Mapping' in xl.sheet_names else None
    vecidx = xl.parse('VecIdx Mapping') if 'VecIdx Mapping' in xl.sheet_names else None

    # Basic cross-validation if both present: shapes must match
    if absidx is not None and vecidx is not None:
        if absidx.shape != vecidx.shape:
            raise ValueError('AbsIdx and VecIdx shapes mismatch')

    # Infer an input_shape from dims if possible
    try:
        if len(dims) > 0:
            sizes = [int(s) for (_d, s, _b) in dims]
            inferred_shape = tuple(sizes)
        else:
            inferred_shape = (1, 16, 16)
    except Exception:
        inferred_shape = (1, 16, 16)
        
    spec = MappingSpec(dims=dims, pixel_bytes=pixel_bits//8, absidx=absidx, vecidx=vecidx)
    spec.input_shape = inferred_shape
    
    return spec

def emit_config_xlsx(out_path: str | Path, choice, pixel_bits: int, zero_mode: str):
    df = pd.DataFrame({
        'lane': list(range(LANES)),
        'lift_B': [l*ATOM_B for l in choice.lifts],
        'residue': choice.residues
    })
    meta = pd.DataFrame({'key':['beta_eff_B','delta_b','pixel_bits','zero_mode'],
                         'value':[choice.beta_eff_B, choice.delta_b, pixel_bits, zero_mode]})
    with pd.ExcelWriter(out_path) as w:
        meta.to_excel(w, index=False, sheet_name='meta')
        df.to_excel(w, index=False, sheet_name='laneization')

def emit_report_txt(out_path: str | Path, choice, candidates: List[Tuple[str,int,int]], gap_budget_B: int | None):
    lines = []
    lines.append('=== Tensor Offline Compiler Report ===')
    lines.append(f'Chosen Dim: {choice.dim}')
    lines.append(f'Effective Stride (beta_eff_B): {choice.beta_eff_B}')
    lines.append(f'Bank Stride (delta_b): {choice.delta_b}')
    lines.append(f'Total Padding Cost (atoms): {choice.total_cost}')
    lines.append(f'Total Padding Cost (bytes): {choice.total_cost * ATOM_B}')
    if gap_budget_B is not None:
        lines.append(f'User Gap Budget (bytes): {gap_budget_B}')
        if getattr(choice, 'dim', '') == 'single_lane_fallback':
            lines.append('Gap-Budget Outcome: EXCEEDED budget -> single-lane fallback triggered')
        else:
            lines.append('Gap-Budget Outcome: WITHIN budget' if (choice.total_cost * ATOM_B) <= int(gap_budget_B) else 'Gap-Budget Outcome: EXCEEDED budget')
    lines.append('')
    lines.append('--- Evaluated Candidates ---')
    for dim, size, stride_B in candidates:
        if stride_B % ATOM_B != 0:
            lines.append(f' - {dim}: Invalid stride {stride_B} (not a multiple of {ATOM_B})')
            continue
        delta_b = stride_B // ATOM_B
        lines.append(f' - {dim}: beta={stride_B} B, Δb={delta_b}')
    Path(out_path).write_text("\n".join(lines))

def ensure_trace_xlsx(trace_csv_path: str) -> str:
    csv_p = Path(trace_csv_path)
    if not csv_p.exists():
        return str(csv_p.with_suffix('.xlsx'))  # nothing to convert
    df = pd.read_csv(csv_p)
    xlsx = csv_p.with_suffix('.xlsx')
    with pd.ExcelWriter(xlsx) as w:
        df.to_excel(w, index=False, sheet_name='trace')
    return str(xlsx)

# ---------------- Repacking ----------------
def repack_elements_to_atoms(num_elems: int, elem_B: int, zero_mode: str) -> Tuple[int, np.ndarray]:
    """
    Ensures element integrity when ATOM_B % elem_B != 0 by packing floor(ATOM_B/elem_B)
    elements per 16-B atom and padding the remainder in the last atom with zeros.
    Returns (num_atoms, mask_valid_elems_per_atom).
    """
    if elem_B <= 0: return 0, np.array([], dtype=int)
    per_atom = ATOM_B // elem_B
    if per_atom < 1: per_atom = 1
    
    num_atoms = int(np.ceil(num_elems / per_atom)) if per_atom > 0 else 0
    counts = np.full((num_atoms,), per_atom, dtype=int)
    
    if num_atoms > 0:
        rem = num_elems % per_atom
        if rem == 0 and num_elems > 0:
            rem = per_atom
        counts[-1] = max(0, rem)

    return num_atoms, counts

def try_build_dummy_workbook(out_path: Path, pixel_bits: int) -> str:
    # Create a minimal valid workbook for validation runs
    df = pd.DataFrame({'dim':['W','H'], 'size':[128,128], 'stride_B':[ATOM_B, ATOM_B*128]})
    with pd.ExcelWriter(out_path) as w:
        df.to_excel(w, index=False, sheet_name='Laneization')
    return str(out_path)