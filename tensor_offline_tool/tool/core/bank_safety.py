from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from .constants import LANES, BANKS, ATOM_B
from .hungarian import hungarian_rect

@dataclass
class LaneizationChoice:
    dim: str
    beta_eff_B: int
    delta_b: int
    lifts: List[int]
    residues: List[int]
    total_cost: int
    rationale: str

def choose_laneization(candidates, gap_budget_atoms: int | None = None, gap_budget_B: int | None = None):
    """
    candidates: tuples either (dim, stride_B) or (dim, size, stride_B)
    """
    best = None

    def solve_for_delta_b(delta_b: int):
        C = np.zeros((LANES, BANKS), dtype=int)
        for l in range(LANES):
            for r in range(BANKS):
                C[l, r] = (r - (delta_b * l)) % BANKS
        row_assign, total = hungarian_rect(C)
        residues = [row_assign[l] for l in range(LANES)]
        lifts = [C[l, residues[l]] for l in range(LANES)]
        return lifts, residues, int(total)

    for item in candidates:
        if len(item) == 2:
            dim, stride_B = item
        else:
            dim, _size, stride_B = item
        if stride_B % ATOM_B != 0:
            continue
        delta_b = stride_B // ATOM_B
        lifts, residues, total = solve_for_delta_b(delta_b)
        rationale = []
        if delta_b % 2 == 1:
            rationale.append("Δb odd -> zero collisions")
        if total == 0:
            rationale.append("zero padding cost")
        choice = LaneizationChoice(dim, stride_B, delta_b, lifts, residues, total, "; ".join(rationale) or "minimal assignment cost")
        if (best is None) or (choice.total_cost < best.total_cost) or (choice.total_cost == 0 and best.total_cost != 0):
            best = choice
    if best is None:
        best = LaneizationChoice("single_lane_fallback", 0, 0, [0]*LANES, list(range(LANES)), 0, "no valid dimension; single-lane fallback")
    budget_B = None
    if gap_budget_B is not None:
        try:
            budget_B = int(gap_budget_B)
        except Exception:
            budget_B = None
    elif gap_budget_atoms is not None:
        # Backward compatibility
        budget_B = int(gap_budget_atoms) * ATOM_B
    if budget_B is not None and (best.total_cost * ATOM_B) > budget_B:
        best = LaneizationChoice("single_lane_fallback", 0, 0, [0]*LANES, list(range(LANES)), 0, "gap budget exceeded; single-lane fallback")
    return best
