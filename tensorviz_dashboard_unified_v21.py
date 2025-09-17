#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tensorviz_dashboard.py
Single-file dashboard for visualizing tensor element mapping across ONNX-like ops.

Author: Gemini (based on user-provided code and specifications)
Date: August 10, 2025

Key Fixes and Enhancements (Version 11):
- [FIXED] Blank Page on Load: Corrected a critical JavaScript error caused by a
  mismatch between the HTML template and the script's event listeners. The UI
  controls for 'Diff Highlighting' and 'Link Zoom' have been restored to the
  header, allowing the script to execute correctly and render the page.
- [MAINTAINED] All previous features, including the configurable hover reference
  and corrected indexing logic, remain intact and are now fully functional.
"""

import argparse
import json
import sys
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

# =========================
# Operator Implementations (Unified)
# =========================

class Operator:
    def apply(self, arr: np.ndarray, mapping: np.ndarray):
        raise NotImplementedError

class Identity(Operator):
    def apply(self, arr: np.ndarray, mapping: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        return arr, mapping, {"type": "identity"}

class Transpose(Operator):
    def __init__(self, perm: Sequence[int]):
        self.perm = [int(p) for p in perm]

    def apply(self, arr: np.ndarray, mapping: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        r = arr.ndim
        if len(self.perm) != r:
            raise ValueError(f"Transpose perm length {len(self.perm)} != rank {r}")
        perm_norm = [p if p >= 0 else p + r for p in self.perm]
        if sorted(perm_norm) != list(range(r)):
            raise ValueError(f"Transpose perm {self.perm} (normalized {perm_norm}) is not a valid permutation of 0..{r-1}")
        diff_info = {"type": "transpose", "perm": perm_norm}
        return arr.transpose(perm_norm), mapping.transpose(perm_norm), diff_info

class Reshape(Operator):
    def __init__(self, newshape: Sequence[int]):
        self.newshape = tuple(int(x) for x in newshape)

    def apply(self, arr: np.ndarray, mapping: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        in_shape = arr.shape
        in_count = int(np.prod(in_shape))
        ns = list(self.newshape)
        if ns.count(-1) > 1:
            raise ValueError("Reshape supports at most one -1 (inferred) dimension.")
        if -1 in ns:
            known = 1
            for v in ns:
                if v != -1: known *= v
            if in_count > 0 and known > 0 and in_count % known != 0:
                raise ValueError(f"Cannot infer reshape dim: {in_count} not divisible by {known}")
            ns[ns.index(-1)] = in_count // known if known > 0 else 0


        out_shape = tuple(ns)
        out_count = int(np.prod(out_shape))
        if out_count != in_count:
            raise ValueError(f"Reshape element count mismatch: {in_count} -> {out_count}")

        diff_info = {"type": "reshape", "from_shape": in_shape, "to_shape": out_shape}
        return arr.reshape(out_shape, order="C"), mapping.reshape(out_shape, order="C"), diff_info

class Squeeze(Operator):
    def __init__(self, axes: Optional[Sequence[int]] = None):
        self.axes = None if axes is None else tuple(int(a) for a in axes)

    def apply(self, arr: np.ndarray, mapping: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        squeezed_axes = self.axes
        if squeezed_axes is None:
            squeezed_axes = tuple(i for i, s in enumerate(arr.shape) if s == 1)

        diff_info = {"type": "squeeze", "axes": squeezed_axes}
        return np.squeeze(arr, axis=self.axes), np.squeeze(mapping, axis=self.axes), diff_info

class Unsqueeze(Operator):
    def __init__(self, axes: Sequence[int]):
        self.axes = tuple(int(a) for a in axes)

    def apply(self, arr: np.ndarray, mapping: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        out_arr, out_map = arr, mapping
        r = arr.ndim
        axes_norm = []
        for a in self.axes:
            a_norm = a if a >= 0 else a + r + 1
            if not (-r - 1 <= a <= r):
                raise ValueError(f"Unsqueeze axis {a} out of allowed range [-{r+1}, {r}] for rank {r}")
            axes_norm.append(a_norm)
        for ax in sorted(axes_norm):
            out_arr = np.expand_dims(out_arr, axis=ax)
            out_map = np.expand_dims(out_map, axis=ax)
            r += 1
        diff_info = {"type": "unsqueeze", "axes": tuple(sorted(axes_norm))}
        return out_arr, out_map, diff_info

class Flatten(Operator):
    def __init__(self, axis: int = 1):
        self.axis = int(axis)

    def apply(self, arr: np.ndarray, mapping: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        a = self.axis
        if a < 0: a += arr.ndim
        
        d0 = int(np.prod(arr.shape[:a])) if a > 0 else 1
        d1 = int(np.prod(arr.shape[a:])) if a < arr.ndim else 1
        
        new_shape = (d0, d1)

        diff_info = {"type": "flatten", "axis": self.axis, "from_axes": list(range(arr.ndim)), "to_shape": new_shape}
        return arr.reshape(new_shape, order="C"), mapping.reshape(new_shape, order="C"), diff_info

class Slice(Operator):
    def __init__(self, starts: Sequence[int], ends: Sequence[int],
                 axes: Optional[Sequence[int]] = None,
                 steps: Optional[Sequence[int]] = None):
        self.starts, self.ends = list(starts), list(ends)
        self.axes, self.steps = axes, steps

    def apply(self, arr: np.ndarray, mapping: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        r = arr.ndim
        slicer = [slice(None)] * r
        axes_to_slice = list(range(r)) if self.axes is None else [int(a) for a in self.axes]
        steps = [1] * len(axes_to_slice) if self.steps is None else [int(s) for s in self.steps]

        for i, ax in enumerate(axes_to_slice):
            slicer[ax] = slice(self.starts[i], self.ends[i], steps[i])

        slicer_tuple = tuple(slicer)
        diff_info = {"type": "slice", "slicer": str(slicer_tuple)}
        return arr[slicer_tuple], mapping[slicer_tuple], diff_info

class Pad(Operator):
    def __init__(self, pads: Sequence[int], mode: str = "constant", value: float = 0.0):
        self.pads, self.mode, self.value = list(pads), str(mode), float(value)

    def apply(self, arr: np.ndarray, mapping: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        r = arr.ndim
        if len(self.pads) != 2 * r:
            raise ValueError("Pad expects pads of length 2*rank")
        
        pad_width = [(self.pads[d], self.pads[r + d]) for d in range(r)]
        padded_arr = np.pad(arr, pad_width, mode=self.mode, constant_values=self.value)
        padded_map = np.pad(mapping, pad_width, mode="constant", constant_values=-1) # -1 indicates no source

        diff_info = {"type": "pad", "pad_width": pad_width, "value": self.value}
        return padded_arr, padded_map, diff_info

class Concat(Operator):
    def __init__(self, axis: int, inputs: None|list=None):
        self.axis = int(axis)
    def apply(self, arr: np.ndarray, mapping: np.ndarray):
        # Supported when a leading 'S' (sequence) axis exists (produced by Split/SplitToSequence).
        if arr.ndim == 0 or arr.shape[0] <= 1:
            return arr, mapping, {"type":"concat", "axis": self.axis, "note":"no-sequence"}
        axis = self.axis if self.axis >= 0 else self.axis + (arr.ndim - 1)
        axis += 1  # account for S at front
        parts = [arr[i] for i in range(arr.shape[0])]
        parts_map = [mapping[i] for i in range(mapping.shape[0])]
        out = np.concatenate(parts, axis=axis-1)
        out_map = np.concatenate(parts_map, axis=axis-1)
        return out, out_map, {"type":"concat", "axis": self.axis}

    def apply_multi(self, tensors):
        # tensors: list of (ids, map, names)
        rank = tensors[0][0].ndim
        ax = self.axis if self.axis>=0 else self.axis + rank
        ids_out = np.concatenate([t[0] for t in tensors], axis=ax)
        map_out = np.concatenate([t[1] for t in tensors], axis=ax)
        return ids_out, map_out, {"type":"concat", "axis": ax, "num_inputs": len(tensors)}

class Split(Operator):
    def __init__(self, axis: int, split: list = None, num_splits: int = None):
        self.axis = int(axis); self.split = None if split is None else [int(s) for s in split]; self.num_splits = None if num_splits is None else int(num_splits)
    def apply(self, arr: np.ndarray, mapping: np.ndarray):
        r = arr.ndim; ax = self.axis if self.axis>=0 else self.axis + r
        length = arr.shape[ax]
        if self.split is not None:
            sizes = self.split
            if sum(sizes) != length: raise ValueError("Split sizes must sum to axis length")
        else:
            if self.num_splits is None: raise ValueError("Provide 'split' or 'num_splits'")
            if length % self.num_splits != 0: raise ValueError("Axis not divisible by num_splits")
            part = length // self.num_splits; sizes = [part]*self.num_splits
        pieces = np.split(arr, np.cumsum(sizes)[:-1], axis=ax)
        pieces_map = np.split(mapping, np.cumsum(sizes)[:-1], axis=ax)
        out = np.stack(pieces, axis=0)     # S axis at front
        out_map = np.stack(pieces_map, axis=0)
        return out, out_map, {"type":"split", "axis": ax, "sizes": sizes}

class Expand(Operator):
    def __init__(self, shape: list):
        self.shape = tuple(int(x) for x in shape)
    def apply(self, arr: np.ndarray, mapping: np.ndarray):
        out = np.broadcast_to(arr, self.shape)
        out_map = np.broadcast_to(mapping, self.shape)
        return out, out_map, {"type":"expand", "shape": self.shape}

class Tile(Operator):
    def __init__(self, repeats: list):
        self.repeats = tuple(int(x) for x in repeats)
    def apply(self, arr: np.ndarray, mapping: np.ndarray):
        out = np.tile(arr, self.repeats)
        out_map = np.tile(mapping, self.repeats)
        return out, out_map, {"type":"tile", "repeats": self.repeats}

class ShapeOp(Operator):
    def apply(self, arr: np.ndarray, mapping: np.ndarray):
        vals = np.array(arr.shape, dtype=np.int64)
        out_map = -np.ones(vals.shape, dtype=np.int64)
        return vals, out_map, {"type":"shape"}

class ConstantOfShape(Operator):
    def __init__(self, value: float = 0.0):
        self.value = value
    def apply(self, arr: np.ndarray, mapping: np.ndarray):
        if arr.ndim != 1: raise ValueError("ConstantOfShape expects 1D shape tensor")
        shape = tuple(int(x) for x in arr.tolist())
        out = np.full(shape, self.value, dtype=np.float32)
        out_map = -np.ones(shape, dtype=np.int64)
        return out, out_map, {"type":"constant_of_shape", "value": self.value}

class DepthToSpace(Operator):
    def __init__(self, blocksize: int):
        self.blocksize = int(blocksize)
    def apply(self, arr: np.ndarray, mapping: np.ndarray):
        if arr.ndim != 4: raise ValueError("DepthToSpace expects NCHW")
        N,C,H,W = arr.shape; b = self.blocksize
        if C % (b*b) != 0: raise ValueError("C must be divisible by blocksize^2")
        outC = C // (b*b)
        x = arr.reshape(N, b, b, outC, H, W).transpose(0,3,4,1,5,2).reshape(N, outC, H*b, W*b)
        m = mapping.reshape(N, b, b, outC, H, W).transpose(0,3,4,1,5,2).reshape(N, outC, H*b, W*b)
        return x, m, {"type":"depth_to_space", "blocksize": b}

class SpaceToDepth(Operator):
    def __init__(self, blocksize: int):
        self.blocksize = int(blocksize)
    def apply(self, arr: np.ndarray, mapping: np.ndarray):
        if arr.ndim != 4: raise ValueError("SpaceToDepth expects NCHW")
        N,C,H,W = arr.shape; b = self.blocksize
        if H % b != 0 or W % b != 0: raise ValueError("H,W must be multiples of blocksize")
        outC = C * (b*b)
        x = (arr.reshape(N, C, H//b, b, W//b, b).transpose(0,3,5,1,2,4).reshape(N, outC, H//b, W//b))
        m = (mapping.reshape(N, C, H//b, b, W//b, b).transpose(0,3,5,1,2,4).reshape(N, outC, H//b, W//b))
        return x, m, {"type":"space_to_depth", "blocksize": b}

class Gather(Operator):
    def __init__(self, axis: int, indices: list):
        self.axis = int(axis); self.indices = np.array(indices, dtype=np.int64)
    def apply(self, arr: np.ndarray, mapping: np.ndarray):
        ax = self.axis if self.axis>=0 else self.axis + arr.ndim
        return np.take(arr, self.indices, axis=ax), np.take(mapping, self.indices, axis=ax), {"type":"gather", "axis": ax}

class GatherElements(Operator):
    def __init__(self, axis: int, indices: list):
        self.axis = int(axis); self.indices = np.array(indices, dtype=np.int64)
    def apply(self, arr: np.ndarray, mapping: np.ndarray):
        ax = self.axis if self.axis>=0 else self.axis + arr.ndim
        return np.take_along_axis(arr, self.indices, axis=ax), np.take_along_axis(mapping, self.indices, axis=ax), {"type":"gather_elements", "axis": ax}

        ax = self.axis if self.axis>=0 else self.axis + arr.ndim
        return np.take_along_axis(arr, self.indices, axis=ax), np.take_along_axis(mapping, self.indices, axis=ax), {"type":"gather_elements", "axis": ax}

class GatherND(Operator):
    def __init__(self, indices: list):
        self.indices = np.array(indices, dtype=np.int64)
    def apply(self, arr: np.ndarray, mapping: np.ndarray):
        idx = self.indices; r = arr.ndim
        if idx.shape[-1] != r: raise ValueError("GatherND supports indices[..., rank] only (scalar gather)")
        out_shape = idx.shape[:-1]
        out = np.empty(out_shape, dtype=arr.dtype); out_map = np.empty(out_shape, dtype=np.int64)
        it = np.nditer(np.zeros(out_shape, dtype=np.uint8), flags=['multi_index'])
        while not it.finished:
            coords = tuple(idx[it.multi_index].tolist())
            out[it.multi_index] = arr[coords]; out_map[it.multi_index] = mapping[coords]
            it.iternext()
        return out, out_map, {"type":"gather_nd"}

class ReverseSequence(Operator):
    def __init__(self, seq_lengths: list, batch_axis: int = 0, time_axis: int = 1):
        self.seq_lengths = np.array(seq_lengths, dtype=np.int64)
        self.batch_axis = int(batch_axis); self.time_axis = int(time_axis)
    def apply(self, arr: np.ndarray, mapping: np.ndarray):
        ba = self.batch_axis if self.batch_axis>=0 else self.batch_axis + arr.ndim
        ta = self.time_axis if self.time_axis>=0 else self.time_axis + arr.ndim
        if self.seq_lengths.shape[0] != arr.shape[ba]: raise ValueError("seq_lengths length must match batch size")
        perm = list(range(arr.ndim)); perm[0], perm[ba] = perm[ba], perm[0]; perm[1], perm[ta] = perm[ta], perm[1]
        inv = np.argsort(perm)
        A = arr.transpose(perm).copy(); M = mapping.transpose(perm).copy()
        for b in range(A.shape[0]):
            L = int(self.seq_lengths[b]); 
            if L>1: A[b,:L] = A[b,:L][::-1]; M[b,:L] = M[b,:L][::-1]
        return A.transpose(inv), M.transpose(inv), {"type":"reverse_sequence", "batch_axis": ba, "time_axis": ta}

class SplitToSequence(Operator):
    def __init__(self, axis: int, split: list = None, num_splits: int = None):
        self.axis = int(axis); self.split = None if split is None else [int(s) for s in split]; self.num_splits = None if num_splits is None else int(num_splits)
    def apply(self, arr: np.ndarray, mapping: np.ndarray):
        r = arr.ndim; ax = self.axis if self.axis>=0 else self.axis + r
        length = arr.shape[ax]
        if self.split is not None:
            sizes = self.split; 
            if sum(sizes) != length: raise ValueError("split sizes must sum to axis length")
        else:
            if self.num_splits is None: raise ValueError("num_splits or split required")
            if length % self.num_splits != 0: raise ValueError("Equal split requires divisibility")
            part = length // self.num_splits; sizes = [part]*self.num_splits
        pieces = np.split(arr, np.cumsum(sizes)[:-1], axis=ax)
        pieces_map = np.split(mapping, np.cumsum(sizes)[:-1], axis=ax)
        out = np.stack(pieces, axis=0); out_map = np.stack(pieces_map, axis=0)
        return out, out_map, {"type":"split_to_sequence", "axis": ax, "sizes": sizes}

class ConcatFromSequence(Operator):
    def __init__(self, axis: int, new_axis: int = 0):
        self.axis = int(axis); self.new_axis = int(new_axis)
    def apply(self, arr: np.ndarray, mapping: np.ndarray):
        if arr.shape[0] <= 1: return arr, mapping, {"type":"concat_from_sequence", "note":"no-sequence"}
        if self.new_axis:  # keep S and insert a new axis; for simplicity, no change
            return arr, mapping, {"type":"concat_from_sequence", "new_axis":1}
        axis = self.axis if self.axis>=0 else self.axis + (arr.ndim - 1)
        axis += 1
        parts = [arr[i] for i in range(arr.shape[0])]
        parts_map = [mapping[i] for i in range(mapping.shape[0])]
        out = np.concatenate(parts, axis=axis-1); out_map = np.concatenate(parts_map, axis=axis-1)
        return out, out_map, {"type":"concat_from_sequence", "axis": self.axis}


OP_REGISTRY = {
    "identity": Identity, "transpose": Transpose, "reshape": Reshape,
    "squeeze": Squeeze, "unsqueeze": Unsqueeze, "flatten": Flatten,
    "slice": Slice, "pad": Pad,
    "concat": Concat, "split": Split, "expand": Expand, "tile": Tile,
    "shape": ShapeOp, "constantofshape": ConstantOfShape,
    "depthtospace": DepthToSpace, "spacetodepth": SpaceToDepth,
    "gather": Gather, "gatherelements": GatherElements, "gathernd": GatherND,
    "reversesequence": ReverseSequence, "splittosequence": SplitToSequence, "concatfromsequence": ConcatFromSequence
}

# =========================
# Engine / helpers
# =========================

def _get_initial_axis_names(config: Dict[str, Any], rank: int) -> List[str]:
    """Determines initial axis names from config, defaulting to NCHW pattern."""
    if config.get("axes"):
        names = config["axes"]
        if len(names) == rank:
            return [str(n) for n in names]
    
    default_4d = ["N", "C", "H", "W"]
    if rank == 4: return default_4d
    if rank == 3: return default_4d[1:]
    if rank == 2: return default_4d[2:]
    return [f"Ax{i}" for i in range(rank)]


def _apply_axis_names(names: List[str], op: Operator, in_shape: Tuple[int, ...], diff_info: Dict[str, Any]) -> List[str]:
    """Propagates axis names through an operation."""
    op_type = diff_info.get("type", "")
    
    if op_type == "transpose": return [names[i] for i in op.perm]
    if op_type == "squeeze": return [n for i, n in enumerate(names) if i not in diff_info["axes"]]
    
    if op_type == "unsqueeze":
        out_names = list(names)
        for ax in sorted(diff_info["axes"], reverse=True):
            out_names.insert(ax, f"NewAx{ax}")
        return out_names
        
    if op_type == "flatten":
        a = op.axis
        if a < 0: a += len(names)
        d0_name = "*".join(names[:a]) or "1"
        d1_name = "*".join(names[a:]) or "1"
        if not d0_name: return [d1_name]
        if not d1_name: return [d0_name]
        return [d0_name, d1_name]

    if op_type == "reshape":
        # For reshape, axis names lose their original meaning.
        # A more advanced implementation could track merges/splits.
        # For now, we generate generic names.
        return [f"Ax{i}" for i in range(len(diff_info["to_shape"]))]
    
    # For Pad, Slice, Identity, etc., names are preserved.
    return names[:]

def _id_tensor(shape: Sequence[int]) -> np.ndarray:
    total = int(np.prod(shape, dtype=np.int64))
    if total == 0:
        return np.empty(shape, dtype=np.int64)
    return np.arange(total, dtype=np.int64).reshape(tuple(shape), order="C")

def get_coords_from_flat_index_py(flat_index: int, shape: Tuple[int, ...]) -> Optional[List[int]]:
    """Python-side utility for calculating coordinates from a flat index."""
    if flat_index < 0: return None
    coords = []
    remainder = flat_index
    
    # Calculate strides
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i+1] * shape[i+1]

    for i in range(len(shape)):
        coord = remainder // strides[i] if strides[i] > 0 else 0
        coords.append(int(coord))
        if strides[i] > 0:
            remainder %= strides[i]
    return coords

def evaluate_pipeline(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluates a pipeline based on a spec-compliant config dictionary."""
    if "shape" not in config.get("tensor", {}):
        raise ValueError("Config must include 'tensor.shape'")
    
    shape = tuple(int(x) for x in config["tensor"]["shape"])
    ops_cfg = config.get("operations", [])
    
    ids = _id_tensor(shape)
    mapping = np.copy(ids) # Mapping for input is itself
    initial_names = _get_initial_axis_names(config.get("tensor", {}), len(shape))
    
    # This map is from flat ID to original coordinates. For the input tensor, this is also the "current" coordinates.
    initial_coords_map = {i: get_coords_from_flat_index_py(i, shape) for i in range(len(ids.flatten()))}

    states: List[Dict[str, Any]] = [{
        "name": "Input", "op_type": "Input", "shape": shape, "axis_names": initial_names,
        "ids": ids, "mapping": mapping, "diff_info": {"type": "identity"},
        "original_shape": shape,
        "flat_ids_to_original_coords": initial_coords_map, # This is for tooltips, to always show the origin
        "flat_ids_to_current_coords": initial_coords_map # For the input state, current coords are original coords
    }]

    cur_ids, cur_map, cur_names = ids, mapping, initial_names
    
    for op_cfg in ops_cfg:
        op_type_str = op_cfg.get("type", "").lower()
        if op_type_str not in OP_REGISTRY:
            raise ValueError(f"Unsupported operation type: {op_cfg.get('type')}")
        
        params = {k: v for k, v in op_cfg.items() if k not in ["type", "name"]}
        op_instance = OP_REGISTRY[op_type_str](**params)
        
        in_shape = cur_ids.shape
        if hasattr(op_instance, 'apply_multi') and isinstance(op_cfg.get('inputs', None), list):
            sources = []
            for ref in op_cfg['inputs']:
                if ref == 'current':
                    sources.append((cur_ids, cur_map, cur_names))
                else:
                    s = states[int(ref)]
                    sources.append((s['ids'], s['mapping'], s['axis_names']))
            nxt_ids, nxt_map, diff_info = op_instance.apply_multi(sources)
        else:
            nxt_ids, nxt_map, diff_info = op_instance.apply(cur_ids, cur_map)
        nxt_names = _apply_axis_names(cur_names, op_instance, in_shape, diff_info)

        # Create a reverse mapping from flat original ID to the coordinates in the current tensor shape
        flat_ids_to_current_coords = {}
        for coords in np.ndindex(nxt_map.shape):
            flat_id = nxt_map[coords]
            if flat_id >= 0:
                flat_ids_to_current_coords[int(flat_id)] = list(coords)

        states.append({
            "name": op_cfg.get("name", op_type_str.capitalize()), "op_type": op_type_str,
            "shape": nxt_ids.shape, "axis_names": nxt_names, "ids": nxt_ids, "mapping": nxt_map,
            "diff_info": diff_info,
            "original_shape": shape,
            "flat_ids_to_original_coords": initial_coords_map, # Always pass the original map for reference
            "flat_ids_to_current_coords": flat_ids_to_current_coords # Pass the new map for the current state
        })
        cur_ids, cur_map, cur_names = nxt_ids, nxt_map, nxt_names
        
    return states

# =========================
# HTML Rendering
# =========================
def _plotly_js_inline() -> str:
    return "<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>"

def render_dashboard_html(states: List[Dict[str, Any]], title: Optional[str] = None) -> str:
    serializable_states = []
    for s in states:
        serializable_states.append({
            "name": s["name"], "op_type": s["op_type"],
            "shape": list(s["shape"]), "axis_names": list(s["axis_names"]),
            "ids": s["ids"].astype(np.int64).tolist(),
            "mapping": s["mapping"].astype(np.int64).tolist(),
            "diff_info": s["diff_info"],
            "original_shape": list(s["original_shape"]),
            "flat_ids_to_original_coords": {k: v for k,v in s["flat_ids_to_original_coords"].items()},
            "flat_ids_to_current_coords": {k: v for k,v in s["flat_ids_to_current_coords"].items()}
        })

    data_json = json.dumps({"states": serializable_states}, separators=(",", ":")).replace("</", "<\\/")
    lib_js = _plotly_js_inline()
    page_title = title or "TensorViz Dashboard"
    max_step = str(len(states) - 1)

    template = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>%%TITLE%%</title>
<style>
:root { --bg:#fff; --fg:#111; --muted:#666; --panel:#f6f7f9; --grid:#d9d9e0; --accent:#3b82f6; --highlight-bg:rgba(59, 130, 246, 0.2); }
.dark { --bg:#0b0b0b; --fg:#f0f0f0; --muted:#aaa; --panel:#171717; --grid:#333; --accent:#60a5fa; --highlight-bg:rgba(96, 165, 250, 0.2); }
body { margin:0; background:var(--bg); color:var(--fg); font-family:Segoe UI, Roboto, Arial, sans-serif; }
header { display:flex; gap:10px; align-items:center; padding:10px 16px; border-bottom:1px solid var(--grid); flex-wrap:wrap; }
h1 { font-size:16px; margin:0 8px 0 0; }
select,input,button { background:var(--panel); color:var(--fg); border:1px solid var(--grid); border-radius:6px; padding:6px 8px; font-size:12px; }
button { cursor:pointer; }
.badge { padding:2px 6px; border-radius:6px; border:1px solid var(--grid); font-size:11px; }
.timeline { display:flex; gap:6px; padding:6px 10px; border-bottom:1px solid var(--grid); overflow:auto; }
.titem { border:1px solid var(--grid); background:var(--panel); border-radius:6px; padding:4px 8px; cursor:pointer; font-size:12px; white-space:nowrap; }
.titem.active { outline:2px solid var(--accent); }
.row { display:grid; grid-template-columns:1fr 1fr; gap:10px; padding:10px; align-items:start; }
.panel { background:var(--panel); border:1px solid var(--grid); border-radius:10px; padding:8px; }
.panel h2 { margin:4px 0 8px 6px; font-size:14px; }
.controls { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.slicers { display:flex; gap:10px; flex-wrap:wrap; margin-top:6px; }
.facet-grid { display:grid; gap:8px; }
.small { font-size:11px; color:var(--muted); }
.plotcell { position: relative; height: 320px; }
.footer { padding:10px; color:var(--muted); font-size:12px; text-align:center; min-height: 3em; }
hr.sep { border:none; border-top:1px solid var(--grid); margin:6px 0; }
.axis-label { padding: 2px 5px; border-radius: 4px; color: white; font-weight: bold; margin-right: 4px; }
</style>
%%PLOTLY%%
</head>
<body class="light">
<header>
  <h1>TensorViz</h1>
  <label>Theme <select id="themeSel"><option value="light">Light</option><option value="dark">Dark</option></select></label>
  <label>Step <input type="range" id="stepRange" min="0" max="%%MAXSTEP%%" value="0"/> <span id="stepLabel" class="badge">0 / %%MAXSTEP%%</span></label>
  <label>Pairing <select id="pairSel"><option value="pair" selected>k → k+1</option><option value="same">k → k</option><option value="input">0 → k+1</option></select></label>
  <label><input type="checkbox" id="diffMode" checked/> Diff Highlighting</label>
  <label><input type="checkbox" id="linkZoom"/> Link zoom</label>
  <label><input type="checkbox" id="hover3d" checked/> 3D Hover</label>
  <label>Hover Ref: <select id="hoverRef"><option value="original">Original Input</option><option value="previous">Previous Step</option></select></label>
  <button id="btnReset">Reset UI</button>
</header>
<div id="timeline" class="timeline"></div>
<div class="row">
  <div class="panel" id="leftPanel">
    <h2 id="leftTitle">Left</h2>
    <div class="controls">
      <label>Axes <select id="leftAxA"></select> × <select id="leftAxB"></select></label>
      <label>Facet <select id="leftFacet"></select></label>
      <label>Color by <select id="leftColorAxis"></select></label>
    </div>
    <div id="leftSlicers" class="slicers"></div><div id="leftGrid" class="facet-grid"></div>
  </div>
  <div class="panel" id="rightPanel">
    <h2 id="rightTitle">Right</h2>
    <div class="controls">
      <label>Axes <select id="rightAxA"></select> × <select id="rightAxB"></select></label>
      <label>Facet <select id="rightFacet"></select></label>
      <label>Color by <select id="rightColorAxis"></select></label>
      <button id="btnExportPNG">PNG</button><button id="btnExportSVG">SVG</button>
      <button id="btnExportGridPNGs">Grid PNGs</button><button id="btnExportJSON">Data</button>
    </div>
    <div id="rightSlicers" class="slicers"></div><div id="rightGrid" class="facet-grid"></div>
  </div>
</div>
<div class="panel" id="panel3d" style="margin:0 10px 10px 10px;">
  <h2>3D View</h2>
  <div class="controls">
    <label>Use step <select id="step3d"></select></label>
    <label>X<select id="axX"></select>Y<select id="axY"></select>Z<select id="axZ"></select></label>
    <label>Color by <select id="colorAxis3d"></select></label>
    <label>Style <select id="style3d"><option value="voxels">Voxels</option><option value="points">Points</option></select></label>
    <button id="resetCam">Reset Camera</button>
  </div>
  <div id="slicers3d" class="slicers"></div>
  <div id="plot3d" style="width:100%;height:560px;"></div>
</div>
<div class="footer">
  <span id="status">Hover over an element to see its coordinate mapping.</span>
</div>

<script id="data" type="application/json">%%DATAJSON%%</script>
<script>
(function(){
  const STORAGE_KEY='tensorviz_ui_v22';
  const DATA = JSON.parse(document.getElementById('data').textContent);
  const states = DATA.states;
  const N = states.length;
  const AXIS_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];

  function turboRGB(t){
    t = Math.max(0, Math.min(1, t));
    const r = 0.13572138 + 4.61539260*t - 42.66032258*t*t + 132.13108234*t*t*t - 152.94239396*t*t*t*t + 59.28637943*t*t*t*t*t;
    const g = 0.09140261 + 2.19418839*t + 4.84296658*t*t - 14.18503333*t*t*t + 4.27729857*t*t*t*t + 2.82956604*t*t*t*t*t;
    const b = 0.10667330 + 12.64194608*t - 60.58204836*t*t + 110.36276771*t*t*t - 89.90310912*t*t*t*t + 27.34824973*t*t*t*t*t;
    return `rgb(${(r*255)|0},${(g*255)|0},${(b*255)|0})`;
  }
  const TURBO_SCALE = Array.from({length: 11}, (_, i) => [i/10, turboRGB(i/10)]);

  let CTX = {leftStep:0, rightStep: Math.min(1,N-1)};

  function getElementData(arr, indices) { let ref = arr; for (const idx of indices) { if(ref === undefined) return undefined; ref = ref[idx]; } return ref; }

  function getAbsIndex(coords, shape) {
      if (!coords || !shape || coords.length !== shape.length) return -1;
      let strides = new Array(shape.length);
      strides[shape.length - 1] = 1;
      for (let i = shape.length - 2; i >= 0; i--) {
          strides[i] = strides[i + 1] * shape[i + 1];
      }
      let absIndex = 0;
      for (let i = 0; i < coords.length; i++) {
          absIndex += coords[i] * strides[i];
      }
      return absIndex;
  }

  function getCoordsFromAbsIndex(absIndex, shape) {
      if (!shape || shape.length === 0) return [];
      let strides = new Array(shape.length);
      strides[shape.length - 1] = 1;
      for (let i = shape.length - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
      const coords = new Array(shape.length).fill(0);
      let rem = absIndex;
      for (let i = 0; i < shape.length; i++) { coords[i] = Math.floor(rem / strides[i]); rem = rem % strides[i]; }
      return coords;
  }

  
  const formatIndexVector = (names, coords) => {
      if (!coords || !names || names.length !== coords.length) return "N/A";
      return `(${coords.map((c, i) => `${names[i] || '?'}:${c}`).join(', ')})`;
  };
  
  function getScalarValues(planeMappings, colorAxis, originalShape, flatIdMap) {
      return planeMappings.map(row => row.map(flatIndex => {
          if (flatIndex < 0) return 0.5; // Padded elements are grey
          const coords = flatIdMap[flatIndex];
          if (!coords) return 0.5;
          const val = coords[colorAxis];
          const maxVal = originalShape[colorAxis] - 1;
          return maxVal > 0 ? val / maxVal : 0.5;
      }));
  }

  function heatmap(div, data, opts) {
      const { panel: panelColor } = { panel: getComputedStyle(document.body).getPropertyValue('--panel').trim() };
      const trace={ type:'heatmap', z:data.scalars, customdata:data.custom,
        hovertemplate:'AbsIdx: %{customdata[0]}<br>VecIdx: %{customdata[1]}<extra></extra>',
        colorscale:TURBO_SCALE, zmin:0, zmax:1, zsmooth:false, showscale:false,
        xgap:2, ygap:2
      };
      const layout={ margin:{l:40,r:10,t:20,b:40},
        xaxis:{title:opts.axB, constrain:'domain'},
        yaxis:{title:opts.axA, autorange:'reversed', scaleanchor:'x', scaleratio:1},
        dragmode:'select', paper_bgcolor: panelColor, plot_bgcolor: 'rgba(0,0,0,0)'
      };
      Plotly.react(div, [trace], layout, {displaylogo:false, responsive:true});
      div.__plot_data = data; div.__layout = layout; div.__opts = opts;
  }
  
  function drawDiffHighlights(div, diffInfo, st, axA) {
      const shapes = [];
      if (diffInfo.type === 'flatten' && diffInfo.from_axes.length > diffInfo.to_shape.length) {
          const mergedAxes = diffInfo.from_axes.slice(0, diffInfo.axis);
          if (mergedAxes.includes(axA)) {
            shapes.push({ type: 'rect', xref: 'paper', yref: 'paper',
                x0: 0, y0: 0, x1: 1, y1: 1,
                line: { color: AXIS_COLORS[axA % AXIS_COLORS.length], width: 4, dash: 'solid' },
                fillcolor: 'rgba(0,0,0,0)'
            });
          }
      }
      Plotly.relayout(div, {shapes: shapes});
  }

  function renderGrid(container, st, axA, axB, facetAx, slices, prefix, colorAxis) {
    container.innerHTML='';
    const facetSize = (facetAx >= 0) ? st.shape[facetAx] : 1;
    container.style.gridTemplateColumns = `repeat(${Math.min(4, facetSize)}, minmax(260px, 1fr))`;

    for(let f=0; f<facetSize; f++){
      const cell=document.createElement('div'); cell.className = 'plotcell';
      const head=document.createElement('div'); head.className='small';
      head.textContent = (facetAx>=0) ? `${st.axis_names[facetAx]}=${f}` : '';
      const plot=document.createElement('div'); plot.style.width="100%"; plot.style.height="100%";
      plot.dataset.prefix = prefix; plot.dataset.facet = String(f);
      window.plotGrid = window.plotGrid || {left:[], right:[]};
      window.plotGrid[prefix][f] = plot;
      cell.appendChild(head); cell.appendChild(plot); container.appendChild(cell);

      const sl = {...slices}; if(facetAx>=0) sl[facetAx]=f;
      
      const planeMappings = (function() {
          const m = []; const idx = new Array(st.shape.length).fill(0);
          for(let ia = 0; ia < st.shape[axA]; ia++) {
              const row = [];
              for(let ib = 0; ib < st.shape[axB]; ib++) {
                  for(let d=0; d<st.shape.length; d++) idx[d] = (d===axA)?ia:(d===axB)?ib:(sl[d]||0);
                  row.push(getElementData(st.mapping, idx));
              } m.push(row);
          } return m;
      })();

      const scalars = getScalarValues(planeMappings, colorAxis, states[0].original_shape, states[0].flat_ids_to_original_coords);
// Build customdata grid: [AbsIdx, VecIdxStr]
const planeCustom = [];
const hoverMode = document.getElementById('hoverRef').value; // 'original' | 'previous'
const idxTmp = new Array(st.shape.length).fill(0);
for (let ia = 0; ia < planeMappings.length; ia++) {
  const rowC = [];
  for (let ib = 0; ib < planeMappings[ia].length; ib++) {
    for (let d = 0; d < st.shape.length; d++) idxTmp[d] = (d===axA)?ia:(d===axB)?ib:(sl[d]||0);
    const flatId = planeMappings[ia][ib];
    if (prefix === 'left') {
      const absIdx = getAbsIndex(idxTmp, st.shape);
      rowC.push([absIdx, formatIndexVector(st.axis_names, idxTmp)]);
    } else {
      if (flatId < 0) { rowC.push([-1, '—']); }
      else if (hoverMode === 'original') {
        let ocoords = states[0].flat_ids_to_current_coords[flatId];
              if (!ocoords) ocoords = getCoordsFromAbsIndex(flatId, states[0].shape);
        rowC.push([flatId, formatIndexVector(states[0].axis_names, ocoords)]);
      } else {
        const prevState = states[Math.max(CTX.rightStep - 1, 0)];
        const pcoords = prevState.flat_ids_to_current_coords[flatId];
        const pabs = getAbsIndex(pcoords, prevState.shape);
        rowC.push([pabs, formatIndexVector(prevState.axis_names, pcoords)]);
      }
    }
  }
  planeCustom.push(rowC);
}
heatmap(plot, {scalars, mappings: planeMappings, custom: planeCustom}, {axA:st.axis_names[axA], axB:st.axis_names[axB]});
      
      plot.on('plotly_hover', (ev) => {
          if (!ev.points || ev.points.length === 0) return;
          const pt = ev.points[0];
          const i = Math.round(pt.y);
          const j = Math.round(pt.x);
          const flatIdx = (plot.__plot_data && plot.__plot_data.mappings && plot.__plot_data.mappings[i] ? plot.__plot_data.mappings[i][j] : -1);
          const statusEl = document.getElementById('status');
          
          if (flatIdx < 0) {
              statusEl.innerHTML = 'Padded Element';
              return;
          }

          let statusHtml = '';
          if (prefix === 'left') {
              const state = states[CTX.leftStep];
              const coords = state.flat_ids_to_current_coords[flatIdx];
              if (coords) {
                  const absIndex = getAbsIndex(coords, state.shape);
                  statusHtml = `<b>Left Panel:</b> Abs: ${absIndex}, Vec: ${formatIndexVector(state.axis_names, coords)}`;
              } else {
                  statusHtml = `Info not available for ID ${flatIdx} at this step.`;
              }
          } else { // Right panel
              const hoverMode = document.getElementById('hoverRef').value;
              const sourceState = (hoverMode === 'original') ? states[0] : states[CTX.leftStep];
              const sourceCoords = sourceState.flat_ids_to_current_coords[flatIdx];
              if (sourceCoords) {
                  const sourceAbsIndex = getAbsIndex(sourceCoords, sourceState.shape);
                  statusHtml = `<b>Source (${hoverMode}):</b> Abs: ${sourceAbsIndex}, Vec: ${formatIndexVector(sourceState.axis_names, sourceCoords)}`;
              } else {
                  statusHtml = `Info not available for ID ${flatIdx} in source step.`;
              }
          }
          statusEl.innerHTML = statusHtml;
          // Linked highlight between panels
          try {
              const facet = parseInt(plot.dataset.facet || '0', 10);
              if (prefix === 'left') {
                  const rightPlots = (window.plotGrid && window.plotGrid.right) ? window.plotGrid.right : [];
                  const rightPlot = rightPlots[facet];
                  if (rightPlot && rightPlot.__plot_data) {
                      const mapR = rightPlot.__plot_data.mappings || [];
                      let hi=-1, hj=-1;
                      for (let i=0;i<mapR.length;i++){ const row=mapR[i]; for(let j=0;j<row.length;j++){ if (row[j]===flatIdx){hi=i;hj=j;break;} } if(hi>=0)break; }
                      const shape = (hi>=0)?[{
                          type:'rect', xref:'x', yref:'y', x0:hj-0.5, x1:hj+0.5, y0:hi-0.5, y1:hi+0.5,
                          line:{width:6, color:'#2563eb'}, fillcolor:'rgba(37,99,235,0.28)'
                      }] : [];
                      Plotly.relayout(rightPlot, {shapes: shape});
                      if (hi>=0) { setTimeout(()=> Plotly.relayout(rightPlot, {'shapes[0].line.width':1}), 140); }
                  }
              } else if (prefix === 'right') {
                  const leftPlots = (window.plotGrid && window.plotGrid.left) ? window.plotGrid.left : [];
                  const leftPlot = leftPlots[facet];
                  if (leftPlot && leftPlot.__plot_data) {
                      const mapL = leftPlot.__plot_data.mappings || [];
                      let hi=-1, hj=-1;
                      for (let i=0;i<mapL.length;i++){ const row=mapL[i]; for(let j=0;j<row.length;j++){ if (row[j]===flatIdx){hi=i;hj=j;break;} } if(hi>=0)break; }
                      const shape = (hi>=0)?[{
                          type:'rect', xref:'x', yref:'y', x0:hj-0.5, x1:hj+0.5, y0:hi-0.5, y1:hi+0.5,
                          line:{width:6, color:'#16a34a'}, fillcolor:'rgba(22,163,74,0.28)'
                      }] : [];
                      Plotly.relayout(leftPlot, {shapes: shape});
                      
      // Mirror: hover on right, highlight corresponding pixel on left
      if (prefix === 'right') {
          const leftPlots = (window.plotGrid && window.plotGrid.left) ? window.plotGrid.left : [];
          const leftPlot = leftPlots[facet];
          if (leftPlot && leftPlot.__plot_data) {
              const mapL = leftPlot.__plot_data.mappings || [];
              let hi=-1, hj=-1; for (let i=0;i<mapL.length;i++){ const row=mapL[i]; for(let j=0;j<row.length;j++){ if (row[j]===flatIdx){hi=i;hj=j;break;} } if(hi>=0)break; }
              const shapeL = (hi>=0)?[{type:'rect',xref:'x',yref:'y',x0:hj-0.5,x1:hj+0.5,y0:hi-0.5,y1:hi+0.5,
                  line:{width:6,color:'#16a34a'},fillcolor:'rgba(22,163,74,0.28)'}]:[];
              Plotly.relayout(leftPlot, {shapes: shapeL});
              if (hi>=0){ setTimeout(()=>Plotly.relayout(leftPlot, {'shapes[0].line.width':1}), 220);} }
      }
if (hi>=0) { setTimeout(()=> Plotly.relayout(leftPlot, {'shapes[0].line.width':1}), 140); }
                  }
              }
          } catch(e) { /* no-op */ }

          if (prefix === 'left') {
              try {
                  const targetFacet = parseInt(plot.dataset.facet || '0', 10);
                  const rightPlots = (window.plotGrid && window.plotGrid.right) ? window.plotGrid.right : [];
                  const rightPlot = rightPlots[targetFacet];
                  if (rightPlot && rightPlot.__plot_data) {
                      const mapR = rightPlot.__plot_data.mappings || [];
                      let hi=-1, hj=-1;
                      for (let ii=0; ii<mapR.length; ii++) { const row = mapR[ii] || []; for (let jj=0; jj<row.length; jj++) { if (row[jj] === flatIdx) { hi=ii; hj=jj; break; } } if (hi>=0) break; }
                      const shape = (hi>=0) ? [{type:'rect', xref:'x', yref:'y', x0:hj-0.5, x1:hj+0.5, y0:hi-0.5, y1:hi+0.5, line:{width:4, color:'#3b82f6'}, fillcolor:'rgba(59,130,246,0.12)'}] : [];
                      Plotly.relayout(rightPlot, {shapes: shape});
                      if (hi>=0) { setTimeout(()=>{ Plotly.relayout(rightPlot, {shapes: [{type:'rect', xref:'x', yref:'y', x0:hj-0.5, x1:hj+0.5, y0:hi-0.5, y1:hi+0.5, line:{width:2, color:'#3b82f6'}, fillcolor:'rgba(59,130,246,0.12)'}]}); }, 180); }
                  }
              } catch(e) {}
          }
      });

      plot.on('plotly_unhover', () => {
          const targetFacet = parseInt(plot.dataset.facet || '0', 10);
          if (prefix === 'left') {
              const rightPlots = (window.plotGrid && window.plotGrid.right) ? window.plotGrid.right : [];
              const rightPlot = rightPlots[targetFacet];
              if (rightPlot) Plotly.relayout(rightPlot, {shapes: []});
          } else if (prefix === 'right') {
              const leftPlots = (window.plotGrid && window.plotGrid.left) ? window.plotGrid.left : [];
              const leftPlot = leftPlots[targetFacet];
              if (leftPlot) Plotly.relayout(leftPlot, {shapes: []});
          }
      });

      plot.on('plotly_selected', (ev) => {
          const idSet = new Set((ev && ev.points) ? ev.points.map(p=>p.customdata).filter(id => id >= 0) : []);
          const otherPrefix = (prefix === 'left') ? 'right' : 'left';
          document.querySelectorAll(`#${otherPrefix}Grid .plotcell > div`).forEach(div => {
              if (!div.__plot_data) return;
              const { mappings } = div.__plot_data;
              const xs=[], ys=[];
              for(let i=0; i<mappings.length; i++) for(let j=0; j<mappings[i].length; j++) if(idSet.has(mappings[i][j])){ xs.push(j); ys.push(i); }
              const highlight = { type: 'scatter', x:xs, y:ys, mode:'markers', marker: {color: 'rgba(0,0,0,0)', size: 12, line: {color: 'yellow', width: 3}}, hoverinfo:'skip' };
              Plotly.react(div, [div.data[0], highlight], div.__layout);
          });
      });

      if (document.getElementById('pairSel').value === 'pair' && document.getElementById('diffMode').checked && prefix === 'left') {
          drawDiffHighlights(plot, states[CTX.rightStep].diff_info, st, axA);
      }
    }
  }

  // --- 3D VIEW ---
  function render3D() {
    const step = parseInt(document.getElementById('step3d').value, 10);
    const st = states[step];
    const axX = parseInt(document.getElementById('axX').value, 10);
    const axY = parseInt(document.getElementById('axY').value, 10);
    const axZ = parseInt(document.getElementById('axZ').value, 10);
    const colorAxis = parseInt(document.getElementById('colorAxis3d').value, 10);
    const style = document.getElementById('style3d').value;
    const slices = window['slices3d'] || {};
    
    const { panel: panelColor, fg: fgColor } = { 
        panel: getComputedStyle(document.body).getPropertyValue('--panel').trim(),
        fg: getComputedStyle(document.body).getPropertyValue('--fg').trim() 
    };
    const xs=[], ys=[], zs=[], cs=[], texts=[], flatIndices=[];
    const idx = new Array(st.shape.length).fill(0);

    const validAxes = [axX, axY, axZ].every(ax => ax < st.shape.length);
    if (!validAxes) {
        document.getElementById('plot3d').innerHTML = '<p style="padding:20px;">Not enough dimensions for 3D view.</p>';
        return;
    }

    const iterShape = [st.shape[axX], st.shape[axY], st.shape[axZ]];
    for (let i = 0; i < iterShape[0]; i++) {
        for (let j = 0; j < iterShape[1]; j++) {
            for (let k = 0; k < iterShape[2]; k++) {
                for(let d=0; d<st.shape.length; d++) {
                    idx[d] = (d===axX)?i : (d===axY)?j : (d===axZ)?k : (slices[d]||0);
                }
                const flatIdx = getElementData(st.mapping, idx);
                if (flatIdx < 0) continue;

                const currentCoords = st.flat_ids_to_current_coords[flatIdx];
                const originalCoords = states[0].flat_ids_to_current_coords[flatIdx];
                if (!currentCoords || !originalCoords) continue;

                xs.push(i); ys.push(j); zs.push(k);
                flatIndices.push(flatIdx);
                
                const scalar = originalCoords[colorAxis] / (states[0].original_shape[colorAxis] -1 || 1);
                cs.push(scalar);
                
                const absIndex = getAbsIndex(currentCoords, st.shape);
                const originalAbsIndex = flatIdx;
                
                texts.push(
                    `<b>Abs. Index:</b> ${absIndex}<br>` +
                    `<b>Index Vector:</b> ${formatIndexVector(st.axis_names, currentCoords)}<br>` +
                    `<hr style="margin: 2px 0; border-style: dashed;">` +
                    `<b>Original ID:</b> ${originalAbsIndex}<br>` +
                    `<b>Original Vector:</b> ${formatIndexVector(states[0].axis_names, originalCoords)}` +
                    `<extra></extra>`
                );
            }
        }
    }

    let traces = [];
    if (style === 'points' || xs.length > 8000) {
        const hoverOn = (document.getElementById('hover3d') ? document.getElementById('hover3d').checked : true);
        traces.push({ type: 'scatter3d', x:xs, y:ys, z:zs, mode:'markers',
            marker: { color: cs, colorscale: TURBO_SCALE, cmin:0, cmax:1, size: 3 },
            hovertext: texts, hovertemplate: hoverOn ? '%{hovertext}<extra></extra>' : undefined, hoverinfo: hoverOn ? 'text' : 'none' });
    } else { // voxels (mesh3d)
        const vX=[], vY=[], vZ=[], i=[], j=[], k=[], vertexcolor=[];
        let vtx = 0;
        const gap = 0.1; // Gap for voxel borders
        for(let n=0; n<xs.length; n++) {
            const x0=xs[n]+gap/2, x1=xs[n]+1-gap/2, y0=ys[n]+gap/2, y1=ys[n]+1-gap/2, z0=zs[n]+gap/2, z1=zs[n]+1-gap/2;
            vX.push(x0,x1,x1,x0,x0,x1,x1,x0); vY.push(y0,y0,y1,y1,y0,y0,y1,y1); vZ.push(z0,z0,z0,z0,z1,z1,z1,z1);
            const c = turboRGB(cs[n]);
            for(let v=0;v<8;v++) vertexcolor.push(c);
            const faces = [[0,1,2,3],[4,5,6,7],[0,4,5,1],[1,5,6,2],[2,6,7,3],[3,7,4,0]];
            for(const f of faces) { i.push(vtx+f[0], vtx+f[0]); j.push(vtx+f[1], vtx+f[2]); k.push(vtx+f[2], vtx+f[3]); }
            vtx += 8;
        }
        traces.push({ type: 'mesh3d', x:vX, y:vY, z:vZ, i:i, j:j, k:k, vertexcolor, flatshading:true, hoverinfo:'none' });
        // Add invisible trace for hover tooltips on voxels
        traces.push({ type: 'scatter3d', x: xs.map(v=>v+0.5), y: ys.map(v=>v+0.5), z: zs.map(v=>v+0.5), mode: 'markers', marker: {size: 2, opacity: 0}, text: texts, hovertemplate: '%{text}' });
    }

    const layout = { margin:{l:0,r:0,t:20,b:0}, paper_bgcolor: panelColor, font: { color: fgColor },
        scene: { 
            xaxis:{title:st.axis_names[axX], gridcolor:fgColor, zerolinecolor:fgColor, linecolor:fgColor}, 
            yaxis:{title:st.axis_names[axY], gridcolor:fgColor, zerolinecolor:fgColor, linecolor:fgColor}, 
            zaxis:{title:st.axis_names[axZ], gridcolor:fgColor, zerolinecolor:fgColor, linecolor:fgColor}
        } 
    };
    Plotly.react(document.getElementById('plot3d'), traces, layout, {displaylogo:false, responsive:true});
  }

  function setup3D() {
      const step = parseInt(document.getElementById('stepRange').value, 10);
      const st = states[Math.min(step + 1, N-1)];
      const names = st.axis_names;

      const sel3d = document.getElementById('step3d');
      sel3d.innerHTML='';
      states.forEach((s,i) => sel3d.add(new Option(`Step ${i}: ${s.name}`, i)));
      sel3d.value = String(st.op_type === 'Input' ? 0 : states.indexOf(st));

      const axisSels = [document.getElementById('axX'), document.getElementById('axY'), document.getElementById('axZ')];
      axisSels.forEach(sel => { sel.innerHTML = ''; names.forEach((n,i) => sel.add(new Option(n,i))); });
      if (names.length >= 3) { axisSels[0].value=0; axisSels[1].value=1; axisSels[2].value=2; }
      else if (names.length === 2) { axisSels[0].value=0; axisSels[1].value=1; axisSels[2].value=1; }
      else if (names.length === 1) { axisSels[0].value=0; axisSels[1].value=0; axisSels[2].value=0; }
      
      const csel = document.getElementById('colorAxis3d'); csel.innerHTML='';
      states[0].axis_names.forEach((n,i) => csel.add(new Option(n,i)));
      csel.value = String(Math.min(states[0].axis_names.length-1, 2)); // Default to a sensible axis
      
      const buildSlicers3D = () => {
          const axesToExclude = axisSels.map(s => parseInt(s.value,10));
          const box=document.getElementById('slicers3d'); box.innerHTML=''; window['slices3d']={};
          for(let d=0; d<names.length; d++){
              if(axesToExclude.includes(d)) continue;
              const wrap=document.createElement('label'); wrap.style.color = AXIS_COLORS[d % AXIS_COLORS.length];
              wrap.innerHTML = `${names[d]}: <input data-ax="${d}" type="range" min="0" max="${Math.max(0,st.shape[d]-1)}" value="0" step="1"/> <span class="badge">0</span>`;
              const input=wrap.querySelector('input'), lab=wrap.querySelector('span');
              input.addEventListener('input', (e) => { const v = parseInt(e.target.value, 10); lab.textContent = v; window['slices3d'][d] = v; render3D(); });
              box.appendChild(wrap); window['slices3d'][d]=0;
          }
      };
      
      buildSlicers3D();
      [...axisSels, sel3d, csel, document.getElementById('style3d')].forEach(el => el.addEventListener('change', () => { buildSlicers3D(); render3D(); }));
      render3D();
  }
  
  function populateSelectors(prefix, names, selectedValues = {}){
    const sels = { A: document.getElementById(prefix+'AxA'), B: document.getElementById(prefix+'AxB'), F: document.getElementById(prefix+'Facet'), C: document.getElementById(prefix+'ColorAxis') };
    Object.values(sels).forEach(s=>s.innerHTML='');
    
    names.forEach((n,i) => { [sels.A, sels.B, sels.F].forEach(s => s.add(new Option(n,i))); });
    states[0].axis_names.forEach((n,i) => sels.C.add(new Option(n,i))); // Color by original axes
    sels.F.add(new Option('(none)', -1));
    
    sels.A.value = selectedValues.A ?? String(Math.max(0, names.length-2));
    sels.B.value = selectedValues.B ?? String(Math.max(0, names.length-1));
    sels.F.value = selectedValues.F ?? "-1";
    sels.C.value = selectedValues.C ?? String(Math.max(0, states[0].axis_names.length-1));
  }
  
  function setupSide(prefix, step) {
      const st = states[step];
      const names = st.axis_names;
      const getAxes = () => ({A:parseInt(sels.A.value,10),B:parseInt(sels.B.value,10),F:parseInt(sels.F.value,10)});
      
      populateSelectors(prefix, names);
      const sels = { A: document.getElementById(prefix+'AxA'), B: document.getElementById(prefix+'AxB'), F: document.getElementById(prefix+'Facet'), C: document.getElementById(prefix+'ColorAxis') };

      const buildSlicers = () => {
          const {A,B,F} = getAxes();
          const box=document.getElementById(prefix+'Slicers'); box.innerHTML=''; window[prefix+'Slices']={};
          for(let d=0; d<names.length; d++){
              if([A,B,F].includes(d)) continue;
              const wrap=document.createElement('label'); wrap.style.color = AXIS_COLORS[d % AXIS_COLORS.length];
              wrap.innerHTML = `${names[d]}: <input data-ax="${d}" type="range" min="0" max="${Math.max(0,st.shape[d]-1)}" value="0" step="1"/> <span class="badge">0</span>`;
              const input=wrap.querySelector('input'), lab=wrap.querySelector('span');
              input.addEventListener('input', (e) => { const v=parseInt(e.target.value,10); lab.textContent=v; window[prefix+'Slices'][d]=v; redraw(prefix); });
              box.appendChild(wrap); window[prefix+'Slices'][d]=0;
          }
      };
      
      Object.values(sels).forEach(s => s.addEventListener('change', () => { buildSlicers(); redraw(prefix); }));
      buildSlicers();
  }

  function redraw(prefix){
    const st = (prefix==='left') ? states[CTX.leftStep] : states[CTX.rightStep];
    const sel = { A: document.getElementById(prefix+'AxA'), B: document.getElementById(prefix+'AxB'), F: document.getElementById(prefix+'Facet'), C: document.getElementById(prefix+'ColorAxis') };
    renderGrid(document.getElementById(prefix+'Grid'), st, parseInt(sel.A.value,10), parseInt(sel.B.value,10), parseInt(sel.F.value,10), window[prefix+'Slices']||{}, prefix, parseInt(sel.C.value,10));
  }
  
  function refresh(){
    const mode = document.getElementById('pairSel').value;
    const stepSel = parseInt(document.getElementById('stepRange').value, 10);
    if (mode === 'pair')      { CTX.leftStep = stepSel; CTX.rightStep = Math.min(N-1, stepSel+1); }
    else if (mode === 'same') { CTX.leftStep = stepSel; CTX.rightStep = stepSel; }
    else /* input */          { CTX.leftStep = 0;       CTX.rightStep = Math.min(N-1, stepSel+1); }

    const lbl = (mode === 'pair') ? `${CTX.leftStep} → ${CTX.rightStep}`
              : (mode === 'same') ? `Step ${stepSel}`
              : `0 → ${CTX.rightStep}`;
    document.getElementById('stepLabel').textContent = lbl;

    for (let i = 0; i < N; i++) document.getElementById('t'+i).classList.toggle('active', i === stepSel);

    document.getElementById('leftTitle').textContent  = `Left: Step ${CTX.leftStep} (${states[CTX.leftStep].name})`;
    document.getElementById('rightTitle').textContent = `Right: Step ${CTX.rightStep} (${states[CTX.rightStep].name})`;

    setupSide('left', CTX.leftStep);
    setupSide('right', CTX.rightStep);
    redraw('left'); redraw('right'); setup3D();
}

  const exportImage = (format, isGrid) => {
    const panels = document.querySelectorAll('#rightGrid .plotcell > div:last-child');
    panels.forEach((p, i) => Plotly.downloadImage(p, {format: format, filename: `tensorviz_step${CTX.rightStep}_${i}`}));
  };
  document.getElementById('btnExportPNG').addEventListener('click', () => exportImage('png'));
  document.getElementById('btnExportSVG').addEventListener('click', () => exportImage('svg'));
  document.getElementById('btnExportGridPNGs').addEventListener('click', () => exportImage('png', true));
  document.getElementById('btnExportJSON').addEventListener('click', ()=>{
      const blob=new Blob([JSON.stringify({states:DATA.states},null,2)],{type:'application/json'});
      const url=URL.createObjectURL(blob); a=document.createElement('a'); a.href=url; a.download='tensorviz_data.json'; a.click();
      setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  
  // WIRING & BOOT
  document.getElementById('themeSel').addEventListener('change', (e) => { document.body.className = e.target.value; refresh(); });
  ['pairSel', 'diffMode', 'linkZoom', 'hoverRef'].forEach(id => document.getElementById(id).addEventListener('change', refresh));
  document.getElementById('hover3d').addEventListener('change', () => render3D());
  document.getElementById('stepRange').addEventListener('input', refresh);
  document.getElementById('btnReset').addEventListener('click', () => { localStorage.removeItem(STORAGE_KEY); location.reload(); });
  document.getElementById('resetCam').addEventListener('click', () => Plotly.relayout(document.getElementById('plot3d'), {'scene.camera':{}}));

  document.body.className = localStorage.getItem(STORAGE_KEY) ? JSON.parse(localStorage.getItem(STORAGE_KEY)).theme || 'light' : 'light';
  document.getElementById('themeSel').value = document.body.className;
  
  (function mkTimeline(){
    const tl=document.getElementById('timeline');
    for(let i=0;i<N;i++){
      const b=document.createElement('button'); b.className='titem'; b.id='t'+i;
      b.textContent=`${i}: ${states[i].name}`;
      b.addEventListener('click',()=>{ document.getElementById('stepRange').value=i; refresh(); });
      tl.appendChild(b);
    }
  })();
  refresh();
})();
</script>
</body></html>
"""
    return (template
            .replace("%%PLOTLY%%", lib_js)
            .replace("%%MAXSTEP%%", max_step)
            .replace("%%DATAJSON%%", data_json)
            .replace("%%TITLE%%", page_title))

# =========================
# CLI
# =========================


# =========================
# XLSX Exporter
# =========================
def _c_order_iter(shape):
    # Iterate over all coordinates in C-order (last axis fastest)
    if not shape:
        yield ()
        return
    import itertools
    ranges = [range(s) for s in shape]
    for coords in itertools.product(*ranges):
        yield coords

def _abs_index_from_coords(coords, shape):
    # Row-major (C-order) absolute index
    if len(coords) != len(shape):
        return -1
    strides = [1]*len(shape)
    for i in range(len(shape)-2, -1, -1):
        strides[i] = strides[i+1]*shape[i+1]
    acc = 0
    for i,c in enumerate(coords):
        acc += c*strides[i]
    return int(acc)

def _format_vec(coords):
    return "(" + ",".join(str(int(x)) for x in coords) + ")"

def export_mapping_xlsx(states, xlsx_path):
    """Create a two-sheet workbook:
       Sheet 1: Input AbsIdx -> list of Output AbsIdx (final state)
       Sheet 2: Input VecIdx -> list of Output VecIdx (final state)
       Ordering of rows: input tensor in C-order (last axis fastest).
    """
    try:
        import pandas as pd
        from openpyxl.utils import get_column_letter
    except Exception as e:
        raise RuntimeError(f"Excel export requires pandas/openpyxl: {e}")
    if not states:
        raise ValueError("No states to export")
    input_state = states[0]
    final_state = states[-1]
    in_shape = tuple(input_state['shape'])
    out_shape = tuple(final_state['shape'])
    import numpy as _np
    mapping = _np.array(final_state['mapping'], dtype=_np.int64)
    # Build inverse: input_abs -> list of output coords
    inv = [[] for _ in range(int(_np.prod(in_shape)) if _np.prod(in_shape)>0 else 0)]
    for out_coords in _c_order_iter(out_shape):
        src = int(mapping[out_coords])
        if src >= 0:
            inv[src].append(out_coords)
    # Build AbsIdx sheet rows
    abs_rows = []
    for in_coords in _c_order_iter(in_shape):
        in_abs = _abs_index_from_coords(in_coords, in_shape)
        outs = inv[in_abs] if in_abs < len(inv) else []
        out_abs_list = [str(_abs_index_from_coords(c, out_shape)) for c in outs]
        abs_rows.append({'Input AbsIdx': in_abs, 'Output AbsIdx': ",".join(out_abs_list)})
    # Build VecIdx sheet rows
    vec_rows = []
    for in_coords in _c_order_iter(in_shape):
        outs = inv[_abs_index_from_coords(in_coords, in_shape)]
        out_vec_list = [ _format_vec(c) for c in outs ]
        vec_rows.append({'Input VecIdx': _format_vec(in_coords), 'Output VecIdx': ";".join(out_vec_list)})
    # Write
    df_abs = pd.DataFrame(abs_rows)
    df_vec = pd.DataFrame(vec_rows)
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df_abs.to_excel(writer, sheet_name='AbsIdx Mapping', index=False)
        df_vec.to_excel(writer, sheet_name='VecIdx Mapping', index=False)
        wb = writer.book
        for ws_name in ['AbsIdx Mapping', 'VecIdx Mapping']:
            ws = wb[ws_name]
            for col in ws.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
                max_len = min(max_len + 2, 80)
                ws.column_dimensions[col[0].column_letter].width = max(12, max_len)
def main():
    ap = argparse.ArgumentParser(
        description="Render a self-contained tensor mapping dashboard (single HTML).",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example usage:
  python %(prog)s --config my_config.json

The config file should follow the specified JSON format:
{
  "name": "Example Pipeline",
  "tensor": { "shape": [1, 2, 3, 4], "axes": ["N", "C", "H", "W"] },
  "operations": [
    { "type": "transpose", "name": "To NHWC", "perm": [0, 2, 3, 1] },
    { "type": "reshape", "name": "Flatten H and W", "newshape": [1, 12, 1] }
  ]
}
"""
    )
    ap.add_argument("--config", required=True, help="Path to JSON config file.")
    ap.add_argument("--out", default=".", help="Output directory (default: current directory).")
    ap.add_argument("--title", default=None, help="Optional HTML page title.")
    args = ap.parse_args()

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    try:
        cfg = load_config_bom_safe(cfg_path)
    except Exception as e:
        print(f"Failed to load or parse config file '{cfg_path}': {e}", file=sys.stderr)
        sys.exit(2)

    try:
        states = evaluate_pipeline(cfg)
    except Exception as e:
        print(f"Error while evaluating the tensor pipeline: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        print("\nPlease check your config file for correctness (e.g., shape consistency, valid permutations).", file=sys.stderr)
        sys.exit(3)

    html_out = render_dashboard_html(states, title=args.title)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tensorviz_{cfg_path.stem}.html"
    out_path.write_text(html_out, encoding="utf-8")
    print(f"Successfully generated dashboard: {out_path.resolve()}")

    # Also write XLSX mapping workbook (final-state mapping)
    xlsx_path = out_dir / f"tensorviz_{cfg_path.stem}_mapping.xlsx"
    export_mapping_xlsx(states, str(xlsx_path))
    print(f"Successfully generated mapping workbook: {xlsx_path.resolve()}")

def load_config_bom_safe(path: pathlib.Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text)

if __name__ == "__main__":
    main()


