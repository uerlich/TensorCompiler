from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import imageio
import io

def _fig_to_ndarray(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi, bbox_inches='tight')
    buf.seek(0)
    arr = imageio.v2.imread(buf)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr

from ..core.constants import LANES

def get_tensor_color_dims(shape: Tuple[int, ...], axes: Optional[List[str]] = None) -> Tuple[Optional[int], Optional[int]]:
    if axes and len(axes) == len(shape):
        try: color_dim = axes.index('C') if 'C' in axes else None
        except ValueError: color_dim = None
        try: tone_dim = axes.index('H') if 'H' in axes else None
        except ValueError: tone_dim = None
        if color_dim is not None and tone_dim is not None: return color_dim, tone_dim
        if color_dim is not None:
            other_dims = [i for i, (s, a) in enumerate(zip(shape, axes)) if s > 1 and a != 'C']
            if other_dims: tone_dim = other_dims[0]
            return color_dim, tone_dim
        if tone_dim is not None:
            other_dims = [i for i, (s, a) in enumerate(zip(shape, axes)) if s > 1 and a != 'H']
            if other_dims: color_dim = other_dims[0]
            return color_dim, tone_dim
    non_one_dims = [i for i, s in enumerate(shape) if s > 1]
    color_dim = non_one_dims[0] if len(non_one_dims) > 0 else None
    tone_dim = non_one_dims[1] if len(non_one_dims) > 1 else None
    return color_dim, tone_dim

def generate_color_map(num_colors: int, num_tones: int) -> List[List[Tuple[float, float, float]]]:
    num_colors = max(1, num_colors)
    num_tones = max(1, num_tones)
    base_colors = []

    try:
        from skimage.color import rgb2lab, deltaE_ciede2000
        # ... (This is the enhanced path that is currently failing silently) ...
        # [The original advanced color selection logic remains here]
        cmaps = [plt.get_cmap(name) for name in ['tab20', 'tab20b', 'Set3']]
        candidate_rgb = np.unique([c for cmap in cmaps for c in cmap.colors], axis=0)
        candidate_lab = rgb2lab(candidate_rgb)
        filtered_candidates = []
        for i in range(candidate_lab.shape[0]):
            l, a, b = candidate_lab[i]
            chroma = np.sqrt(a**2 + b**2)
            if l > 25 and l < 90 and chroma > 25:
                filtered_candidates.append({'rgb': candidate_rgb[i], 'lab': candidate_lab[i]})
        if not filtered_candidates: raise ImportError("Color filtering resulted in an empty candidate pool.")
        if num_colors <= len(filtered_candidates):
            selected_indices = [0] 
            selected_labs = [filtered_candidates[0]['lab']]
            candidate_indices = list(range(1, len(filtered_candidates)))
            for _ in range(num_colors - 1):
                max_min_dist = -1; best_candidate_idx = -1
                for cand_idx in candidate_indices:
                    min_dist_to_selected = float('inf')
                    for sel_lab in selected_labs:
                        dist = deltaE_ciede2000(filtered_candidates[cand_idx]['lab'], sel_lab)
                        if dist < min_dist_to_selected: min_dist_to_selected = dist
                    if min_dist_to_selected > max_min_dist:
                        max_min_dist = min_dist_to_selected; best_candidate_idx = cand_idx
                selected_indices.append(best_candidate_idx)
                selected_labs.append(filtered_candidates[best_candidate_idx]['lab'])
                candidate_indices.remove(best_candidate_idx)
            base_colors = [tuple(filtered_candidates[i]['rgb']) for i in selected_indices]
        else:
            base_colors = [tuple(c['rgb']) for c in filtered_candidates]

    except ImportError as e:
        # --- DIAGNOSTIC MODIFICATION START ---
        import traceback
        print("--- DETAILED IMPORT ERROR ---")
        print(f"An ImportError occurred while trying to import from scikit-image.")
        print(f"This is the specific error message: {e}")
        print("Full traceback:")
        traceback.print_exc()
        print("--- END OF DETAILED ERROR ---")
        print("\nWarning: Using fallback color scheme due to the error above. Please provide the detailed error for diagnosis.")
        # --- DIAGNOSTIC MODIFICATION END ---
        
        # Fallback logic
        tab20 = plt.get_cmap('tab20')
        colors1 = [tab20(i) for i in range(20)]
        base_colors = [colors1[i] for i in range(num_colors)]

    color_map = []
    for i in range(num_colors):
        base_color = base_colors[i % len(base_colors)]
        tones = []
        h, s, v = mcolors.rgb_to_hsv(base_color[:3])
        for j in range(num_tones):
            new_v = 0.6 + 0.4 * (j / (num_tones - 1)) if num_tones > 1 else 1.0
            tones.append(mcolors.hsv_to_rgb((h, s, new_v)))
        color_map.append(tones)
    return color_map

def _make_dim_colors(shape, axes):
    color_dim, tone_dim = get_tensor_color_dims(shape, axes)
    num_colors = shape[color_dim] if color_dim is not None and color_dim < len(shape) else 1
    num_tones = shape[tone_dim] if tone_dim is not None and tone_dim < len(shape) else 1
    cmap = generate_color_map(num_colors, num_tones)
    size = int(np.prod(shape)) if shape else 0
    colors = np.full((size, 3), 0.85, dtype=float)
    if size > 0:
        coords = np.unravel_index(np.arange(size), shape)
        color_idx = coords[color_dim] if color_dim is not None and color_dim < len(shape) else np.zeros(size, dtype=int)
        tone_idx  = coords[tone_dim]  if tone_dim  is not None and tone_dim < len(shape) else np.zeros(size, dtype=int)
        for i in range(size):
            colors[i] = cmap[color_idx[i] % num_colors][tone_idx[i] % num_tones]
    return colors

def draw_element_memory_panel(ax: plt.Axes, title: str, element_colors: np.ndarray, panel_width_elems: int, read_mask_elems: Optional[np.ndarray] = None, hatch_mask_elems: Optional[np.ndarray] = None):
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#e0e0e0"); ax.invert_yaxis()
    num_elems = element_colors.shape[0]
    if num_elems == 0 or panel_width_elems <= 0: return
    panel_height_elems = int(np.ceil(num_elems / panel_width_elems))
    if panel_height_elems == 0: return
    ax.set_xlim(-0.5, panel_width_elems - 0.5); ax.set_ylim(panel_height_elems - 0.5, -0.5)
    element_patches = [Rectangle((i % panel_width_elems - 0.5, i // panel_width_elems - 0.5), 1, 1) for i in range(num_elems)]
    ax.add_collection(PatchCollection(element_patches, facecolor=element_colors, edgecolor='none'))
    if read_mask_elems is not None:
        read_indices = np.where(read_mask_elems)[0]
        if read_indices.size > 0:
            r, c = np.unravel_index(read_indices, (panel_height_elems, panel_width_elems))
            border_patches = [Rectangle((ci - 0.5, ri - 0.5), 1, 1) for ri, ci in zip(r, c)]
            ax.add_collection(PatchCollection(border_patches, facecolor='none', edgecolor='black', linewidth=1.0))
    if hatch_mask_elems is not None:
        hatch_indices = np.where(hatch_mask_elems)[0]
        if hatch_indices.size > 0:
            r, c = np.unravel_index(hatch_indices, (panel_height_elems, panel_width_elems))
            hatch_patches = [Rectangle((ci - 0.5, ri - 0.5), 1, 1) for ri, ci in zip(r, c)]
            ax.add_collection(PatchCollection(hatch_patches, facecolor='none', edgecolor='gray', hatch='///'))

def draw_buffer_panel(ax: plt.Axes, title: str, buffer_atoms: List[int], lane_colors: List):
    cols, rows = 32, 4
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    colors = np.ones((rows, cols, 4))
    for i, lane_idx in enumerate(buffer_atoms):
        if lane_idx != -1: colors[i // cols, i % cols] = lane_colors[int(lane_idx) % LANES]
    ax.imshow(colors, interpolation='nearest', aspect='auto'); ax.grid(which='both', color='lightgray', linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True); ax.set_yticks(np.arange(-.5, rows, 1), minor=True); ax.tick_params(which='minor', size=0)

def render_visualization(out_path: str, frames: List[Dict[str, Any]], title: str, tensor_info: Dict[str, Any]):
    if not frames: print("No frames to render."); return
    in_shape = tensor_info.get("input_shape"); in_axes = tensor_info.get("input_axes", [])
    total_input_elements = tensor_info.get('total_input_elements', 0)
    input_tensor_colors = _make_dim_colors(in_shape, in_axes)
    input_panel_width = int(np.prod(in_shape[1:])) if in_shape and len(in_shape) > 1 else 64
    out_shape = tensor_info.get("output_shape")
    total_output_elements = tensor_info.get('total_output_elements', 0)
    output_panel_width = int(np.prod(out_shape[1:])) if out_shape and len(out_shape) > 1 else 64
    lane_color_palette = [plt.get_cmap('turbo')(i/LANES) for i in range(LANES)]
    images = []
    num_frames = len(frames)
    print(f"Starting visualization rendering for {num_frames} frames...")
    for i, f in enumerate(frames):
        if (i + 1) % 20 == 0 or i == num_frames - 1 or i == 0: print(f"  - Rendering frame {i + 1}/{num_frames}...")
        fig = plt.figure(figsize=(12, 8), dpi=100)
        ax_in = fig.add_axes([0.05, 0.65, 0.55, 0.28])
        draw_element_memory_panel(ax_in, "L2 Source Memory (Input Tensor)", input_tensor_colors, input_panel_width, read_mask_elems=f["input_mem_read_mask"])
        ax_bA = fig.add_axes([0.05, 0.38, 0.27, 0.22])
        draw_buffer_panel(ax_bA, f"Buffer A (2048B) — {f['roleA']}", f["bufA_atoms"], lane_color_palette)
        ax_bB = fig.add_axes([0.33, 0.38, 0.27, 0.22])
        draw_buffer_panel(ax_bB, f"Buffer B (2048B) — {f['roleB']}", f["bufB_atoms"], lane_color_palette)
        ax_out = fig.add_axes([0.05, 0.05, 0.55, 0.28])
        output_mem_source_indices = f["output_mem_content"]
        output_element_colors = np.full((total_output_elements, 3), 0.85, dtype=float)
        valid_mask = output_mem_source_indices >= 0
        written_indices = np.where(valid_mask)[0]
        if written_indices.size > 0:
            src_indices = output_mem_source_indices[written_indices]
            output_element_colors[written_indices] = input_tensor_colors[src_indices]
        draw_element_memory_panel(ax_out, "L1 Destination Memory (Output Tensor)", output_element_colors, output_panel_width, hatch_mask_elems=~valid_mask)
        ax_info = fig.add_axes([0.65, 0.78, 0.32, 0.15]); ax_info.axis("off")
        ax_info.text(0.0, 0.95, title, fontsize=11, weight="bold", va='top')
        ax_info.text(0.0, 0.60, f"Cycle {f['cycle']}", fontsize=10, va='top')
        ax_info.text(0.0, 0.35, f"Δb={f.get('delta_b',0)} | Σ gaps(B)={sum(f.get('lifts_atoms',[]))*16}", fontsize=10, va='top')
        ax_grid = fig.add_axes([0.65, 0.58, 0.32, 0.18])
        ax_grid.set_title("Bank grid (last commit)", fontsize=9); ax_grid.set_xticks([]); ax_grid.set_yticks([])
        banks = f.get('banks', [])
        for idx in range(LANES):
            r, c = divmod(idx, 4)
            rect = plt.Rectangle((c/4, 1-(r+1)/4), 1/4, 1/4, facecolor=lane_color_palette[idx], edgecolor='white', lw=1)
            ax_grid.add_patch(rect); bank = banks[idx] if idx < len(banks) and banks[idx] is not None else '–'
            ax_grid.text(c/4+0.125, 1-(r+0.5)/4, f"L{idx}\nB{bank}", fontsize=7, color="black", ha='center', va='center')
        ax_l2_util = fig.add_axes([0.65, 0.48, 0.32, 0.05]) 
        ax_l2_util.set_title("L2 Read Bus Utilization", fontsize=9, y=0.9)
        ax_l2_util.set_xticks([]); ax_l2_util.set_yticks([]); ax_l2_util.set_xlim(0, 1)
        l2_util = f.get('l2_bus_utilization', 0.0)
        ax_l2_util.barh([0], [l2_util], color='seagreen' if l2_util > 0 else 'gray', height=0.5)
        if l2_util > 0: ax_l2_util.text(l2_util / 2, 0, f"L2: {l2_util:.1%}", ha='center', va='center', color='white', fontsize=8, weight='bold')
        ax_l1_util = fig.add_axes([0.65, 0.39, 0.32, 0.05])
        ax_l1_util.set_title("L1 Commit Bus Utilization", fontsize=9, y=0.9)
        ax_l1_util.set_xticks([]); ax_l1_util.set_yticks([]); ax_l1_util.set_xlim(0, 1)
        l1_util = f.get('l1_bus_utilization', 0.0)
        ax_l1_util.barh([0], [l1_util], color='darkblue' if l1_util > 0 else 'gray', height=0.5)
        if l1_util > 0: ax_l1_util.text(l1_util / 2, 0, f"L1: {l1_util:.1%}", ha='center', va='center', color='white', fontsize=8, weight='bold')
        ax_leg = fig.add_axes([0.65, 0.05, 0.32, 0.30]); ax_leg.axis("off"); ax_leg.set_title("Lane Legend", fontsize=9, loc='left')
        for idx in range(LANES):
            r, c = divmod(idx, 2); y, x = 0.9 - r * 0.11, c * 0.5
            ax_leg.add_patch(plt.Rectangle((x, y-0.04), 0.08, 0.04, facecolor=lane_color_palette[idx]))
            ax_leg.text(x + 0.1, y - 0.035, f"Lane {idx}", fontsize=8, va='top')
        img = _fig_to_ndarray(fig)
        images.append(img)
        plt.close(fig)
    print(f"Rendering complete. Saving file to {out_path}...")
    imageio.mimsave(out_path, images, fps=5, macro_block_size=1)
    print("✓ Save complete.")