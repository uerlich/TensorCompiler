from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, List
import numpy as np

from .core.constants import ATOM_B
from .core.io_utils import (read_workbook, emit_config_xlsx, emit_report_txt,
                            ensure_trace_xlsx, try_build_dummy_workbook)
from .core.bank_safety import choose_laneization
from .core.simulator import SimConfig, simulate
# --- FIX --- Removed unused and faulty 'write_animation' import
from .vis.visualize import render_visualization
from tensorviz_dashboard_unified_v21 import evaluate_pipeline

def compile_and_run(workbook_path: str,
                    pixel_bits: int,
                    zero_mode: str,
                    viz: str,
                    gap_budget_B: Optional[int],
                    final_mapping: Optional[np.ndarray] = None,
                    final_shape: Optional[tuple] = None,
                    input_axes: Optional[List[str]] = None):
    """
    Executes the full compilation and simulation pipeline.
    """
    out_dir = Path(workbook_path).with_suffix("").parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(workbook_path).stem

    if not Path(workbook_path).exists():
        workbook_path = try_build_dummy_workbook(Path(out_dir) / f"{base_name}_auto.xlsx", pixel_bits)

    mapping_spec = read_workbook(workbook_path, pixel_bits=pixel_bits)

    if final_mapping is None or final_shape is None:
        mock_op_config = {
            "tensor": {"shape": mapping_spec.input_shape, "axes": [d[0] for d in mapping_spec.dims]},
            "operations": []
        }
        states = evaluate_pipeline(mock_op_config)
        final_state = states[-1]
        final_mapping = np.array(final_state['mapping'])
        final_shape = final_state['shape']

    if input_axes is None:
        input_axes = [d[0] for d in mapping_spec.dims]

    gap_budget_atoms = None if gap_budget_B is None else gap_budget_B // ATOM_B  # kept for backward compat (unused)
    gap_budget_B_int = None if gap_budget_B is None else int(gap_budget_B)
    choice = choose_laneization(mapping_spec.dims, gap_budget_B=gap_budget_B_int)

    config_path = out_dir / f"{base_name}_config.xlsx"
    report_path = out_dir / f"{base_name}_report.txt"
    emit_config_xlsx(config_path, choice, pixel_bits, zero_mode)
    emit_report_txt(report_path, choice, mapping_spec.dims, gap_budget_B)

    sim_cfg = SimConfig(beta_eff_B=choice.beta_eff_B,
                        lifts_B=[l*ATOM_B for l in choice.lifts],
                        zero_mode=zero_mode,
                        pixel_B=pixel_bits // 8)
                        
    sim_outputs = simulate(mapping_spec, choice, sim_cfg, out_prefix=str(out_dir / base_name),
                           final_mapping=final_mapping, final_shape=final_shape, input_axes=input_axes)
    if sim_outputs is None:
        raise RuntimeError("Simulation returned no outputs. Please verify the workbook contains a valid workload.")

    trace_xlsx_path = ensure_trace_xlsx(sim_outputs.trace_path)

    viz_path = out_dir / f"{base_name}_viz.gif"
    render_visualization(
        out_path=str(viz_path),
        frames=sim_outputs.frames,
        title=f"Simulation: {base_name}",
        tensor_info=sim_outputs.tensor_info
    )

    if viz.lower() == 'mp4':
        try:
            import imageio
            reader = imageio.get_reader(str(viz_path))
            mp4_path = out_dir / f"{base_name}_viz.mp4"
            writer = imageio.get_writer(str(mp4_path), fps=10, codec='libx264', format='FFMPEG')
            for frame in reader:
                writer.append_data(frame)
            writer.close()
            reader.close()
            viz_path = mp4_path
        except Exception as e:
            print("MP4 transcode failed, leaving GIF:", e)

    print(f"✓ Compilation and simulation successful. Outputs are in: {out_dir}")
    return {"config": str(config_path), "report": str(report_path), "viz": str(viz_path), "trace": str(trace_xlsx_path)}

def run_cli():
    p = argparse.ArgumentParser(description="Tensor Offline Compiler & Accelerator Simulator (Spec V0.2 compliant)")
    p.add_argument("--workbook", required=True, help="Path to input workbook (.xlsx)")
    p.add_argument("--pixel-bits", type=int, required=True, help="Element size (bits) of a pixel/element (e.g., 8/16/24/32)" )
    p.add_argument("--zero-mode", choices=["idle_lanes","zero_pad"], default="idle_lanes", help="How to treat empty lanes in commits")
    p.add_argument("--viz", choices=["gif","mp4"], default="gif", help="Visualization output format")
    p.add_argument("--gap_budget_B", type=int, default=None, help="Optional total gap budget (bytes) for bank-safety optimization")
    args = p.parse_args()
    compile_and_run(args.workbook, args.pixel_bits, args.zero_mode, args.viz, args.gap_budget_B)

def run_gui():
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox

    root = tk.Tk()
    root.title("Tensor Offline Compiler & Simulator")

    workbook_var = tk.StringVar()
    pixel_bits_var = tk.StringVar(value="16")
    zero_mode_var = tk.StringVar(value="idle_lanes")
    viz_var = tk.StringVar(value="gif")
    gap_var = tk.StringVar(value="")

    def browse():
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx"), ("All", "*.*")])
        if path:
            workbook_var.set(path)

    def run():
        try:
            gap = int(gap_var.get()) if gap_var.get().strip() else None
            outputs = compile_and_run(workbook_var.get(), int(pixel_bits_var.get()), zero_mode_var.get(), viz_var.get(), gap)
            messagebox.showinfo("Done", f"Generated:\n{outputs['config']}\n{outputs['report']}\n{outputs['trace']}\n{outputs['viz']}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill='both', expand=True)

    ttk.Label(frm, text="Workbook").grid(row=0, column=0, sticky='e'); ttk.Entry(frm, textvariable=workbook_var, width=40).grid(row=0, column=1, sticky='we'); ttk.Button(frm, text="Browse", command=browse).grid(row=0, column=2)
    ttk.Label(frm, text="Pixel Bits").grid(row=1, column=0, sticky='e'); ttk.Entry(frm, textvariable=pixel_bits_var, width=10).grid(row=1, column=1, sticky='w')
    ttk.Label(frm, text="Zero Mode").grid(row=2, column=0, sticky='e'); ttk.OptionMenu(frm, zero_mode_var, zero_mode_var.get(), "idle_lanes", "zero_pad").grid(row=2, column=1, sticky='w')
    ttk.Label(frm, text="Visualization").grid(row=3, column=0, sticky='e'); ttk.OptionMenu(frm, viz_var, viz_var.get(), "gif", "mp4").grid(row=3, column=1, sticky='w')
    ttk.Label(frm, text="Gap Budget (bytes)").grid(row=4, column=0, sticky='e'); ttk.Entry(frm, textvariable=gap_var, width=12).grid(row=4, column=1, sticky='w')
    ttk.Button(frm, text="Run", command=run).grid(row=5, column=1, sticky='w', pady=6)

    root.mainloop()

if __name__ == '__main__':
    run_gui()