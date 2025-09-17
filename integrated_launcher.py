#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path

# --- PATH BOOTSTRAP: make local packages importable no matter the working dir ---
import sys as _sys
from pathlib import Path as _Path
_BASE = _Path(__file__).resolve().parent
_PKG_ROOT = _BASE / "tensor_offline_tool"
_PKG_PARENT = _PKG_ROOT.parent
for _p in (_BASE, _BASE / "enhancements", _PKG_PARENT):
    _ps = str(_p)
    if _ps not in _sys.path:
        _sys.path.insert(0, _ps)
# -------------------------------------------------------------------------------


import tkinter as tk

import importlib.util as _ilu
from pathlib import Path as _Path
_BASE = _Path(__file__).resolve().parent
_shim_file = _BASE / "enhancements" / "mpl_compat.py"
if _shim_file.exists():
    _spec = _ilu.spec_from_file_location("mpl_compat", str(_shim_file))
    _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    _mod.install_matplotlib_rgb_shim()

from tkinter import filedialog, ttk, messagebox

from tensor_offline_tool.tool.main import compile_and_run as _compile_and_run
from enhancements.enhanced_runner import enhanced_compile_and_run
import tensorviz_dashboard_unified_v21 as tvu21

def run_with_json_config(json_path: Path, out_dir: Path, page_title: str|None=None) -> Path:
    cfg = tvu21.load_config_bom_safe(json_path)
    states = tvu21.evaluate_pipeline(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = tvu21.render_dashboard_html(states, title=page_title or cfg.get("name") or f"TensorViz {json_path.stem}")
    html_path = out_dir / f"tensorviz_{json_path.stem}.html"
    html_path.write_text(html, encoding="utf-8")
    xlsx_path = out_dir / f"tensorviz_{json_path.stem}_mapping.xlsx"
    tvu21.export_mapping_xlsx(states, str(xlsx_path))
    return xlsx_path



def run_cli(argv: list[str] | None=None):
    p = argparse.ArgumentParser(description="Integrated launcher: tensor_offline_tool2 + tensorviz_dashboard_unified_v21")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--workbook", help="Excel mapping (.xlsx) for tensor_offline_tool2 (legacy path).")
    src.add_argument("--json-config", help="ONNX-ops JSON for tensorviz; mapping is auto-generated then compiled.")
    p.add_argument("--pixel-bits", type=int, default=8, choices=[8,16,32,64])
    p.add_argument("--zero-mode", default="idle_lanes", choices=["idle_lanes","zero_pad"])
    p.add_argument("--viz", default="gif", choices=["gif","mp4"])
    p.add_argument("--gap_budget_B", type=int, default=None)
    p.add_argument("--out", default=".")
    p.add_argument("--title", default=None)
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.json_config:
        json_path = Path(args.json_config)
        run_with_json_config(json_path, out_dir, page_title=args.title)
        outs = enhanced_compile_and_run(str(json_path), str(out_dir), args.pixel_bits, args.zero_mode, args.viz, args.gap_budget_B)
    else:
        outs = _compile_and_run(args.workbook, args.pixel_bits, args.zero_mode, args.viz, args.gap_budget_B)
    print("OK")
    return outs



def run_gui():
    root = tk.Tk()
    root.title("Tensor Offline Tool 2 + TensorViz (Unified)")

    src_mode = tk.StringVar(value="json")
    path_var = tk.StringVar(value="")
    out_var  = tk.StringVar(value=str(Path.cwd()))
    pixel_var = tk.IntVar(value=8)
    zero_var = tk.StringVar(value="idle_lanes")
    viz_var  = tk.StringVar(value="gif")
    gap_var  = tk.StringVar(value="")
    title_var = tk.StringVar(value="")

    def choose_src_file():
        if src_mode.get() == "xlsx":
            f = filedialog.askopenfilename(filetypes=[("Excel workbook","*.xlsx"), ("All files","*.*")])
        else:
            f = filedialog.askopenfilename(filetypes=[("JSON config","*.json"), ("All files","*.*")])
        if f: path_var.set(f)

    def choose_out_dir():
        d = filedialog.askdirectory()
        if d: out_var.set(d)

    def run_button():
        try:
            gap = int(gap_var.get()) if gap_var.get().strip() else None
        except Exception:
            messagebox.showerror("Invalid gap budget", "Gap Budget must be an integer number of bytes.")
            return
        try:
            out_dir = Path(out_var.get())
            out_dir.mkdir(parents=True, exist_ok=True)

            if src_mode.get() == "json":
                json_path = Path(path_var.get())
                run_with_json_config(json_path, out_dir, page_title=(title_var.get().strip() or None))
                enhanced_compile_and_run(str(json_path), str(out_dir), pixel_var.get(), zero_var.get(), viz_var.get(), gap)
            else:
                workbook = path_var.get()
                _compile_and_run(workbook, pixel_var.get(), zero_var.get(), viz_var.get(), gap)

            messagebox.showinfo("Done", "Compilation & simulation completed. Outputs are in the chosen output folder.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    frm = ttk.Frame(root, padding=12); frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="Source Type").grid(row=0, column=0, sticky="w")
    ttk.Combobox(frm, textvariable=src_mode, values=["json", "xlsx"], width=10).grid(row=0, column=1, sticky="w")

    ttk.Label(frm, text="Input Path").grid(row=1, column=0, sticky="w", pady=2)
    ttk.Entry(frm, textvariable=path_var, width=56).grid(row=1, column=1, sticky="we", pady=2)
    ttk.Button(frm, text="Browse...", command=choose_src_file).grid(row=1, column=2, padx=4, pady=2)

    ttk.Label(frm, text="Output Folder").grid(row=2, column=0, sticky="w", pady=2)
    ttk.Entry(frm, textvariable=out_var, width=56).grid(row=2, column=1, sticky="we", pady=2)
    ttk.Button(frm, text="Choose...", command=choose_out_dir).grid(row=2, column=2, padx=4, pady=2)

    ttk.Label(frm, text="Pixel bit width (E)").grid(row=3, column=0, sticky="w", pady=2)
    ttk.Combobox(frm, textvariable=pixel_var, values=[8,16,32,64], width=12).grid(row=3, column=1, sticky="w", pady=2)

    ttk.Label(frm, text="Zero mode").grid(row=4, column=0, sticky="w", pady=2)
    ttk.Combobox(frm, textvariable=zero_var, values=["idle_lanes","zero_pad"], width=12).grid(row=4, column=1, sticky="w", pady=2)

    ttk.Label(frm, text="Visualization format").grid(row=5, column=0, sticky="w", pady=2)
    ttk.Combobox(frm, textvariable=viz_var, values=["gif","mp4"], width=10).grid(row=5, column=1, sticky="w", pady=2)

    ttk.Label(frm, text="Gap Budget (Bytes)").grid(row=6, column=0, sticky="w", pady=2)
    ttk.Entry(frm, textvariable=gap_var, width=16).grid(row=6, column=1, sticky="w", pady=2)

    ttk.Label(frm, text="Dashboard Title (optional)").grid(row=7, column=0, sticky="w", pady=2)
    ttk.Entry(frm, textvariable=title_var, width=32).grid(row=7, column=1, sticky="w", pady=2)

    ttk.Button(frm, text="Run", command=run_button).grid(row=8, column=0, columnspan=3, pady=10)

    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli()
    else:
        run_gui()