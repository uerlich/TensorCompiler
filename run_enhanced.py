#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from enhancements.enhanced_runner import enhanced_compile_and_run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to JSON config (axes/shape)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--pixel-bits", type=int, default=8)
    ap.add_argument("--zero-mode", choices=["idle_lanes","zero_pad"], default="idle_lanes")
    ap.add_argument("--viz", choices=["gif","mp4"], default="gif")
    ap.add_argument("--gap-budget-B", type=int, default=None)
    args = ap.parse_args()

    res = enhanced_compile_and_run(args.json, args.out, args.pixel_bits, args.zero_mode, args.viz, args.gap_budget_B)
    print("Enhanced run outputs:", res)

if __name__ == "__main__":
    main()