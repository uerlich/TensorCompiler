
from __future__ import annotations
from pathlib import Path
import csv
import imageio
import numpy as np
import matplotlib.pyplot as plt
from .mpl_compat import install_matplotlib_rgb_shim
install_matplotlib_rgb_shim()

def synthesize_beats(total_bytes: int, commit_B: int = 256, beat_B: int = 64):
    beats = []
    remaining = int(total_bytes)
    commit_id = 0
    while remaining > 0:
        chunk = min(commit_B, remaining)
        full_beats = chunk // beat_B
        tail = chunk % beat_B
        for _ in range(full_beats):
            beats.append((commit_id, beat_B))
        if tail:
            beats.append((commit_id, tail))
        remaining -= chunk
        commit_id += 1
    return beats

def write_beats_csv(beats, out_csv: str|Path):
    out_csv = Path(out_csv)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beat_idx","commit_id","bytes"])
        for i,(cid, nB) in enumerate(beats):
            w.writerow([i, cid, nB])
    return str(out_csv)

def _fig_to_ndarray(fig):
    import numpy as np
    fig.canvas.draw()
    try:
        # Most backends (older mpl): tostring_rgb
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.tostring_rgb()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        return img
    except Exception:
        # Newer/other backends: buffer_rgba
        arr = np.asarray(fig.canvas.buffer_rgba())
        return arr[:, :, :3].copy()

def render_beat_timeline_gif(beats, out_gif: str|Path):
    out_gif = Path(out_gif)
    frames = []
    occ = 0
    current_cid = beats[0][0] if beats else 0
    for i,(cid, nB) in enumerate(beats):
        if cid != current_cid:
            occ = 0
            current_cid = cid
        occ = min(256, occ + nB)
        fig = plt.figure(figsize=(6, 2.5), dpi=120)
        ax = fig.add_axes([0.08,0.3,0.9,0.6])
        ax.bar([0],[occ], width=0.6)
        ax.set_ylim(0,256)
        ax.set_xticks([])
        ax.set_ylabel("Bytes in-commit")
        ax.set_title(f"Beat {i} — Commit {cid} (+{nB}B)")
        fig.canvas.draw()
        img = _fig_to_ndarray(fig)
        frames.append(img)
        plt.close(fig)
    imageio.mimsave(out_gif, frames, duration=0.2)
    return str(out_gif)
