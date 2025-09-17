
def install_matplotlib_rgb_shim():
    """Ensure fig.canvas.tostring_rgb() exists by aliasing to buffer_rgba().
    Works on Matplotlib builds that removed/never exposed tostring_rgb()."""
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except Exception:
        return
    if not hasattr(FigureCanvasAgg, "tostring_rgb"):
        def tostring_rgb(self):
            import numpy as np
            # buffer_rgba() -> numpy array (H,W,4); drop alpha and return bytes
            arr = np.asarray(self.buffer_rgba())
            return arr[:, :, :3].copy().tobytes()
        FigureCanvasAgg.tostring_rgb = tostring_rgb
