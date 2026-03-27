#!/usr/bin/env python3
"""
Step 0: Create Pool Mask
========================
Click to place a closed polygon around the weld pool, smooth it,
and save a binary mask (white = inside, black = outside).

Usage:
    python step0_create_mask.py
    python step0_create_mask.py --image "4 mm images/frame_00000.png" --output "4 mm pool_mask.png"

Controls:
    Left-click       Add a boundary point
    Right-click / Z  Undo last point
    Backspace        Undo last point
    Enter            Save mask and quit
    Escape           Quit without saving
"""

import os
import sys
import argparse
import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageTk, ImageDraw


_PT_R      = 5       # control-point dot radius (px)
_PT_COL    = "#FF4444"
_PT0_COL   = "#FFFF00"   # first point in yellow so you can close the loop
_FILL_A    = 55      # 0-255 preview fill opacity


# ── smoothing ─────────────────────────────────────────────────────────────────

def _smooth_polygon(points, n_interp: int = 600, sigma: int = 0):
    """
    1. Arc-length-interpolate the closed polygon to `n_interp` evenly-spaced pts.
    2. Apply a Gaussian blur (periodic / wrap) with the given sigma.

    sigma is in units of the interpolated array indices (not pixels).
    Returns a list of (x, y) float tuples.
    """
    n = len(points)
    if n < 3:
        return list(points)

    # Close the loop for interpolation
    pts = np.array(points + [points[0]], dtype=float)

    # Cumulative arc-length parameterisation
    d     = np.diff(pts, axis=0)
    seg   = np.hypot(d[:, 0], d[:, 1])
    cum   = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total == 0:
        return list(points)

    u  = np.linspace(0.0, total, n_interp, endpoint=False)
    xi = np.interp(u, cum, pts[:, 0])
    yi = np.interp(u, cum, pts[:, 1])

    if sigma > 0:
        # Gaussian kernel
        half   = sigma * 3
        k      = np.arange(-half, half + 1, dtype=float)
        kernel = np.exp(-0.5 * (k / sigma) ** 2)
        kernel /= kernel.sum()

        # Periodic (wrap) convolution: tile × 3, convolve, take middle slice
        xi = np.convolve(np.tile(xi, 3), kernel, mode="same")[n_interp: 2 * n_interp]
        yi = np.convolve(np.tile(yi, 3), kernel, mode="same")[n_interp: 2 * n_interp]

    return list(zip(xi.tolist(), yi.tolist()))


# ── GUI ───────────────────────────────────────────────────────────────────────

class MaskCreator:
    def __init__(self, root: tk.Tk, image_path: str, output_path: str) -> None:
        self.root        = root
        self.image_path  = image_path
        self.output_path = output_path
        self.points: list[tuple[float, float]] = []
        self._bg_photo   = None   # kept alive to prevent GC
        self._overlay    = None

        self.root.title("Pool Mask Creator  –  Step 0")
        self.root.configure(bg="#1e1e1e")

        # Load image and compute display scale
        self.orig = Image.open(image_path).convert("RGB")
        self.orig_w, self.orig_h = self.orig.size

        sw, sh    = root.winfo_screenwidth(), root.winfo_screenheight()
        self.scale = min((sw - 80) / self.orig_w,
                         (sh - 180) / self.orig_h,
                         1.0)
        self.dw = int(self.orig_w * self.scale)
        self.dh = int(self.orig_h * self.scale)

        disp = self.orig.resize((self.dw, self.dh), Image.LANCZOS)
        self._bg_photo = ImageTk.PhotoImage(disp)

        self.smooth_var = tk.IntVar(value=0)
        self._build_ui()

        self.root.bind("<Return>",    lambda _: self.save_mask())
        self.root.bind("<Escape>",    lambda _: self.root.quit())
        self.root.bind("<BackSpace>", lambda _: self.undo())
        self.root.bind("<z>",         lambda _: self.undo())
        self.root.bind("<Z>",         lambda _: self.undo())

        self._redraw()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        bar = tk.Frame(self.root, bg="#2b2b2b", pady=6)
        bar.pack(side=tk.TOP, fill=tk.X)

        def btn(label, cmd, bg="#404040"):
            return tk.Button(bar, text=label, command=cmd,
                             bg=bg, fg="white", relief=tk.FLAT,
                             padx=12, pady=5, cursor="hand2",
                             activebackground="#606060", activeforeground="white")

        btn("Undo  [Z / ⌫]", self.undo).pack(side=tk.LEFT, padx=6)
        btn("Clear All",      self.clear).pack(side=tk.LEFT, padx=2)
        btn("Save  [↵]",      self.save_mask, bg="#2e7d32").pack(side=tk.LEFT, padx=8)

        tk.Label(bar, text="Smoothing:",
                 bg="#2b2b2b", fg="#bbbbbb").pack(side=tk.LEFT, padx=(20, 4))
        tk.Scale(bar, variable=self.smooth_var,
                 from_=0, to=30, resolution=1, orient=tk.HORIZONTAL, length=170,
                 bg="#2b2b2b", fg="white", troughcolor="#555555",
                 highlightthickness=0, command=lambda _: self._redraw()
                 ).pack(side=tk.LEFT)
        tk.Label(bar, text="(0 = raw polygon)",
                 bg="#2b2b2b", fg="#777777",
                 font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(
            value="Left-click: add point  |  Right-click / Z: undo  |  Enter: save")
        tk.Label(bar, textvariable=self.status_var,
                 bg="#2b2b2b", fg="#cccccc").pack(side=tk.RIGHT, padx=12)

        outer = tk.Frame(self.root, bg="#1e1e1e")
        outer.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(outer,
                                width=self.dw, height=self.dh,
                                bg="black", cursor="crosshair",
                                highlightthickness=0)
        hbar = tk.Scrollbar(outer, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vbar = tk.Scrollbar(outer, orient=tk.VERTICAL,   command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set,
                              scrollregion=(0, 0, self.dw, self.dh))
        vbar.pack(side=tk.RIGHT,  fill=tk.Y)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self._on_left)
        self.canvas.bind("<Button-3>", lambda e: self.undo())

    # ── events ────────────────────────────────────────────────────────────────

    def _on_left(self, event: tk.Event) -> None:
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.points.append((x, y))
        self._redraw()
        n = len(self.points)
        self.status_var.set(
            f"{n} point{'s' if n != 1 else ''}  |  Z/⌫: undo  |  Enter: save")

    def undo(self) -> None:
        if self.points:
            self.points.pop()
            self._redraw()
            n = len(self.points)
            self.status_var.set(f"{n} pts  |  Z/⌫: undo  |  Enter: save")

    def clear(self) -> None:
        self.points.clear()
        self._redraw()
        self.status_var.set("Cleared.  Left-click to start placing points.")

    # ── drawing ───────────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._bg_photo)

        n = len(self.points)
        if n == 0:
            return

        if n >= 3:
            contour = _smooth_polygon(self.points, sigma=self.smooth_var.get())

            overlay = Image.new("RGBA", (self.dw, self.dh), (0, 0, 0, 0))
            d = ImageDraw.Draw(overlay)
            d.polygon(contour, fill=(255, 255, 255, _FILL_A))
            d.line(contour + [contour[0]], fill=(0, 220, 0, 220), width=2)

            self._overlay = ImageTk.PhotoImage(overlay)
            self.canvas.create_image(0, 0, anchor="nw", image=self._overlay)

        # Raw polygon edges (dashed) so user can see the original clicks
        if n >= 2:
            flat = [c for pt in self.points for c in pt] + list(self.points[0])
            self.canvas.create_line(flat, fill="#666666", width=1, dash=(4, 5))

        # Control-point dots
        for i, (x, y) in enumerate(self.points):
            r = _PT_R
            self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                    fill=_PT0_COL if i == 0 else _PT_COL,
                                    outline="white", width=1)

    # ── save ──────────────────────────────────────────────────────────────────

    def save_mask(self) -> None:
        if len(self.points) < 3:
            messagebox.showwarning("Too few points",
                                   "Place at least 3 boundary points first.")
            return

        contour_canvas = _smooth_polygon(self.points, n_interp=4000,
                                         sigma=self.smooth_var.get())
        inv            = 1.0 / self.scale
        contour_orig   = [(x * inv, y * inv) for x, y in contour_canvas]

        mask = Image.new("L", (self.orig_w, self.orig_h), 0)
        ImageDraw.Draw(mask).polygon(contour_orig, fill=255)

        out_dir = os.path.dirname(os.path.abspath(self.output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        mask.save(self.output_path)

        messagebox.showinfo(
            "Mask Saved",
            f"Saved to:\n{self.output_path}\n\n"
            f"Resolution : {self.orig_w} × {self.orig_h} px\n"
            f"Control pts: {len(self.points)}\n"
            f"Smoothing  : {self.smooth_var.get()}")
        self.root.quit()


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Draw pool boundary mask")
    parser.add_argument("--image",  default=os.path.join("2 mm images", "frame_00000.png"))
    parser.add_argument("--output", default="2 mm pool_mask.png")
    args = parser.parse_args()

    image_path  = os.path.join(script_dir, args.image)
    output_path = os.path.join(script_dir, args.output)

    if not os.path.isfile(image_path):
        sys.exit(f"Image not found: {image_path}")

    root = tk.Tk()
    MaskCreator(root, image_path, output_path)
    root.mainloop()
    try:
        root.destroy()
    except tk.TclError:
        pass


if __name__ == "__main__":
    main()
