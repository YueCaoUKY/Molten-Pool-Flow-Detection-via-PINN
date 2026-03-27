#!/usr/bin/env python3
"""
Step 2: Interactive Two-Stage Calibration
==========================================
Stage 1 - Click 4 pool boundary points to define horizontal and vertical axes.
Stage 2 - Fine-tune with sliders; see real-time ratio and symmetry IoU.

Usage:
    python step2_calibration.py --case 0mm
    python step2_calibration.py --case 4mm
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration App
# ═══════════════════════════════════════════════════════════════════════════════

class CalibrationApp:
    """Two-stage interactive calibration: point selection -> fine-tuning."""

    def __init__(self, mask_path, output_dir, target_ratio):
        # Load mask
        self.mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.mask_gray is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        self.mask_bool = self.mask_gray > 127
        self.h, self.w = self.mask_gray.shape
        print(f"Mask: {self.w}x{self.h}, {self.mask_bool.sum()} px")

        self.output_dir = output_dir
        self.target_ratio = target_ratio

        # State
        self.clicked_points = []
        self.H_initial = np.eye(3, dtype=np.float64)
        self.center = np.array([self.w / 2.0, self.h / 2.0])
        self.sym_y_default = self.h / 2.0
        self.stage = 1

        # Saved slider values (used when loading existing calibration)
        self.init_slider_vals = None

        # Try to load existing calibration
        self._try_load_existing()

    # ─── Load existing calibration ───────────────────────────────────────────

    def _try_load_existing(self):
        """Check for existing calibration.npz and load as initial parameters."""
        calib_path = os.path.join(self.output_dir, "calibration.npz")
        if not os.path.exists(calib_path):
            return

        try:
            data = np.load(calib_path, allow_pickle=True)
        except Exception as e:
            print(f"Warning: could not load {calib_path}: {e}")
            return

        # Load clicked points
        if 'clicked_points' in data:
            pts = data['clicked_points']
            self.clicked_points = [tuple(p) for p in pts]

        # Load H_initial
        if 'H_initial' in data:
            self.H_initial = data['H_initial'].astype(np.float64)

        # Recompute center and sym_y_default from clicked points
        if len(self.clicked_points) == 4:
            p_left = np.array(self.clicked_points[0])
            p_right = np.array(self.clicked_points[1])
            p_top = np.array(self.clicked_points[2])
            p_bottom = np.array(self.clicked_points[3])
            h_mid = (p_left + p_right) / 2.0
            v_mid = (p_top + p_bottom) / 2.0
            H_len = np.linalg.norm(p_right - p_left)
            V_len = H_len / self.target_ratio
            dst_pts = np.array([
                [h_mid[0] - H_len / 2, h_mid[1]],
                [h_mid[0] + H_len / 2, h_mid[1]],
                [v_mid[0], v_mid[1] - V_len / 2],
                [v_mid[0], v_mid[1] + V_len / 2],
            ])
            self.center = dst_pts.mean(axis=0).astype(np.float64)
            self.sym_y_default = float(h_mid[1])

        # Load slider values
        self.init_slider_vals = {
            'rotation': float(data['fine_rotation']) if 'fine_rotation' in data else 0,
            'tilt':     float(data['fine_tilt']) if 'fine_tilt' in data else 0,
            'scale':    float(data['fine_scale']) if 'fine_scale' in data else 1.0,
            'aspect':   float(data['fine_aspect']) if 'fine_aspect' in data else 1.0,
            'shift_x':  float(data['fine_shift_x']) if 'fine_shift_x' in data else 0,
            'shift_y':  float(data['fine_shift_y']) if 'fine_shift_y' in data else 0,
            'sym_y':    float(data['symmetry_line_y']) if 'symmetry_line_y' in data else self.sym_y_default,
        }

        print(f"Loaded existing calibration from {calib_path}")
        print(f"  Clicked points: {len(self.clicked_points)}")
        print(f"  Slider values: rotation={self.init_slider_vals['rotation']:.2f}, "
              f"tilt={self.init_slider_vals['tilt']:.2f}, "
              f"scale={self.init_slider_vals['scale']:.3f}, "
              f"aspect={self.init_slider_vals['aspect']:.3f}")

    # ─── Run ─────────────────────────────────────────────────────────────────

    def run(self):
        """Launch the interactive GUI."""
        if len(self.clicked_points) == 4 and self.init_slider_vals is not None:
            # Existing calibration loaded -> skip Stage 1, go to Stage 2
            print("Skipping Stage 1 (using loaded points). Starting Stage 2...")
            self.fig = plt.figure(figsize=(14, 9))
            self._transition_to_stage2()
        else:
            self._build_stage1()
        plt.show()

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 1: Point Selection
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_stage1(self):
        """Create Stage 1 figure for point selection."""
        self.clicked_points = []
        self.stage = 1

        self.fig = plt.figure(figsize=(14, 9))
        self.ax_s1 = self.fig.add_subplot(111)
        self.ax_s1.imshow(self.mask_gray, cmap='gray')

        # Draw contour for reference
        contours, _ = cv2.findContours(
            self.mask_bool.astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea).squeeze()
            if cnt.ndim == 2:
                self.ax_s1.plot(cnt[:, 0], cnt[:, 1], 'g-', lw=1, alpha=0.5)

        self.ax_s1.set_title(
            "Stage 1: Click LEFT endpoint of horizontal axis (pool boundary)",
            fontsize=12, fontweight='bold')
        self.ax_s1.set_aspect('equal')
        self.ax_s1.set_xlim(0, self.w)
        self.ax_s1.set_ylim(self.h, 0)

        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.draw()

    def _on_click(self, event):
        """Handle mouse click to collect 4 boundary points."""
        if event.inaxes != self.ax_s1 or event.button != 1:
            return

        x, y = event.xdata, event.ydata
        self.clicked_points.append((x, y))
        n = len(self.clicked_points)

        # Draw the point with label
        labels = ['L', 'R', 'T', 'B']
        colors = ['yellow', 'yellow', 'cyan', 'cyan']
        self.ax_s1.plot(x, y, 'o', color=colors[n - 1], markersize=10, markeredgecolor='black')
        self.ax_s1.annotate(labels[n - 1], (x, y),
                            textcoords="offset points", xytext=(8, 8),
                            color=colors[n - 1], fontsize=12, fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.5, pad=2))

        # Draw connecting lines
        if n == 2:
            p1, p2 = self.clicked_points[0], self.clicked_points[1]
            self.ax_s1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'y-', lw=2, label='Horizontal axis')
            self.ax_s1.legend(loc='upper left', fontsize=9)

        if n == 4:
            p3, p4 = self.clicked_points[2], self.clicked_points[3]
            self.ax_s1.plot([p3[0], p4[0]], [p3[1], p4[1]], 'c-', lw=2, label='Vertical axis')
            self.ax_s1.legend(loc='upper left', fontsize=9)
            self.fig.canvas.draw()

            # Compute initial homography and transition
            self._compute_initial_H()
            self._transition_to_stage2()
            return

        # Update title with next instruction
        prompts = [
            "Click LEFT endpoint of horizontal axis (pool boundary)",
            "Click RIGHT endpoint of horizontal axis",
            "Click TOP endpoint of vertical axis (pool boundary)",
            "Click BOTTOM endpoint of vertical axis",
        ]
        if n < 4:
            self.ax_s1.set_title(
                f"Stage 1: Point {n}/4 placed. Next: {prompts[n]}",
                fontsize=12, fontweight='bold')

        self.fig.canvas.draw_idle()

    def _compute_initial_H(self):
        """
        Compute initial perspective transform from 4 clicked boundary points.

        Source: 4 clicked points (left, right, top, bottom of pool boundary)
        Target: each line straightened around its own midpoint, lengths
                adjusted so H_len / V_len = target_ratio.

        Works whether the two lines intersect (4mm case) or not (0mm case
        where the vertical line is to the right of the horizontal line).
        """
        p_left = np.array(self.clicked_points[0])
        p_right = np.array(self.clicked_points[1])
        p_top = np.array(self.clicked_points[2])
        p_bottom = np.array(self.clicked_points[3])

        # Midpoints of each line
        h_mid = (p_left + p_right) / 2.0
        v_mid = (p_top + p_bottom) / 2.0

        # Lengths in source image
        H_len = np.linalg.norm(p_right - p_left)
        V_len = H_len / self.target_ratio

        # Target points: each line straightened around its own midpoint
        src_pts = np.array([p_left, p_right, p_top, p_bottom], dtype=np.float32)
        dst_pts = np.array([
            [h_mid[0] - H_len / 2, h_mid[1]],   # left  -> horizontal at h_mid y
            [h_mid[0] + H_len / 2, h_mid[1]],   # right -> horizontal at h_mid y
            [v_mid[0], v_mid[1] - V_len / 2],   # top   -> vertical at v_mid x
            [v_mid[0], v_mid[1] + V_len / 2],   # bottom -> vertical at v_mid x
        ], dtype=np.float32)

        self.H_initial = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Center for fine-tuning = geometric mean of the 4 target points
        self.center = dst_pts.mean(axis=0).astype(np.float64)
        # Symmetry line defaults to the horizontal central line's y-coordinate
        self.sym_y_default = float(h_mid[1])

        cx, cy = self.center
        print(f"\nInitial homography computed from 4-point selection")
        print(f"  H-line midpoint: ({h_mid[0]:.1f}, {h_mid[1]:.1f})")
        print(f"  V-line midpoint: ({v_mid[0]:.1f}, {v_mid[1]:.1f})")
        print(f"  Fine-tune center: ({cx:.1f}, {cy:.1f})")
        print(f"  H_len={H_len:.1f} px, V_len={V_len:.1f} px")
        print(f"  Initial ratio: {H_len / V_len:.3f} (target: {self.target_ratio:.3f})")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2: Fine-tuning
    # ═══════════════════════════════════════════════════════════════════════════

    def _transition_to_stage2(self):
        """Disconnect click handler, rebuild figure for Stage 2."""
        if hasattr(self, 'cid_click'):
            self.fig.canvas.mpl_disconnect(self.cid_click)
        self.stage = 2

        self.fig.clear()

        # Two panels
        self.ax_left = self.fig.add_axes([0.02, 0.38, 0.46, 0.58])
        self.ax_right = self.fig.add_axes([0.52, 0.38, 0.46, 0.58])

        # Draw left panel: original mask with clicked reference axes
        self.ax_left.imshow(self.mask_gray, cmap='gray')
        for i, (x, y) in enumerate(self.clicked_points):
            labels = ['L', 'R', 'T', 'B']
            colors = ['yellow', 'yellow', 'cyan', 'cyan']
            self.ax_left.plot(x, y, 'o', color=colors[i], markersize=8,
                              markeredgecolor='black')
            self.ax_left.annotate(labels[i], (x, y),
                                  textcoords="offset points", xytext=(6, 6),
                                  color=colors[i], fontsize=10, fontweight='bold')
        p1, p2 = self.clicked_points[0], self.clicked_points[1]
        p3, p4 = self.clicked_points[2], self.clicked_points[3]
        self.ax_left.plot([p1[0], p2[0]], [p1[1], p2[1]], 'y-', lw=2)
        self.ax_left.plot([p3[0], p4[0]], [p3[1], p4[1]], 'c-', lw=2)
        self.ax_left.set_title('Original Mask + Reference Axes', fontweight='bold', fontsize=10)
        self.ax_left.set_aspect('equal')
        self.ax_left.set_xlim(0, self.w)
        self.ax_left.set_ylim(self.h, 0)

        # Build sliders and buttons
        self._build_stage2_controls()

        # Initial draw
        self._update(None)
        self.fig.canvas.draw()

    def _build_stage2_controls(self):
        """Create 7 sliders and 2 buttons for Stage 2."""
        # Use loaded slider values if available, otherwise defaults
        sv = self.init_slider_vals or {}
        sym_y_init = sv.get('sym_y', self.sym_y_default)

        slider_defs = [
            # (y_pos, label, vmin, vmax, vinit, vstep, color)
            (0.30, 'Rotation fine (deg)', -15, 15, sv.get('rotation', 0), 0.5, None),
            (0.27, 'Tilt fine (deg)',      -45, 45, sv.get('tilt', 0), 0.5, None),
            (0.24, 'Scale',               0.3, 3.0, sv.get('scale', 1.0), 0.01, None),
            (0.21, 'Aspect',              0.3, 3.0, sv.get('aspect', 1.0), 0.01, None),
            (0.18, 'Shift X',       -self.w / 2, self.w / 2, sv.get('shift_x', 0), 1, None),
            (0.15, 'Shift Y',       -self.h / 2, self.h / 2, sv.get('shift_y', 0), 1, None),
            (0.12, 'Symmetry Y',          0, self.h, sym_y_init, 0.5, 'yellow'),
        ]

        self.sl = {}
        names = ['rotation', 'tilt', 'scale', 'aspect', 'shift_x', 'shift_y', 'sym_y']
        for (yp, label, vmin, vmax, vinit, vstep, color), name in zip(slider_defs, names):
            ax = self.fig.add_axes([0.15, yp, 0.72, 0.018])
            kwargs = dict(label=label, valmin=vmin, valmax=vmax,
                          valinit=vinit, valstep=vstep)
            if color:
                kwargs['color'] = color
            s = Slider(ax, **kwargs)
            s.on_changed(self._update)
            self.sl[name] = s

        # Buttons
        ax_resel = self.fig.add_axes([0.15, 0.04, 0.15, 0.04])
        self.btn_reselect = Button(ax_resel, 'Reselect Points', color='lightyellow')
        self.btn_reselect.on_clicked(self._on_reselect)

        ax_reset = self.fig.add_axes([0.42, 0.04, 0.12, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset Sliders', color='lightyellow')
        self.btn_reset.on_clicked(self._on_reset)

        ax_save = self.fig.add_axes([0.72, 0.04, 0.12, 0.04])
        self.btn_save = Button(ax_save, 'Save', color='lightgreen')
        self.btn_save.on_clicked(self._on_save)

    # ─── Fine-tune homography ────────────────────────────────────────────────

    def _compute_fine_tune_H(self, rotation_deg, tilt_deg, scale, aspect,
                             shift_x, shift_y):
        """
        Build a parametric fine-tune homography centered on the corrected-space center.
        Same camera model as old 002_Calibration.py, but applied as delta adjustment.
        """
        cx, cy = self.center
        f = 0.5 * self.w

        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

        theta = np.deg2rad(-tilt_deg)
        R_tilt = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ], dtype=np.float64)

        phi = np.deg2rad(-rotation_deg)
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        R_rot = np.array([
            [cos_p, -sin_p, cx * (1 - cos_p) + cy * sin_p],
            [sin_p, cos_p,  cy * (1 - cos_p) - cx * sin_p],
            [0, 0, 1]
        ], dtype=np.float64)

        K_inv = np.linalg.inv(K)
        H_persp = K @ R_tilt @ K_inv
        H_base = R_rot @ H_persp

        S = np.diag([scale, scale * aspect, 1.0])
        H = S @ H_base

        # Re-center
        pool_pt = H @ np.array([cx, cy, 1.0])
        pool_pt = pool_pt / pool_pt[2]
        tx = cx - pool_pt[0] + shift_x
        ty = cy - pool_pt[1] + shift_y
        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

        return T @ H

    def _get_compound_H(self):
        """Compose fine-tune H on top of initial H."""
        H_fine = self._compute_fine_tune_H(
            self.sl['rotation'].val, self.sl['tilt'].val,
            self.sl['scale'].val, self.sl['aspect'].val,
            self.sl['shift_x'].val, self.sl['shift_y'].val,
        )
        return H_fine @ self.H_initial

    # ─── Measurements ────────────────────────────────────────────────────────

    def _measure_contour(self, mask_bool):
        """
        Measure horizontal and vertical extents of corrected pool contour.

        Returns h_len, v_len, cx, cy.
        """
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0, self.w / 2, self.h / 2, None

        cnt = max(contours, key=cv2.contourArea).squeeze()
        if cnt.ndim != 2:
            return 0, 0, self.w / 2, self.h / 2, None

        M = cv2.moments(mask_bool.astype(np.uint8))
        if M['m00'] == 0:
            return 0, 0, self.w / 2, self.h / 2, None

        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        # Horizontal width at centroid y
        pts_at_cy = cnt[np.abs(cnt[:, 1] - cy) < 3]
        h_len = float(pts_at_cy[:, 0].max() - pts_at_cy[:, 0].min()) if len(pts_at_cy) > 1 else 0

        # Vertical height at centroid x
        pts_at_cx = cnt[np.abs(cnt[:, 0] - cx) < 3]
        v_len = float(pts_at_cx[:, 1].max() - pts_at_cx[:, 1].min()) if len(pts_at_cx) > 1 else 0

        # Fallback: bounding box
        if h_len == 0:
            h_len = float(cnt[:, 0].max() - cnt[:, 0].min())
        if v_len == 0:
            v_len = float(cnt[:, 1].max() - cnt[:, 1].min())

        return h_len, v_len, cx, cy, cnt

    def _compute_symmetry_iou(self, mask_bool, sym_y):
        """
        IoU between mirrored top-half and actual bottom-half of corrected mask.
        Returns float in [0, 1].
        """
        h, w = mask_bool.shape
        sym_y_int = int(round(sym_y))
        sym_y_int = max(1, min(h - 2, sym_y_int))

        # Bottom half
        bottom = np.zeros((h, w), dtype=bool)
        bottom[sym_y_int:, :] = mask_bool[sym_y_int:, :]

        # Mirror top half below symmetry line
        top_region = mask_bool[:sym_y_int, :]
        top_flipped = top_region[::-1, :]

        mirrored = np.zeros((h, w), dtype=bool)
        n_rows = min(top_flipped.shape[0], h - sym_y_int)
        mirrored[sym_y_int:sym_y_int + n_rows, :] = top_flipped[:n_rows, :]

        intersection = np.logical_and(bottom, mirrored).sum()
        union = np.logical_or(bottom, mirrored).sum()

        return float(intersection) / max(float(union), 1.0)

    # ─── Update callback ─────────────────────────────────────────────────────

    def _update(self, _):
        """Slider changed -> recompute corrected mask and redraw right panel."""
        H_compound = self._get_compound_H()
        sym_y = self.sl['sym_y'].val

        # Warp mask
        corrected_img = cv2.warpPerspective(
            self.mask_gray, H_compound, (self.w, self.h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        corrected_bool = corrected_img > 127

        # Measure
        h_len, v_len, cx, cy, cnt = self._measure_contour(corrected_bool)
        current_ratio = h_len / max(v_len, 1e-5)
        iou = self._compute_symmetry_iou(corrected_bool, sym_y)

        # Draw right panel
        self.ax_right.clear()

        # Show corrected mask
        self.ax_right.imshow(corrected_img, cmap='gray')

        # Green contour
        if cnt is not None and cnt.ndim == 2:
            self.ax_right.plot(cnt[:, 0], cnt[:, 1], 'g-', lw=2)

        # Yellow horizontal center line
        self.ax_right.axhline(cy, color='yellow', lw=1.5, ls='-', alpha=0.8)

        # Cyan vertical center line
        self.ax_right.axvline(cx, color='cyan', lw=1.5, ls='-', alpha=0.8)

        # Magenta symmetry line
        self.ax_right.axhline(sym_y, color='magenta', lw=2, ls='--', alpha=0.9)

        # Measurement annotations (H-length)
        self.ax_right.annotate(
            '', xy=(cx + h_len / 2, cy), xytext=(cx - h_len / 2, cy),
            arrowprops=dict(arrowstyle='<->', color='yellow', lw=2))
        self.ax_right.text(cx, cy - 15, f'H={h_len:.0f}',
                           ha='center', fontsize=10, color='yellow', fontweight='bold',
                           bbox=dict(facecolor='black', alpha=0.6, pad=2))

        # Measurement annotations (V-length)
        self.ax_right.annotate(
            '', xy=(cx, cy + v_len / 2), xytext=(cx, cy - v_len / 2),
            arrowprops=dict(arrowstyle='<->', color='cyan', lw=2))
        self.ax_right.text(cx + 20, cy, f'V={v_len:.0f}',
                           ha='left', fontsize=10, color='cyan', fontweight='bold',
                           bbox=dict(facecolor='black', alpha=0.6, pad=2))

        # Color-coded ratio and IoU
        ratio_ok = abs(current_ratio - self.target_ratio) / max(self.target_ratio, 1e-5) < 0.05
        iou_ok = iou > 0.90

        ratio_color = 'lime' if ratio_ok else 'red'
        iou_color = 'lime' if iou_ok else ('orange' if iou > 0.80 else 'red')

        title = (f"Ratio: {current_ratio:.3f}  (target: {self.target_ratio:.3f})   |   "
                 f"Symmetry IoU: {iou:.3f}")
        self.ax_right.set_title(title, fontsize=11, fontweight='bold')

        # Large ratio indicator in plot
        self.ax_right.text(
            self.w - 10, 30,
            f"Ratio = {current_ratio:.3f}",
            ha='right', fontsize=14, fontweight='bold', color=ratio_color,
            bbox=dict(facecolor='black', alpha=0.7, pad=4))
        self.ax_right.text(
            self.w - 10, 65,
            f"IoU = {iou:.3f}",
            ha='right', fontsize=14, fontweight='bold', color=iou_color,
            bbox=dict(facecolor='black', alpha=0.7, pad=4))

        self.ax_right.set_xlim(0, self.w)
        self.ax_right.set_ylim(self.h, 0)
        self.ax_right.set_aspect('equal')

        self.fig.canvas.draw_idle()

    # ─── Button callbacks ────────────────────────────────────────────────────

    def _on_reset(self, event):
        """Reset all sliders to default values."""
        self.sl['rotation'].set_val(0)
        self.sl['tilt'].set_val(0)
        self.sl['scale'].set_val(1.0)
        self.sl['aspect'].set_val(1.0)
        self.sl['shift_x'].set_val(0)
        self.sl['shift_y'].set_val(0)
        self.sl['sym_y'].set_val(self.sym_y_default)

    def _on_reselect(self, event):
        """Close current figure and restart Stage 1 from scratch."""
        self.init_slider_vals = None  # clear loaded values
        plt.close(self.fig)
        self._build_stage1()
        plt.show()

    def _on_save(self, event):
        """Save calibration.npz and corrected mask to output_dir."""
        H_compound = self._get_compound_H()
        sym_y = float(self.sl['sym_y'].val)

        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, "calibration.npz")

        np.savez(
            save_path,
            # Core calibration
            H=H_compound,
            H_initial=self.H_initial,
            # Symmetry parameters
            symmetry_line_y=sym_y,
            symmetry_slope=0.0,
            symmetry_x_center=float(self.center[0]),
            # Clicked reference points (for reproducibility)
            clicked_points=np.array(self.clicked_points),
            # Slider values (for reloading)
            fine_rotation=float(self.sl['rotation'].val),
            fine_tilt=float(self.sl['tilt'].val),
            fine_scale=float(self.sl['scale'].val),
            fine_aspect=float(self.sl['aspect'].val),
            fine_shift_x=float(self.sl['shift_x'].val),
            fine_shift_y=float(self.sl['shift_y'].val),
            # Config
            target_ratio=self.target_ratio,
            image_shape=np.array([self.h, self.w]),
        )

        # Also save corrected mask image
        corrected = cv2.warpPerspective(self.mask_gray, H_compound, (self.w, self.h))
        cv2.imwrite(os.path.join(self.output_dir, "mask_corrected.png"), corrected)

        print(f"\n{'=' * 60}")
        print(f"CALIBRATION SAVED")
        print(f"{'=' * 60}")
        print(f"  Output dir:      {self.output_dir}")
        print(f"  calibration.npz  (H, symmetry params, slider values)")
        print(f"  mask_corrected.png")
        print(f"  Symmetry line Y: {sym_y:.1f}")
        print(f"  Target ratio:    {self.target_ratio:.3f}")
        print(f"{'=' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from config import get_case

    parser = argparse.ArgumentParser(description="Step 2: Interactive calibration")
    parser.add_argument("--case", required=True, help="Case name (e.g., 0mm, 4mm)")
    args = parser.parse_args()

    cfg = get_case(args.case)
    app = CalibrationApp(
        mask_path=cfg["mask_path"],
        output_dir=cfg["output_dir"],
        target_ratio=cfg["target_ratio"],
    )
    app.run()
