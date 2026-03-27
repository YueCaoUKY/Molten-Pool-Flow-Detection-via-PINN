#!/usr/bin/env python3
"""
Step 4: Data Illustration
==========================
Generates four high-resolution figures per frame for each case:
  (1) Original video frame with RAFT flow arrows overlaid
  (2) Original mask boundary with raw RAFT flow arrows
  (3) Corrected mask boundary with corrected RAFT flow arrows
  (4) Symmetric mask boundary with symmetric RAFT flow arrows

Usage:
    python step4_data_illustration.py --case 0mm                # 1 frame (default frame 0)
    python step4_data_illustration.py --case 0mm --frames 5     # 5 frames evenly spaced across all RAFT frames
    python step4_data_illustration.py --case 4mm --frames 10    # 10 frames evenly spaced
"""

import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

matplotlib.rcParams.update({
    'font.family':     'serif',
    'font.serif':      ['Times New Roman'],
    'font.size':        9,
    'axes.titlesize':   9,
    'axes.labelsize':   9,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  8,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'mathtext.fontset': 'stix',
})


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_contour_xy(mask_bool):
    """Return the largest contour as (x_array, y_array), or None."""
    contours, _ = cv2.findContours(
        mask_bool.astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea).squeeze()
    if cnt.ndim != 2:
        return None
    cnt_closed = np.vstack([cnt, cnt[:1]])
    return cnt_closed[:, 0], cnt_closed[:, 1]


def sample_flow_on_grid(flow, mask_bool, step, min_mag_frac):
    """
    Sample flow on a regular grid inside the mask.
    Returns x, y, u, v, mag arrays (1-D, only valid grid points) and max_mag.
    """
    h, w = mask_bool.shape
    yg, xg = np.mgrid[step // 2:h:step, step // 2:w:step]
    in_mask = mask_bool[yg, xg]

    u = flow[0, yg, xg]
    v = flow[1, yg, xg]
    mag = np.sqrt(u ** 2 + v ** 2)

    max_mag = mag[in_mask].max() if in_mask.any() else 1e-5
    min_mag = min_mag_frac * max_mag

    keep = in_mask & (mag >= min_mag)
    return xg[keep], yg[keep], u[keep], v[keep], mag[keep], max_mag


def add_flow_quiver(ax, x, y, u, v, mag, max_mag, vis):
    """Draw flow arrows with magnitude-based coloring using quiver."""
    cmap = plt.get_cmap(vis["cmap"])
    norm = Normalize(vmin=0, vmax=max_mag)

    q = ax.quiver(
        x, y,
        u * vis["arrow_scale"],
        v * vis["arrow_scale"],
        mag,
        cmap=cmap, norm=norm,
        angles='xy', scale_units='xy', scale=1,
        width=vis["arrow_width"],
        headwidth=vis["arrow_headwidth"],
        headlength=vis["arrow_headlength"],
        alpha=0.85,
        zorder=5,
    )
    return q, norm


def add_colorbar(fig, ax, norm, cmap_name, label="Magnitude (px/frame)",
                 text_color='black'):
    """Add a colorbar to the right of the axes."""
    sm = ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, color=text_color)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=text_color)
    return cb


def format_ax(ax, w, h):
    """Common axis formatting for pool images."""
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect('equal')
    ax.axis('off')


# ── Figure generators ────────────────────────────────────────────────────────

def figure_flow_on_video(video_path, flow, mask_sample, stride, frame_idx,
                         vis, save_path, mask_boundary=None):
    """Figure 1: Original video frame with RAFT flow arrows overlaid."""
    if mask_boundary is None:
        mask_boundary = mask_sample
    h, w = mask_sample.shape

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx * stride)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx * stride} from video")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    x, y, u, v, mag, max_mag = sample_flow_on_grid(
        flow, mask_sample, vis["arrow_step"], vis["min_mag_frac"])

    vel_scale = vis.get("vel_scale", 1.0)
    vel_label = vis.get("vel_label", "Magnitude (px/frame)")
    mag = mag * vel_scale
    max_mag = max_mag * vel_scale

    fig, ax = plt.subplots(figsize=(2.8, 2.8 * h / w))
    ax.imshow(frame_rgb, alpha=0.85)

    cnt = get_contour_xy(mask_boundary)
    if cnt is not None:
        ax.plot(cnt[0], cnt[1], color='lime',
                lw=vis["boundary_width"], alpha=0.9)

    q, norm = add_flow_quiver(ax, x, y, u, v, mag, max_mag, vis)
    add_colorbar(fig, ax, norm, vis["cmap"], label=vel_label)

    format_ax(ax, w, h)

    fig.savefig(save_path, bbox_inches='tight',
                pad_inches=0.05, facecolor='white')
    plt.close(fig)
    print(f"    {os.path.basename(save_path)}")


def figure_raw_flow(flow, mask_sample, frame_idx, vis, save_path,
                    mask_boundary=None):
    """Figure 2: Original mask boundary + raw RAFT flow (black background)."""
    if mask_boundary is None:
        mask_boundary = mask_sample
    h, w = mask_sample.shape

    x, y, u, v, mag, max_mag = sample_flow_on_grid(
        flow, mask_sample, vis["arrow_step"], vis["min_mag_frac"])

    vel_scale = vis.get("vel_scale", 1.0)
    vel_label = vis.get("vel_label", "Magnitude (px/frame)")
    mag = mag * vel_scale
    max_mag = max_mag * vel_scale

    fig, ax = plt.subplots(figsize=(2.8, 2.8 * h / w))
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    cnt = get_contour_xy(mask_boundary)
    if cnt is not None:
        ax.plot(cnt[0], cnt[1], color='lime', lw=vis["boundary_width"], alpha=0.8)

    q, norm = add_flow_quiver(ax, x, y, u, v, mag, max_mag, vis)
    add_colorbar(fig, ax, norm, vis["cmap"], label=vel_label, text_color='white')

    format_ax(ax, w, h)
    ax.set_title(f"Raw RAFT Flow (frame {frame_idx})",
                 fontsize=9, fontweight='bold', color='white')

    fig.savefig(save_path, bbox_inches='tight',
                pad_inches=0.05, facecolor='black')
    plt.close(fig)
    print(f"    {os.path.basename(save_path)}")


def figure_corrected_flow(flow, mask_sample, frame_idx, vis, save_path,
                          mask_boundary=None):
    """Figure 3: Corrected mask boundary + corrected RAFT flow (white background)."""
    if mask_boundary is None:
        mask_boundary = mask_sample
    h, w = mask_sample.shape

    x, y, u, v, mag, max_mag = sample_flow_on_grid(
        flow, mask_sample, vis["arrow_step"], vis["min_mag_frac"])

    vel_scale = vis.get("vel_scale", 1.0)
    vel_label = vis.get("vel_label", "Magnitude (px/frame)")
    mag = mag * vel_scale
    max_mag = max_mag * vel_scale

    fig, ax = plt.subplots(figsize=(2.8, 2.8 * h / w))

    cnt = get_contour_xy(mask_boundary)
    if cnt is not None:
        ax.plot(cnt[0], cnt[1], color=vis["boundary_color"],
                lw=vis["boundary_width"], alpha=0.9)

    q, norm = add_flow_quiver(ax, x, y, u, v, mag, max_mag, vis)
    add_colorbar(fig, ax, norm, vis["cmap"], label=vel_label)

    format_ax(ax, w, h)
    ax.set_title(f"Corrected RAFT Flow (frame {frame_idx})",
                 fontsize=9, fontweight='bold')

    fig.savefig(save_path, bbox_inches='tight',
                pad_inches=0.05, facecolor='white')
    plt.close(fig)
    print(f"    {os.path.basename(save_path)}")


def figure_symmetric_flow(flow, mask_sample, sym_y, frame_idx, vis, save_path,
                          mask_boundary=None):
    """Figure 4: Symmetric mask boundary + symmetric RAFT flow (white background)."""
    if mask_boundary is None:
        mask_boundary = mask_sample
    h, w = mask_sample.shape

    x, y, u, v, mag, max_mag = sample_flow_on_grid(
        flow, mask_sample, vis["arrow_step"], vis["min_mag_frac"])

    vel_scale = vis.get("vel_scale", 1.0)
    vel_label = vis.get("vel_label", "Magnitude (px/frame)")
    mag = mag * vel_scale
    max_mag = max_mag * vel_scale

    fig, ax = plt.subplots(figsize=(2.8, 2.8 * h / w))

    cnt = get_contour_xy(mask_boundary)
    if cnt is not None:
        ax.plot(cnt[0], cnt[1], color=vis["boundary_color"],
                lw=vis["boundary_width"], alpha=0.9)

    # Symmetry line
    if sym_y is not None:
        ax.axhline(sym_y, color='gold', lw=2, ls='--', alpha=0.7,
                    label=f'Symmetry line y={sym_y:.0f}')
        ax.legend(loc='upper right')

    q, norm = add_flow_quiver(ax, x, y, u, v, mag, max_mag, vis)
    add_colorbar(fig, ax, norm, vis["cmap"], label=vel_label)

    format_ax(ax, w, h)
    ax.set_title(f"Symmetric RAFT Flow (frame {frame_idx})",
                 fontsize=9, fontweight='bold')

    fig.savefig(save_path, bbox_inches='tight',
                pad_inches=0.05, facecolor='white')
    plt.close(fig)
    print(f"    {os.path.basename(save_path)}")


# ── Main ─────────────────────────────────────────────────────────────────────

def generate_frame_set(frame_idx, case_name, cfg, vis, data_dir, fig_dir,
                       metadata, mask_original, mask_raft,
                       mask_corrected, mask_symmetric,
                       mask_raft_corrected, mask_raft_symmetric, sym_y):
    """Generate one set of 4 figures for a given frame index."""
    prefix = f"frame{frame_idx:04d}"
    stride = int(metadata['stride'])

    # Load flows for this frame
    flow_raw = np.load(os.path.join(data_dir, "flow_raw", f"{frame_idx:04d}.npy"))

    flow_corrected = None
    flow_corr_path = os.path.join(data_dir, "flow_corrected", f"{frame_idx:04d}.npy")
    if mask_corrected is not None and os.path.exists(flow_corr_path):
        flow_corrected = np.load(flow_corr_path)

    flow_symmetric = None
    flow_sym_path = os.path.join(data_dir, "flow_symmetric", f"{frame_idx:04d}.npy")
    if mask_symmetric is not None and os.path.exists(flow_sym_path):
        flow_symmetric = np.load(flow_sym_path)

    print(f"  Frame {frame_idx}:")

    # Fig 1: flow on video (RAFT mask for sampling, full mask for boundary)
    figure_flow_on_video(
        cfg["video_path"], flow_raw, mask_raft, stride, frame_idx, vis,
        os.path.join(fig_dir, f"{prefix}_fig1_flow_on_video.png"),
        mask_boundary=mask_original)

    # Fig 2: raw flow (RAFT mask for sampling, full mask for boundary)
    figure_raw_flow(
        flow_raw, mask_raft, frame_idx, vis,
        os.path.join(fig_dir, f"{prefix}_fig2_raw_flow.png"),
        mask_boundary=mask_original)

    # Fig 3: corrected flow (RAFT corrected mask for sampling, full corrected for boundary)
    if flow_corrected is not None:
        figure_corrected_flow(
            flow_corrected, mask_raft_corrected, frame_idx, vis,
            os.path.join(fig_dir, f"{prefix}_fig3_corrected_flow.png"),
            mask_boundary=mask_corrected)

    # Fig 4: symmetric flow (RAFT symmetric mask for sampling, full symmetric for boundary)
    if flow_symmetric is not None:
        figure_symmetric_flow(
            flow_symmetric, mask_raft_symmetric, sym_y, frame_idx, vis,
            os.path.join(fig_dir, f"{prefix}_fig4_symmetric_flow.png"),
            mask_boundary=mask_symmetric)


def run_illustration(case_name, cfg, vis, n_frames):
    """Generate illustration figures for one case."""
    output_dir = cfg["output_dir"]
    data_dir = os.path.join(output_dir, "RAFT_data")

    fig_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"data illustration {case_name}")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Data Illustration - case: {case_name}")
    print(f"{'=' * 60}")

    # Load metadata
    metadata = np.load(os.path.join(data_dir, "metadata.npz"))
    total_flow = int(metadata['n_flow_frames'])
    stride_val = int(metadata['stride'])
    fps = cfg.get("fps", float(metadata['fps']))  # prefer config; fall back to video metadata
    dt = stride_val / fps
    print(f"fps={fps:.1f}, stride={stride_val}, dt={dt*1000:.3f} ms, "
          f"n_flow={total_flow}")

    # Compute physical velocity conversion: px/frame-pair -> mm/s
    pixel_size_mm = cfg.get("pixel_size_mm", None)
    if pixel_size_mm is not None:
        vis["vel_scale"] = pixel_size_mm / dt / 1000.0  # mm/s -> m/s
        vis["vel_label"] = "Velocity (m/s)"
        print(f"Velocity conversion: {vis['vel_scale']:.2f} (mm/s) per (px/frame)")
    else:
        vis["vel_scale"] = 1.0
        vis["vel_label"] = "Magnitude (px/frame)"

    # Select evenly spaced frame indices across all available flow frames
    n_frames = min(n_frames, total_flow)
    if n_frames >= total_flow:
        frame_indices = list(range(total_flow))
    else:
        frame_indices = [int(round(i * (total_flow - 1) / (n_frames - 1)))
                         for i in range(n_frames)] if n_frames > 1 else [0]
    print(f"Generating {len(frame_indices)} frame set(s) from {total_flow} total "
          f"(indices: {frame_indices}), 4 figures each\n")

    # Load masks (once)
    mask_original = np.load(os.path.join(data_dir, "mask_original.npy"))

    # RAFT input mask (subset of pool used for RAFT input; falls back to full mask)
    mr_path = os.path.join(data_dir, "mask_raft.npy")
    mask_raft = np.load(mr_path) if os.path.exists(mr_path) else mask_original

    mask_corrected = None
    mc_path = os.path.join(data_dir, "mask_corrected.npy")
    if os.path.exists(mc_path):
        mask_corrected = np.load(mc_path)

    mask_symmetric = None
    ms_path = os.path.join(data_dir, "mask_symmetric.npy")
    if os.path.exists(ms_path):
        mask_symmetric = np.load(ms_path)

    # RAFT-region masks for corrected/symmetric (falls back to full masks)
    mrc_path = os.path.join(data_dir, "mask_raft_corrected.npy")
    mask_raft_corrected = np.load(mrc_path) if os.path.exists(mrc_path) else mask_corrected

    mrs_path = os.path.join(data_dir, "mask_raft_symmetric.npy")
    mask_raft_symmetric = np.load(mrs_path) if os.path.exists(mrs_path) else mask_symmetric

    # Load symmetry line y from calibration
    sym_y = None
    cal_path = os.path.join(output_dir, "calibration.npz")
    if os.path.exists(cal_path):
        cal = np.load(cal_path)
        sym_y = float(cal["symmetry_line_y"])

    if mask_corrected is None:
        print("WARNING: Corrected data not found. Figures 3 & 4 will be skipped.")
    if mask_symmetric is None:
        print("WARNING: Symmetric data not found. Figure 4 will be skipped.")

    # Generate sets
    for fi in frame_indices:
        generate_frame_set(
            fi, case_name, cfg, vis, data_dir, fig_dir,
            metadata, mask_original, mask_raft,
            mask_corrected, mask_symmetric,
            mask_raft_corrected, mask_raft_symmetric, sym_y)

    total_figs = len(frame_indices) * (2
                             + (1 if mask_corrected is not None else 0)
                             + (1 if mask_symmetric is not None else 0))
    print(f"\n{total_figs} figures saved to: {fig_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import argparse
    from config import get_case, FLOW_VIS

    parser = argparse.ArgumentParser(description="Step 4: Data illustration figures")
    parser.add_argument("--case", required=True, help="Case name (e.g., 0mm, 4mm)")
    parser.add_argument("--frames", type=int, default=1,
                        help="Number of frames to illustrate (default: 1)")
    args = parser.parse_args()

    cfg = get_case(args.case)
    vis = FLOW_VIS.copy()
    vis["arrow_scale"] *= cfg.get("arrow_scale_factor", 1.0)

    run_illustration(args.case, cfg, vis, args.frames)
