#!/usr/bin/env python3
"""
Step 3: Apply Calibration to All RAFT Flow Fields
===================================================
Loads calibration from step2 and applies homography + symmetrization
to all raw RAFT flow fields.

Usage:
    python step3_apply_correction.py --case 0mm
    python step3_apply_correction.py --case 4mm
"""

import numpy as np
import cv2
import os
from tqdm import tqdm
from scipy.ndimage import map_coordinates


# ── Flow correction functions ────────────────────────────────────────────────

def apply_homography_to_flow(flow, H, output_shape):
    """
    Apply homography correction to flow field with Jacobian correction.

    Parameters
    ----------
    flow : ndarray [2, H_in, W_in] - (u, v) optical flow
    H : ndarray [3, 3] - homography matrix
    output_shape : tuple (H_out, W_out)

    Returns
    -------
    flow_corrected : ndarray [2, H_out, W_out] float32
    """
    h_out, w_out = output_shape

    # Warp flow components
    flow_u_warped = cv2.warpPerspective(flow[0], H, (w_out, h_out),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
    flow_v_warped = cv2.warpPerspective(flow[1], H, (w_out, h_out),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)

    # Compute Jacobian for flow correction
    epsilon = 1.0
    H_inv = np.linalg.inv(H)

    y_out, x_out = np.mgrid[:h_out, :w_out].astype(np.float32)
    pts_out = np.stack([x_out.ravel(), y_out.ravel(), np.ones(h_out * w_out)], axis=0)
    pts_in = H_inv @ pts_out
    pts_in = pts_in[:2] / (pts_in[2:3] + 1e-10)

    pts_out_dx = pts_out.copy()
    pts_out_dx[0] += epsilon
    pts_in_dx = H_inv @ pts_out_dx
    pts_in_dx = pts_in_dx[:2] / (pts_in_dx[2:3] + 1e-10)

    pts_out_dy = pts_out.copy()
    pts_out_dy[1] += epsilon
    pts_in_dy = H_inv @ pts_out_dy
    pts_in_dy = pts_in_dy[:2] / (pts_in_dy[2:3] + 1e-10)

    J11 = ((pts_in_dx[0] - pts_in[0]) / epsilon).reshape(h_out, w_out)
    J12 = ((pts_in_dy[0] - pts_in[0]) / epsilon).reshape(h_out, w_out)
    J21 = ((pts_in_dx[1] - pts_in[1]) / epsilon).reshape(h_out, w_out)
    J22 = ((pts_in_dy[1] - pts_in[1]) / epsilon).reshape(h_out, w_out)

    det = J11 * J22 - J12 * J21 + 1e-10
    J11_inv = J22 / det
    J12_inv = -J12 / det
    J21_inv = -J21 / det
    J22_inv = J11 / det

    flow_u_corrected = J11_inv * flow_u_warped + J12_inv * flow_v_warped
    flow_v_corrected = J21_inv * flow_u_warped + J22_inv * flow_v_warped

    return np.stack([flow_u_corrected, flow_v_corrected], axis=0).astype(np.float32)


# ── Symmetrization functions ────────────────────────────────────────────────

def mirror_coordinates(x, y, y_center, slope, x_center):
    """Reflect (x, y) about line  y = y_center + slope * (x - x_center)."""
    m = slope
    d = y_center - m * x_center
    t = (m * x - y + d) / (m ** 2 + 1.0)
    x_mirror = x - 2.0 * m * t
    y_mirror = y + 2.0 * t
    return x_mirror, y_mirror


def reflect_flow_vectors(u, v, slope):
    """Reflect flow vectors about a line with given slope."""
    m = slope
    m2 = m * m
    d = 1.0 + m2
    R00 = (1.0 - m2) / d
    R01 = 2.0 * m / d
    R10 = 2.0 * m / d
    R11 = (m2 - 1.0) / d
    return R00 * u + R01 * v, R10 * u + R11 * v


def symmetrize_flow_and_mask(flow, mask_bool, y_center, slope, x_center):
    """
    Mirror the bottom half of the flow field onto the top half.

    Parameters
    ----------
    flow       : ndarray [2, H, W] float
    mask_bool  : ndarray [H, W] bool
    y_center   : float - y of central line at x = x_center
    slope      : float - tangent of line tilt angle
    x_center   : float - reference x

    Returns
    -------
    sym_flow : ndarray [2, H, W] float32
    sym_mask : ndarray [H, W] bool
    """
    h, w = flow.shape[1], flow.shape[2]
    yc, xc = np.mgrid[0:h, 0:w].astype(np.float64)

    y_line = y_center + slope * (xc - x_center)
    above = yc < y_line

    x_mir, y_mir = mirror_coordinates(xc, yc, y_center, slope, x_center)
    x_mir = np.clip(x_mir, 0, w - 1)
    y_mir = np.clip(y_mir, 0, h - 1)

    sym_flow = flow.copy().astype(np.float64)
    for c in range(2):
        vals = map_coordinates(flow[c].astype(np.float64),
                               [y_mir[above], x_mir[above]],
                               order=1, mode='nearest')
        sym_flow[c][above] = vals

    u_ab, v_ab = sym_flow[0][above].copy(), sym_flow[1][above].copy()
    u_ref, v_ref = reflect_flow_vectors(u_ab, v_ab, slope)
    sym_flow[0][above] = u_ref
    sym_flow[1][above] = v_ref

    sym_mask = mask_bool.copy()
    m_vals = map_coordinates(mask_bool.astype(np.float64),
                             [y_mir[above], x_mir[above]],
                             order=0, mode='constant', cval=0)
    sym_mask[above] = m_vals > 0.5

    return sym_flow.astype(np.float32), sym_mask


# ── Main processing ──────────────────────────────────────────────────────────

def apply_corrections(output_dir):
    """
    Apply calibration (homography + symmetry) to all raw RAFT flows.

    Expects:
        {output_dir}/calibration.npz
        {output_dir}/RAFT_data/flow_raw/
        {output_dir}/RAFT_data/mask_original.npy

    Creates:
        {output_dir}/RAFT_data/flow_corrected/
        {output_dir}/RAFT_data/flow_symmetric/
        {output_dir}/RAFT_data/mask_corrected.npy
        {output_dir}/RAFT_data/mask_symmetric.npy
    """
    data_dir = os.path.join(output_dir, "RAFT_data")

    # Load calibration
    cal_path = os.path.join(output_dir, "calibration.npz")
    cal = np.load(cal_path)
    H = cal["H"]
    sym_y = float(cal["symmetry_line_y"])
    sym_slope = float(cal["symmetry_slope"])
    sym_x_center = float(cal["symmetry_x_center"])
    image_shape = tuple(cal["image_shape"])
    h, w = image_shape

    print(f"Loaded calibration from {cal_path}")
    print(f"  Image: {w}x{h}")
    print(f"  Symmetry line: y={sym_y:.1f}, slope={sym_slope:.4f}")

    # Load original mask and RAFT mask
    mask_original = np.load(os.path.join(data_dir, "mask_original.npy"))
    mr_path = os.path.join(data_dir, "mask_raft.npy")
    mask_raft = np.load(mr_path) if os.path.exists(mr_path) else mask_original

    # Compute corrected masks (full pool + RAFT region)
    mask_corrected_img = cv2.warpPerspective(
        mask_original.astype(np.uint8) * 255, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    mask_corrected = mask_corrected_img > 127

    mask_raft_corr_img = cv2.warpPerspective(
        mask_raft.astype(np.uint8) * 255, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    mask_raft_corrected = mask_raft_corr_img > 127

    # Compute symmetric masks
    dummy_flow = np.zeros((2, h, w), dtype=np.float32)
    _, mask_symmetric = symmetrize_flow_and_mask(
        dummy_flow, mask_corrected, sym_y, sym_slope, sym_x_center
    )
    _, mask_raft_symmetric = symmetrize_flow_and_mask(
        dummy_flow, mask_raft_corrected, sym_y, sym_slope, sym_x_center
    )

    print(f"  Original mask:       {mask_original.sum()} valid pixels")
    print(f"  RAFT mask:           {mask_raft.sum()} valid pixels")
    print(f"  Corrected mask:      {mask_corrected.sum()} valid pixels")
    print(f"  RAFT corr. mask:     {mask_raft_corrected.sum()} valid pixels")
    print(f"  Symmetric mask:      {mask_symmetric.sum()} valid pixels")
    print(f"  RAFT symm. mask:     {mask_raft_symmetric.sum()} valid pixels")

    # Save masks
    np.save(os.path.join(data_dir, "mask_corrected.npy"), mask_corrected)
    np.save(os.path.join(data_dir, "mask_symmetric.npy"), mask_symmetric)
    np.save(os.path.join(data_dir, "mask_raft_corrected.npy"), mask_raft_corrected)
    np.save(os.path.join(data_dir, "mask_raft_symmetric.npy"), mask_raft_symmetric)

    # Process flows
    flow_raw_dir = os.path.join(data_dir, "flow_raw")
    flow_corr_dir = os.path.join(data_dir, "flow_corrected")
    flow_sym_dir = os.path.join(data_dir, "flow_symmetric")
    os.makedirs(flow_corr_dir, exist_ok=True)
    os.makedirs(flow_sym_dir, exist_ok=True)

    flow_files = sorted(f for f in os.listdir(flow_raw_dir) if f.endswith(".npy"))
    if not flow_files:
        raise FileNotFoundError(f"No .npy files found in {flow_raw_dir}")

    print(f"\nProcessing {len(flow_files)} flow fields...")

    for fname in tqdm(flow_files, desc="Correcting flows"):
        flow_raw = np.load(os.path.join(flow_raw_dir, fname))

        # Homography + Jacobian correction
        flow_corrected = apply_homography_to_flow(flow_raw, H, (h, w))
        flow_corrected[:, ~mask_raft_corrected] = 0  # zero out outside RAFT region
        np.save(os.path.join(flow_corr_dir, fname), flow_corrected)

        # Symmetrization
        flow_symmetric, _ = symmetrize_flow_and_mask(
            flow_corrected, mask_corrected, sym_y, sym_slope, sym_x_center
        )
        flow_symmetric[:, ~mask_raft_symmetric] = 0  # zero out outside RAFT region
        np.save(os.path.join(flow_sym_dir, fname), flow_symmetric)

    # Statistics on first frame
    sample_corr = np.load(os.path.join(flow_corr_dir, flow_files[0]))
    mag_corr = np.sqrt(sample_corr[0] ** 2 + sample_corr[1] ** 2)

    sample_sym = np.load(os.path.join(flow_sym_dir, flow_files[0]))
    mag_sym = np.sqrt(sample_sym[0] ** 2 + sample_sym[1] ** 2)

    print(f"\n{'=' * 60}")
    print(f"Correction complete - {len(flow_files)} frames")
    print(f"  flow_corrected/  max magnitude: {mag_corr[mask_corrected].max():.2f} px/frame")
    print(f"  flow_symmetric/  max magnitude: {mag_sym[mask_symmetric].max():.2f} px/frame")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import argparse
    from config import get_case

    parser = argparse.ArgumentParser(description="Step 3: Apply correction and symmetrization")
    parser.add_argument("--case", required=True, help="Case name (e.g., 0mm, 4mm)")
    args = parser.parse_args()

    cfg = get_case(args.case)
    apply_corrections(output_dir=cfg["output_dir"])
