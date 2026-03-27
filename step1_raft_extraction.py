#!/usr/bin/env python3
"""
Step 1: RAFT Optical Flow Extraction
=====================================
Extracts dense optical flow from weld pool video using RAFT
and saves per-frame flow fields to disk.

Usage:
    python step1_raft_extraction.py --case 0mm
    python step1_raft_extraction.py --case 2mm --start 00-14-33 --end 00-22-10
    python step1_raft_extraction.py --case 4mm --start 00-14-33 --end 00-22-10

Time format:  HH-MM-SS  (hours-minutes-seconds, dashes as separators)
              MM-SS     (minutes-seconds)
When --start/--end are given they override the n_frames limit and process
the full clip between those two timestamps at the configured stride.
"""

import torch
import cv2
import numpy as np
import os
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from tqdm import tqdm


def preprocess(img, device):
    """BGR uint8 frame -> [1, 3, H, W] float32 tensor on device, range [0,1]."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (torch.from_numpy(img).permute(2, 0, 1)
            .float().unsqueeze(0).to(device) / 255.0)


def build_raft_mask(pool_mask_bool, mask_region="full"):
    """
    Build the mask used to zero-out input frames before RAFT.

    Parameters
    ----------
    pool_mask_bool : ndarray [H, W] bool
    mask_region : str
        "full"       - entire pool mask
        "left_half"  - left half of the pool mask (split at horizontal midpoint)

    Returns
    -------
    raft_mask : ndarray [H, W] bool
    """
    if mask_region == "full":
        return pool_mask_bool.copy()

    if mask_region == "left_half":
        cols = np.where(pool_mask_bool.any(axis=0))[0]
        mid_x = (cols[0] + cols[-1]) // 2
        raft_mask = pool_mask_bool.copy()
        raft_mask[:, mid_x:] = False
        return raft_mask

    raise ValueError(f"Unknown raft_mask_region: '{mask_region}'")


def _parse_time_str(t: str) -> float:
    """Convert 'HH-MM-SS' or 'MM-SS' to total seconds (float)."""
    parts = t.strip().split("-")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    raise ValueError(f"Invalid time '{t}'. Expected HH-MM-SS or MM-SS.")


def _save_verification_clip(cap, f0, f1, h, w, output_dir, out_fps=15):
    """
    Save the full video clip from frame f0 to f1 for visual verification.

    Parameters
    ----------
    cap : cv2.VideoCapture  (already opened)
    f0, f1 : int            start / end frame indices
    h, w : int              frame dimensions
    output_dir : str        base output directory
    out_fps : float         playback frame rate of the saved clip
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "clip_verification.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(path, fourcc, out_fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    n_written = 0
    for _ in range(f1 - f0):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        n_written += 1
    writer.release()
    print(f"  Saved verification clip: {path}  (frames {f0}-{f1}, {n_written} frames, {n_written/out_fps:.1f}s at {out_fps} fps)")


def run_flow_detection(video_path, mask_path, output_dir, stride=5, n_frames=100,
                       time_start=None, time_end=None, raft_mask_region="full"):
    """
    Extract RAFT optical flow from video and save per-frame flow fields.

    Frame collection modes:
      - time_start + time_end given : seek to that clip, collect all frames
                                      at `stride` spacing (n_frames ignored)
      - otherwise                   : collect up to n_frames from the start

    Creates:
        {output_dir}/RAFT_data/flow_raw/0000.npy ... NNNN.npy  -- [2,H,W] float32
        {output_dir}/RAFT_data/mask_original.npy                -- bool [H,W]
        {output_dir}/RAFT_data/metadata.npz
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load mask
    pool_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if pool_mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    pool_mask_bool = pool_mask > 127
    h_img, w_img = pool_mask.shape
    print(f"Mask: {w_img}x{h_img},  {pool_mask_bool.sum()} px")

    # Build RAFT input mask (may be a subset of pool mask)
    raft_mask = build_raft_mask(pool_mask_bool, raft_mask_region)
    raft_mask_3ch = raft_mask[:, :, np.newaxis].astype(np.uint8)  # [H, W, 1] for broadcasting
    print(f"RAFT input mask: region='{raft_mask_region}', {raft_mask.sum()} px")

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total} frames @ {fps:.1f} FPS")

    frames = []

    if time_start is not None and time_end is not None:
        t0 = _parse_time_str(time_start)
        t1 = _parse_time_str(time_end)
        f0 = int(round(t0 * fps))
        f1 = int(round(t1 * fps))
        clip_frames = f1 - f0
        expected = max(0, clip_frames // stride)
        print(f"Time range : {time_start}  ->  {time_end}")
        print(f"Frame range: {f0}  ->  {f1}  ({clip_frames} frames,  ~{expected} sampled at stride={stride})")

        # Save verification clips at start and end of the time range
        _save_verification_clip(cap, f0, f1, h_img, w_img, output_dir)

        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        local = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if local % stride == 0:
                frames.append(frame)
                if len(frames) >= n_frames:
                    break
            local += 1
            if f0 + local > f1:
                break
    else:
        idx = 0
        while len(frames) < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                frames.append(frame)
            idx += 1

        # Save verification clip covering the collected frame range
        f0 = 0
        f1 = idx
        _save_verification_clip(cap, f0, f1, h_img, w_img, output_dir)

    cap.release()
    print(f"Collected {len(frames)} frames (stride={stride})")

    # RAFT model
    model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()

    # Output directory
    data_dir = os.path.join(output_dir, "RAFT_data")
    flow_dir = os.path.join(data_dir, "flow_raw")
    os.makedirs(flow_dir, exist_ok=True)

    dt = stride / fps

    # RAFT inference
    n_pairs = len(frames) - 1
    print(f"\nRunning RAFT on {n_pairs} frame pairs ...")
    with torch.no_grad():
        for i in tqdm(range(n_pairs)):
            img1 = preprocess(frames[i], device)
            img2 = preprocess(frames[i + 1], device)
            flow_raw = model(img1, img2)[-1][0].cpu().numpy()  # [2, H, W]
            np.save(os.path.join(flow_dir, f"{i:04d}.npy"), flow_raw)

            if i < 3:
                mag = np.sqrt(flow_raw[0] ** 2 + flow_raw[1] ** 2)
                print(f"  Frame {i}: max magnitude = {mag[pool_mask_bool].max():.2f} px/frame "
                      f"({mag[pool_mask_bool].max() * fps / stride:.1f} px/s)")

    # Save masks & metadata
    np.save(os.path.join(data_dir, "mask_original.npy"), pool_mask_bool)
    np.save(os.path.join(data_dir, "mask_raft.npy"), raft_mask)
    np.savez(os.path.join(data_dir, "metadata.npz"),
             fps=fps, stride=stride, dt=dt,
             n_flow_frames=n_pairs,
             image_shape=np.array([h_img, w_img]))

    print(f"\n{'=' * 55}")
    print(f"Flow detection complete - {n_pairs} frames -> {data_dir}/")
    print(f"  flow_raw/         {n_pairs} files")
    print(f"  mask_original.npy")
    print(f"  metadata.npz")
    print(f"  dt = {dt * 1000:.2f} ms/frame")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    import argparse
    from config import get_case

    parser = argparse.ArgumentParser(description="Step 1: RAFT optical flow extraction")
    parser.add_argument("--case",  required=True,
                        help="Case name (e.g., 0mm, 2mm, 4mm)")
    parser.add_argument("--start", default=None, metavar="HH-MM-SS",
                        help="Start time in the video (e.g. 00-14-33). "
                             "Overrides n_frames when combined with --end.")
    parser.add_argument("--end",   default=None, metavar="HH-MM-SS",
                        help="End time in the video (e.g. 00-22-10). "
                             "Overrides n_frames when combined with --start.")
    args = parser.parse_args()

    cfg = get_case(args.case)

    # CLI --start/--end take priority; fall back to whatever is in config
    time_start = args.start or cfg.get("time_start")
    time_end   = args.end   or cfg.get("time_end")

    run_flow_detection(
        video_path=cfg["video_path"],
        mask_path=cfg["mask_path"],
        output_dir=cfg["output_dir"],
        stride=cfg["stride"],
        n_frames=cfg["n_frames"],
        time_start=time_start,
        time_end=time_end,
        raft_mask_region=cfg.get("raft_mask_region", "full"),
    )
