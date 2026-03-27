# %% [Cell 0]
# ============================================================================
# step5_4mm_PINN.py
# Phase-conditioned PINN for transient flow field after droplet impact
# ============================================================================
# This code learns a time-varying smooth flow field from noisy RAFT frames
# covering one period after a molten wire droplet impacts the weld pool.
#
# Key differences from the quasi-static (0mm) version:
#   - The network input is (x_tilde, y_tilde, t_hat) where t_hat in [0,1]
#     represents the normalized phase within the observation window.
#   - Each RAFT frame has an associated t_hat value.
#   - The N-S equation includes the unsteady term: du/dt.
#   - No explicit impact force is modelled — the RAFT frames are AFTER droplet
#     impact, so the flow is decaying. The physics captures this via the
#     unsteady N-S equation with viscous dissipation.
#   - An energy decay regularizer encourages monotonically decreasing kinetic
#     energy over time (physical: no energy source after impact).
#   - Arc force is retained as the welding arc is still active during the
#     observation window.
#   - Frame-level confidence weighting: frames are scored by spatial coherence
#     and only the best ~10-20 frames contribute strongly.
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# %% [Cell 1]
# ── Physical constants ──
PIXEL_SIZE_MM = 10.53 / 512
PIXEL_SIZE_M  = PIXEL_SIZE_MM * 1e-3
FPS           = 1500
STRIDE        = 5
DT            = STRIDE / FPS           # time between consecutive RAFT frames (s)
RHO           = 7200.0                  # density of molten steel (kg/m^3)

VEL_FACTOR = PIXEL_SIZE_M / DT

print(f"Pixel size : {PIXEL_SIZE_M:.4e} m/px  ({PIXEL_SIZE_MM:.4f} mm/px)")
print(f"Frame rate : {FPS} Hz,  stride = {STRIDE},  dt = {DT:.6f} s")
print(f"Vel factor : {VEL_FACTOR:.4e} m/s per (px/stride)")
print(f"Density    : {RHO:.0f} kg/m^3")

data_dir = 'output_2mm/RAFT_data'
result_dir = 'PINN_results_2mm_A_KE1'
arc_x_frac_init = 0.633

# data_dir = 'output_4mm/RAFT_data'
# result_dir = 'PINN_results_4mm_A_KE1'
# arc_x_frac_init = 0.647


# ── PINN configuration ──
CONFIG = {
    'data_dir':              data_dir,
    'results_dir':           result_dir,
    'crop_padding':          10,

    # Frame selection: keep frames whose spatial coherence score
    # exceeds this fraction of the best frame's score
    'frame_coherence_threshold': 0.3,

    # Boundary band
    'boundary_band_width':     50,
    'boundary_inner_fraction': 0.5,
    'boundary_outer_fraction': 0.5,

    # Sampling per epoch
    'n_obs_per_step':        2000,
    'n_pde_per_step':        1500,
    'n_boundary_points':     800,

    # Loss weights
    'lambda_data':           5.0,
    'lambda_boundary':       2.0,
    'lambda_pde':            1.0,
    'lambda_p_ref':          0.5,
    'lambda_symmetry':       2.0,
    'lambda_mirror_symmetry': 5.0,   # 4mm case: 5 # penalize u(x,y)!=u(x,-y) and v(x,y)!=-v(x,-y)
    'lambda_energy_decay':   1,
    'lambda_temporal_smooth': 0.1*0,


    # Arc force (arc is still on during observation)
    'disable_arc_force':     False,

    # ── Arc parameters estimated from high-speed images ──
    # Position: electrode center as fraction of centerline width (left to right)
    'arc_x_frac_init':       arc_x_frac_init,
    # Width: visible arc radius estimated from images (meters)
    'arc_sigma_init_m':      2.5e-3,     # ~2.5 mm from image

    # Amplitude: rough estimate with wide tolerance
    'arc_amp_init':          1.0,        # arbitrary init, network will learn
    'arc_amp_prior_Nm3':     32463,        # rough prior ~30,000 N/m^3
    'lambda_arc_amp_prior':  1.0,        # soft constraint
    'arc_amp_prior_log_std': 0.4,        # log(1.5)≈0.405, allows ±50% variation

    # Soft priors on position and width
    'lambda_arc_pos_prior':  2.0,        # constrain position to image estimate
    'arc_x_frac_measured':   arc_x_frac_init,
    'arc_x_frac_tolerance':  0.05,       # ±5% of centerline width
    'lambda_arc_sigma_prior': 1.0,       # constrain width to image estimate
    'arc_sigma_prior_log_std': 0.3,      # allows ~1.35x variation from image estimate

    # Training
    'n_epochs':              1000,
    'lr':                    1e-3,
    'lr_physics':            5e-3,
    'hidden_dims':           [128, 256, 256, 128],
    'lambda_peak':           0.1,
}


# %% [Cell 2]
# ============================================================================
# Data Loading with per-frame timing and confidence scoring
# ============================================================================

def compute_frame_coherence(u_frame, v_frame, mask, kernel_size=5):
    """Score a frame's spatial coherence using local flow consistency.

    A noisy frame has random directions pixel-to-pixel -> low coherence.
    A coherent frame has smooth, consistent flow -> high coherence.
    Returns a scalar score in [0, 1].
    """
    mag = np.sqrt(u_frame**2 + v_frame**2)
    active = mask & (mag > 1e-6)
    if active.sum() < 10:
        return 0.0

    # Compute local angular consistency using a box filter
    # Normalize flow direction
    eps = 1e-8
    dx = np.where(active, u_frame / (mag + eps), 0.0).astype(np.float32)
    dy = np.where(active, v_frame / (mag + eps), 0.0).astype(np.float32)

    k = kernel_size
    kernel = np.ones((k, k), np.float32) / (k * k)
    mean_dx = cv2.filter2D(dx, -1, kernel)
    mean_dy = cv2.filter2D(dy, -1, kernel)

    # Local mean direction magnitude (1 = perfectly coherent, 0 = random)
    coherence_map = np.sqrt(mean_dx**2 + mean_dy**2)
    return float(coherence_map[active].mean())


def load_data(cfg):
    """Load RAFT data with per-frame timing, coherence scoring, and
    sparse observation extraction."""
    data_dir = cfg['data_dir']

    # ── Mask ──
    mask = np.load(os.path.join(data_dir, 'mask_symmetric.npy'))
    print(f"Mask: {mask.shape}, {mask.sum()} active pixels")

    # ── Crop to mask bounding box ──
    rows, cols = np.where(mask)
    pad = cfg['crop_padding']
    r0 = max(0, rows.min() - pad)
    r1 = min(mask.shape[0], rows.max() + pad)
    c0 = max(0, cols.min() - pad)
    c1 = min(mask.shape[1], cols.max() + pad)
    mask_crop = mask[r0:r1, c0:c1]
    h_c, w_c = mask_crop.shape
    print(f"Crop: [{r0}:{r1}, {c0}:{c1}] -> {w_c} x {h_c}")

    # ── Pool center ──
    rows_m, cols_m = np.where(mask_crop)
    cx_px = (cols_m.min() + cols_m.max()) / 2.0
    cy_px = (rows_m.min() + rows_m.max()) / 2.0

    # ── Pool dimensions ──
    W_px = cols_m.max() - cols_m.min()
    H_px = rows_m.max() - rows_m.min()
    W_m = W_px * PIXEL_SIZE_M
    H_m = H_px * PIXEL_SIZE_M
    L = max(W_m, H_m)

    print(f"Pool center (crop px): cx={cx_px:.1f}, cy={cy_px:.1f}")
    print(f"Pool size: {W_m*1e3:.2f} mm x {H_m*1e3:.2f} mm")
    print(f"Characteristic length L = {L*1e3:.2f} mm = {L:.4e} m")

    # ── Load flow frames ──
    flow_dir = os.path.join(data_dir, 'flow_symmetric')
    flow_files = sorted(glob.glob(os.path.join(flow_dir, '*.npy')))
    n_frames = len(flow_files)
    print(f"Flow frames: {n_frames}")

    # ── Total observation time window ──
    T_total = (n_frames - 1) * DT  # seconds
    T_char = T_total if T_total > 0 else 1.0  # characteristic time
    print(f"Observation window: {T_total*1e3:.2f} ms  ({n_frames} frames)")

    # ── First pass: load all frames, compute coherence scores ──
    all_flow = np.zeros((n_frames, 2, h_c, w_c), dtype=np.float32)
    coherence_scores = np.zeros(n_frames)

    for i in tqdm(range(n_frames), desc="Loading & scoring"):
        f = np.load(flow_files[i])[:, r0:r1, c0:c1]
        all_flow[i] = f * VEL_FACTOR  # -> m/s
        coherence_scores[i] = compute_frame_coherence(
            all_flow[i, 0], all_flow[i, 1], mask_crop)

    # ── Frame confidence weights from coherence ──
    max_coherence = coherence_scores.max()
    if max_coherence > 0:
        frame_weights = coherence_scores / max_coherence
    else:
        frame_weights = np.ones(n_frames)

    threshold = cfg['frame_coherence_threshold']
    n_good = (frame_weights >= threshold).sum()
    print(f"Coherence scores: min={coherence_scores.min():.3f}, "
          f"max={coherence_scores.max():.3f}")
    print(f"Frames with coherence >= {threshold:.1f} of max: {n_good}/{n_frames}")

    # ── Per-frame sparse extraction with timing ──
    sparse_ix_list = []
    sparse_iy_list = []
    sparse_u_list  = []
    sparse_v_list  = []
    sparse_t_list  = []       # normalized time t_hat in [0, 1]
    sparse_w_list  = []       # per-point confidence weight
    global_max_speed = 0.0
    speed_range_pct = 0.7     # keep top 30% of speed dynamic range per frame

    # Track which frames actually contributed observations (for recon error)
    selected_frame_indices = []

    for i in range(n_frames):
        if frame_weights[i] < threshold:
            continue  # skip very noisy frames

        mag = np.sqrt(all_flow[i, 0]**2 + all_flow[i, 1]**2)
        nontrivial = mask_crop & (mag > 1e-6)
        if nontrivial.sum() == 0:
            continue

        speeds_nt = mag[nontrivial]
        speed_lo, speed_hi = speeds_nt.min(), speeds_nt.max()
        if speed_hi > global_max_speed:
            global_max_speed = speed_hi

        speed_thresh = speed_lo + speed_range_pct * (speed_hi - speed_lo)
        valid = nontrivial & (mag >= speed_thresh)
        if valid.sum() == 0:
            continue

        # Frame passed all filters and contributed observations
        selected_frame_indices.append(i)

        vy, vx = np.where(valid)
        n_pts = len(vx)
        t_hat = i / max(n_frames - 1, 1)  # normalized time [0, 1]

        sparse_ix_list.append(vx)
        sparse_iy_list.append(vy)
        sparse_u_list.append(all_flow[i, 0, vy, vx])
        sparse_v_list.append(all_flow[i, 1, vy, vx])
        sparse_t_list.append(np.full(n_pts, t_hat, dtype=np.float32))
        sparse_w_list.append(np.full(n_pts, frame_weights[i], dtype=np.float32))

    # ── Concatenate ──
    sparse_ix = np.concatenate(sparse_ix_list)
    sparse_iy = np.concatenate(sparse_iy_list)
    sparse_u  = np.concatenate(sparse_u_list)
    sparse_v  = np.concatenate(sparse_v_list)
    sparse_t  = np.concatenate(sparse_t_list)
    sparse_w  = np.concatenate(sparse_w_list)

    U = float(global_max_speed) if global_max_speed > 0 else 1e-5

    # Normalized coordinates and dimensionless velocity
    sparse_xt    = (sparse_ix - cx_px) * PIXEL_SIZE_M / L
    sparse_yt    = (sparse_iy - cy_px) * PIXEL_SIZE_M / L
    sparse_u_hat = sparse_u / U
    sparse_v_hat = sparse_v / U

    print(f"Characteristic velocity U = {U:.4f} m/s")
    print(f"Characteristic time T = {T_char*1e3:.2f} ms")
    print(f"Sparse observations: {len(sparse_ix)} total from "
          f"{n_good} usable frames")

    # ── Interior points (spatial, no time — used for PDE collocation) ──
    iy, ix = np.where(mask_crop)
    x_tilde = (ix - cx_px) * PIXEL_SIZE_M / L
    y_tilde = (iy - cy_px) * PIXEL_SIZE_M / L
    interior_coords = np.stack([x_tilde, y_tilde], axis=1)

    # ── Centerline points ──
    cl_tol = 4.0 * PIXEL_SIZE_M / L
    cl_mask = np.abs(y_tilde) < cl_tol
    cl_pts = interior_coords[cl_mask]
    cl_x_min = float(cl_pts[:, 0].min()) if len(cl_pts) > 0 else -0.5
    cl_x_max = float(cl_pts[:, 0].max()) if len(cl_pts) > 0 else  0.5

    # ── Contour for visualization ──
    contours, _ = cv2.findContours(mask_crop.astype(np.uint8),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea).squeeze().astype(float)

    return {
        'mask_crop': mask_crop, 'crop': (r0, r1, c0, c1),
        'h_crop': h_c, 'w_crop': w_c,
        'cx_px': cx_px, 'cy_px': cy_px,
        'L': L, 'U': U, 'T_char': T_char,
        'n_frames': n_frames,
        'frame_weights': frame_weights,
        'all_flow': all_flow,
        'selected_frame_indices': selected_frame_indices,  # for recon error
        # Sparse observations with time
        'sparse_xt': sparse_xt, 'sparse_yt': sparse_yt,
        'sparse_t': sparse_t,
        'sparse_u_hat': sparse_u_hat, 'sparse_v_hat': sparse_v_hat,
        'sparse_w': sparse_w,
        # For PDE collocation
        'interior_coords': interior_coords,
        'interior_ix': ix, 'interior_iy': iy,
        'cl_pts': cl_pts,
        'cl_x_min': cl_x_min, 'cl_x_max': cl_x_max,
        'bnd_cnt_px': cnt,
    }

data = load_data(CONFIG)


# %% [Cell 3]
# ── Soft Boundary Band ──

def create_boundary_band(mask, band_width, inner_frac, outer_frac,
                          cx_px, cy_px, L):
    """Soft boundary band in normalized coordinates."""
    dist_inside  = distance_transform_edt(mask)
    dist_outside = distance_transform_edt(~mask)
    signed_dist  = dist_outside - dist_inside

    in_band = np.abs(signed_dist) <= band_width
    by, bx = np.where(in_band)

    distances = np.abs(signed_dist[by, bx])
    inner_threshold = band_width * inner_frac
    weights = np.ones(len(distances))
    outer = distances > inner_threshold
    weights[outer] = 1.0 - (distances[outer] - inner_threshold) / \
                            (band_width - inner_threshold)
    weights = np.clip(weights, 0.0, 1.0)

    grad_y, grad_x = np.gradient(signed_dist)
    normals = np.stack([grad_x[by, bx], grad_y[by, bx]], axis=1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    normals = normals / norms

    x_tilde = (bx - cx_px) * PIXEL_SIZE_M / L
    y_tilde = (by - cy_px) * PIXEL_SIZE_M / L
    pts = np.stack([x_tilde, y_tilde], axis=1)

    print(f"Boundary band: {len(bx)} pts, weight mean={weights.mean():.3f}")
    return pts, normals, weights

bnd_pts, bnd_normals, bnd_weights = create_boundary_band(
    data['mask_crop'],
    CONFIG['boundary_band_width'],
    CONFIG['boundary_inner_fraction'],
    CONFIG['boundary_outer_fraction'],
    data['cx_px'], data['cy_px'], data['L'])


# %% [Cell 4]
# ============================================================================
# Neural Network: Phase-conditioned FlowNet
# ============================================================================
# Input: (x_tilde, y_tilde, t_hat) where t_hat in [0,1] is normalized time
# Output: (psi_hat, p_hat) — dimensionless stream function and pressure
# ============================================================================

class FlowNet(nn.Module):
    """MLP: (x_tilde, y_tilde, t_hat) -> (psi_hat, p_hat)  [dimensionless].

    The stream function psi ensures divergence-free velocity at each time:
        u_hat =  d(psi_hat)/d(y_tilde)
        v_hat = -d(psi_hat)/d(x_tilde)
    """

    def __init__(self, hidden_dims):
        super().__init__()
        layers = []
        in_dim = 3  # (x, y, t)
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))  # psi_hat, p_hat
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xyt):
        """xyt: (N, 3) with columns [x_tilde, y_tilde, t_hat]."""
        out = self.net(xyt)
        return out[:, 0:1], out[:, 1:2]


def get_velocity_hat(model, xyt):
    """Dimensionless velocity from stream function.
    Requires grad w.r.t. x_tilde and y_tilde (columns 0, 1).

    u_hat =  d(psi_hat)/d(y_tilde)
    v_hat = -d(psi_hat)/d(x_tilde)
    """
    xyt = xyt.clone().requires_grad_(True)
    psi_hat, p_hat = model(xyt)
    grad = torch.autograd.grad(
        psi_hat, xyt, torch.ones_like(psi_hat), create_graph=True)[0]
    u_hat =  grad[:, 1:2]   # d_psi / d_y
    v_hat = -grad[:, 0:1]   # -d_psi / d_x
    return u_hat, v_hat, p_hat, xyt


# %% [Cell 5]
# ── Arc Force (unchanged — arc is still active) ──

class ArcForce(nn.Module):
    """Gaussian body force in normalized coordinates."""

    def __init__(self, x_frac_init, sigma_init, amp_init,
                 y_center, x_min, x_max):
        super().__init__()
        frac = float(np.clip(x_frac_init, 0.01, 0.99))
        self.raw_x     = nn.Parameter(torch.tensor(
            np.log(frac / (1 - frac)), dtype=torch.float32))
        self.log_sigma = nn.Parameter(torch.tensor(
            np.log(float(sigma_init)), dtype=torch.float32))
        self.log_amp   = nn.Parameter(torch.tensor(
            np.log(float(amp_init)),   dtype=torch.float32))
        self.y_center = float(y_center)
        self.x_min    = float(x_min)
        self.x_max    = float(x_max)

    def _x_arc(self):
        return self.x_min + (self.x_max - self.x_min) * torch.sigmoid(self.raw_x)

    def forward(self, xy):
        """xy: (N, 2) or first two columns of (N, 3) — spatial only."""
        x_arc = self._x_arc()
        sigma = torch.exp(self.log_sigma)
        amp   = torch.exp(self.log_amp)
        dx = xy[:, 0:1] - x_arc
        dy = xy[:, 1:2] - self.y_center
        r2 = dx**2 + dy**2
        envelope = amp * torch.exp(-r2 / (2 * sigma**2))
        return envelope * dx / sigma**2, envelope * dy / sigma**2

    def get_params(self):
        with torch.no_grad():
            return (self._x_arc().item(),
                    torch.exp(self.log_sigma).item(),
                    torch.exp(self.log_amp).item())


# %% [Cell 6]
# ============================================================================
# Unsteady Navier-Stokes Residual (Dimensionless)
# ============================================================================
# The dimensionless unsteady incompressible N-S:
#
#   St * d(u_hat)/d(t_hat)
#   + u_hat * d(u_hat)/d(x_tilde) + v_hat * d(u_hat)/d(y_tilde)
#   = -d(p_hat)/d(x_tilde) + nu_tilde * laplacian(u_hat) + F_tilde_x
#
# where St = L / (U * T) is the Strouhal number (unsteady scaling).
# For our normalization: t_hat = t / T, x_tilde = x / L, u_hat = u / U.
# ============================================================================

def compute_ns_residual_unsteady(model, arc_force, xyt, log_nu_hat, St, cfg):
    """Dimensionless unsteady N-S residual at collocation points.

    xyt: (N, 3) tensor with [x_tilde, y_tilde, t_hat]
    St:  Strouhal number L/(U*T)
    """
    nu_hat = torch.exp(log_nu_hat)
    xyt = xyt.clone().requires_grad_(True)
    psi_hat, p_hat = model(xyt)
    ones = torch.ones_like(psi_hat)

    # Gradients of psi w.r.t. (x, y, t)
    g_psi = torch.autograd.grad(psi_hat, xyt, ones,
                                create_graph=True, retain_graph=True)[0]
    dpsi_dx = g_psi[:, 0:1]
    dpsi_dy = g_psi[:, 1:2]
    dpsi_dt = g_psi[:, 2:3]

    u_hat =  dpsi_dy    # d_psi / d_y
    v_hat = -dpsi_dx    # -d_psi / d_x

    # Pressure gradient
    g_p = torch.autograd.grad(p_hat, xyt, ones,
                              create_graph=True, retain_graph=True)[0]
    dp_dx, dp_dy = g_p[:, 0:1], g_p[:, 1:2]

    # Velocity gradients
    g_u = torch.autograd.grad(u_hat, xyt, ones,
                              create_graph=True, retain_graph=True)[0]
    du_dx, du_dy, du_dt = g_u[:, 0:1], g_u[:, 1:2], g_u[:, 2:3]

    g_v = torch.autograd.grad(v_hat, xyt, ones,
                              create_graph=True, retain_graph=True)[0]
    dv_dx, dv_dy, dv_dt = g_v[:, 0:1], g_v[:, 1:2], g_v[:, 2:3]

    # Second spatial derivatives
    d2u_dx2 = torch.autograd.grad(du_dx, xyt, ones,
                                  create_graph=True, retain_graph=True)[0][:, 0:1]
    d2u_dy2 = torch.autograd.grad(du_dy, xyt, ones,
                                  create_graph=True, retain_graph=True)[0][:, 1:2]
    d2v_dx2 = torch.autograd.grad(dv_dx, xyt, ones,
                                  create_graph=True, retain_graph=True)[0][:, 0:1]
    d2v_dy2 = torch.autograd.grad(dv_dy, xyt, ones,
                                  create_graph=True)[0][:, 1:2]

    # Arc force (spatial only)
    if cfg.get('disable_arc_force', False):
        Fx = torch.zeros_like(u_hat)
        Fy = torch.zeros_like(u_hat)
    else:
        Fx, Fy = arc_force(xyt[:, :2])

    # Unsteady N-S residuals
    res_x = (St * du_dt
             + u_hat * du_dx + v_hat * du_dy
             + dp_dx - nu_hat * (d2u_dx2 + d2u_dy2) - Fx)
    res_y = (St * dv_dt
             + u_hat * dv_dx + v_hat * dv_dy
             + dp_dy - nu_hat * (d2v_dx2 + d2v_dy2) - Fy)
    return res_x, res_y


# %% [Cell 7]
# ============================================================================
# Training Loop
# ============================================================================

def train(model, arc_force, data, bnd_data, cfg):
    U, L = data['U'], data['L']
    T_char = data['T_char']
    mask = data['mask_crop']

    # Strouhal number
    St = L / (U * T_char) if (U * T_char) > 0 else 1.0
    print(f"Strouhal number St = L/(U*T) = {St:.4f}")

    # Tensors on device
    bnd_pts_t = torch.tensor(bnd_data[0], dtype=torch.float32, device=DEVICE)
    bnd_n_t   = torch.tensor(bnd_data[1], dtype=torch.float32, device=DEVICE)
    bnd_w_t   = torch.tensor(bnd_data[2], dtype=torch.float32, device=DEVICE)
    all_int   = torch.tensor(data['interior_coords'],
                             dtype=torch.float32, device=DEVICE)
    cl_pts    = torch.tensor(data['cl_pts'],
                             dtype=torch.float32, device=DEVICE)

    # Sparse observations with time
    obs_xt    = data['sparse_xt']
    obs_yt    = data['sparse_yt']
    obs_t     = data['sparse_t']
    obs_u_hat = data['sparse_u_hat']
    obs_v_hat = data['sparse_v_hat']
    obs_w     = data['sparse_w']
    n_obs     = len(obs_xt)
    print(f"Sparse observation points: {n_obs}")

    # Pressure gauge: enforce p=0 at pool centre for ALL times
    # We sample several time points each epoch to pin the gauge across time
    n_p_gauge = 8  # number of time samples for pressure gauge

    # Learnable viscosity
    log_nu_hat = nn.Parameter(
        torch.tensor(np.log(0.01), dtype=torch.float32, device=DEVICE))

    # ── Arc force prior targets (from image measurements + rough estimate) ──
    # Width: estimated arc radius from images
    sigma_target_tilde = cfg['arc_sigma_init_m'] / L

    # Position: electrode center from image
    x_arc_target_tilde = data['cl_x_min'] + \
        cfg['arc_x_frac_measured'] * (data['cl_x_max'] - data['cl_x_min'])
    x_arc_tol_tilde = cfg['arc_x_frac_tolerance'] * \
        (data['cl_x_max'] - data['cl_x_min'])

    # Amplitude: rough prior in tilde coords
    # A_tilde = A_phys * L / (rho * U^2)
    amp_target_tilde = cfg.get('arc_amp_prior_Nm3', 3e4) * L / (RHO * U**2)

    print(f"Arc priors:")
    print(f"  sigma_arc = {cfg['arc_sigma_init_m']*1e3:.2f} mm  ->  sigma_tilde = {sigma_target_tilde:.4f}")
    print(f"  x_arc_frac = {cfg['arc_x_frac_measured']:.3f}  ->  x_tilde = {x_arc_target_tilde:.4f}")
    print(f"  A_arc = {cfg.get('arc_amp_prior_Nm3', 3e4):.0f} N/m^3  ->  A_tilde = {amp_target_tilde:.4f}  (±50%)")

    opt_groups = [
        {'params': model.parameters(), 'lr': cfg['lr']},
        {'params': [log_nu_hat],       'lr': cfg['lr_physics']},
    ]
    if not cfg.get('disable_arc_force', False):
        opt_groups.append({'params': arc_force.parameters(),
                           'lr': cfg['lr_physics']})
    optimizer = torch.optim.Adam(opt_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg['n_epochs'])

    keys = ['total', 'data', 'boundary', 'pde', 'p_ref', 'symmetry',
            'mirror_sym', 'arc_prior', 'energy_decay', 'temporal_smooth',
            'nu_hat', 'nu_phys', 'arc_x', 'arc_sigma', 'arc_amp']
    history = {k: [] for k in keys}

    # Time sample points for energy decay regularization
    n_energy_pts = 8
    t_energy = torch.linspace(0, 1, n_energy_pts, device=DEVICE)

    for epoch in tqdm(range(cfg['n_epochs']), desc="Training"):
        optimizer.zero_grad()

        # ── L_data: sample from sparse observations (with time) ──
        n_s = min(cfg['n_obs_per_step'], n_obs)
        sel = np.random.choice(n_obs, n_s, replace=False)

        xyt_d = torch.tensor(
            np.stack([obs_xt[sel], obs_yt[sel], obs_t[sel]], axis=1),
            dtype=torch.float32, device=DEVICE)
        up, vp, _, _ = get_velocity_hat(model, xyt_d)
        ut = torch.tensor(obs_u_hat[sel], dtype=torch.float32, device=DEVICE)
        vt = torch.tensor(obs_v_hat[sel], dtype=torch.float32, device=DEVICE)
        wt = torch.tensor(obs_w[sel], dtype=torch.float32, device=DEVICE)

        # Speed-based + coherence weights
        speed_obs = torch.sqrt(ut**2 + vt**2)
        w_speed = speed_obs / (speed_obs.mean() + 1e-8)
        w_total = w_speed * wt

        L_data = (w_total * (up.squeeze() - ut)**2).mean() + \
                 (w_total * (vp.squeeze() - vt)**2).mean()

        # Peak matching
        speed_pred = torch.sqrt(up.squeeze()**2 + vp.squeeze()**2)
        L_peak = (speed_obs.max() - speed_pred.max())**2
        L_data = L_data + cfg.get('lambda_peak', 0.1) * L_peak

        # ── L_boundary (no-penetration at all times) ──
        L_bnd = torch.tensor(0.0, device=DEVICE)
        if cfg['lambda_boundary'] > 0:
            nb = min(cfg['n_boundary_points'], bnd_pts_t.shape[0])
            ib = torch.randperm(bnd_pts_t.shape[0])[:nb]
            # Random time for each boundary point
            t_bnd = torch.rand(nb, 1, device=DEVICE)
            xyt_bnd = torch.cat([bnd_pts_t[ib], t_bnd], dim=1)
            ub, vb, _, _ = get_velocity_hat(model, xyt_bnd)
            u_n = ub.squeeze() * bnd_n_t[ib, 0] + vb.squeeze() * bnd_n_t[ib, 1]
            L_bnd = (bnd_w_t[ib] * u_n**2).mean()

        # ── L_pde (unsteady N-S residual) ──
        nc = min(cfg['n_pde_per_step'], all_int.shape[0])
        ic = torch.randperm(all_int.shape[0])[:nc]
        # Random time for collocation points
        t_col = torch.rand(nc, 1, device=DEVICE)
        xyt_col = torch.cat([all_int[ic], t_col], dim=1)
        rx, ry = compute_ns_residual_unsteady(
            model, arc_force, xyt_col, log_nu_hat, St, cfg)
        L_pde = (rx**2).mean() + (ry**2).mean()

        # ── L_p_ref (pressure gauge at pool center for all times) ──
        # Sample random times and enforce p(0, 0, t) = 0 for each
        t_gauge = torch.rand(n_p_gauge, 1, device=DEVICE)
        xyt_gauge = torch.cat([
            torch.zeros(n_p_gauge, 2, device=DEVICE), t_gauge], dim=1)
        _, _, p_at_gauge, _ = get_velocity_hat(model, xyt_gauge)
        L_p_ref = (p_at_gauge**2).mean()

        # ── L_symmetry (v = 0 on centerline at all times) ──
        L_sym = torch.tensor(0.0, device=DEVICE)
        if cfg['lambda_symmetry'] > 0 and cl_pts.shape[0] > 0:
            ncl = min(200, cl_pts.shape[0])
            icl = torch.randperm(cl_pts.shape[0])[:ncl]
            t_cl = torch.rand(ncl, 1, device=DEVICE)
            xyt_cl = torch.cat([cl_pts[icl], t_cl], dim=1)
            _, vcl, _, _ = get_velocity_hat(model, xyt_cl)
            L_sym = (vcl**2).mean()

        # ── L_mirror: full mirror symmetry about y_tilde = 0 ──
        # u(x, y, t) = u(x, -y, t)  and  v(x, y, t) = -v(x, -y, t)
        L_mirror = torch.tensor(0.0, device=DEVICE)
        if cfg.get('lambda_mirror_symmetry', 0) > 0:
            n_mir = min(400, all_int.shape[0])
            i_mir = torch.randperm(all_int.shape[0])[:n_mir]
            xy_mir = all_int[i_mir]  # (n_mir, 2)
            t_mir = torch.rand(n_mir, 1, device=DEVICE)

            # Original points (x, y, t)
            xyt_orig = torch.cat([xy_mir, t_mir], dim=1)
            # Mirrored points (x, -y, t)
            xy_flip = xy_mir.clone()
            xy_flip[:, 1] = -xy_flip[:, 1]
            xyt_flip = torch.cat([xy_flip, t_mir], dim=1)

            u_orig, v_orig, p_orig, _ = get_velocity_hat(model, xyt_orig)
            u_flip, v_flip, p_flip, _ = get_velocity_hat(model, xyt_flip)

            # u should be equal, v should be negated, p should be equal
            L_mirror = ((u_orig - u_flip)**2).mean() + \
                       ((v_orig + v_flip)**2).mean() + \
                       ((p_orig - p_flip)**2).mean()

        # ── L_energy_decay: KE should decrease over time (post-impact) ──
        # Sample spatial points, evaluate at successive times
        L_energy = torch.tensor(0.0, device=DEVICE)
        if cfg['lambda_energy_decay'] > 0:
            n_spatial = min(300, all_int.shape[0])
            i_e = torch.randperm(all_int.shape[0])[:n_spatial]
            xy_e = all_int[i_e]  # (n_spatial, 2)
            KE_list = []
            for ti in t_energy:
                t_col_e = ti.expand(n_spatial, 1)
                xyt_e = torch.cat([xy_e, t_col_e], dim=1)
                u_e, v_e, _, _ = get_velocity_hat(model, xyt_e)
                KE = (u_e**2 + v_e**2).mean()
                KE_list.append(KE)

            # Penalize increases in KE between consecutive time steps
            for j in range(1, len(KE_list)):
                diff = KE_list[j] - KE_list[j-1]
                L_energy = L_energy + torch.relu(diff)  # only penalize increases

        # ── L_temporal_smooth: penalize rapid temporal oscillations ──
        L_tsmooth = torch.tensor(0.0, device=DEVICE)
        if cfg['lambda_temporal_smooth'] > 0:
            n_ts = min(300, all_int.shape[0])
            i_ts = torch.randperm(all_int.shape[0])[:n_ts]
            xy_ts = all_int[i_ts]
            t1 = torch.rand(n_ts, 1, device=DEVICE)
            dt_eps = 0.02  # small time step
            t2 = torch.clamp(t1 + dt_eps, 0, 1)

            xyt1 = torch.cat([xy_ts, t1], dim=1)
            xyt2 = torch.cat([xy_ts, t2], dim=1)
            u1, v1, _, _ = get_velocity_hat(model, xyt1)
            u2, v2, _, _ = get_velocity_hat(model, xyt2)
            L_tsmooth = ((u2 - u1)**2 + (v2 - v1)**2).mean() / (dt_eps**2)

        # ── L_arc_prior: position and width from image, amplitude free ──
        L_arc_prior = torch.tensor(0.0, device=DEVICE)
        if not cfg.get('disable_arc_force', False):
            # Width prior (log-normal, from image-estimated arc radius)
            if cfg.get('lambda_arc_sigma_prior', 0) > 0:
                sigma_tilde = torch.exp(arc_force.log_sigma)
                log_std_sig = cfg['arc_sigma_prior_log_std']
                L_arc_prior = L_arc_prior + \
                    cfg['lambda_arc_sigma_prior'] * \
                    (torch.log(sigma_tilde) - np.log(sigma_target_tilde))**2 \
                    / (2 * log_std_sig**2)

            # Position prior (Gaussian, from image measurement)
            if cfg.get('lambda_arc_pos_prior', 0) > 0:
                x_arc_tilde = arc_force._x_arc()
                L_arc_prior = L_arc_prior + \
                    cfg['lambda_arc_pos_prior'] * \
                    ((x_arc_tilde - x_arc_target_tilde) / x_arc_tol_tilde)**2

            # Amplitude prior (log-normal, rough estimate ±50%)
            if cfg.get('lambda_arc_amp_prior', 0) > 0:
                amp_tilde = torch.exp(arc_force.log_amp)
                log_std_amp = cfg['arc_amp_prior_log_std']
                L_arc_prior = L_arc_prior + \
                    cfg['lambda_arc_amp_prior'] * \
                    (torch.log(amp_tilde) - np.log(amp_target_tilde))**2 \
                    / (2 * log_std_amp**2)

        # ── Total loss ──
        loss = (cfg['lambda_data']            * L_data +
                cfg['lambda_boundary']        * L_bnd +
                cfg['lambda_pde']             * L_pde +
                cfg['lambda_p_ref']           * L_p_ref +
                cfg['lambda_symmetry']        * L_sym +
                cfg.get('lambda_mirror_symmetry', 0) * L_mirror +
                cfg['lambda_energy_decay']    * L_energy +
                cfg['lambda_temporal_smooth'] * L_tsmooth +
                L_arc_prior)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Record
        nu_hat_val  = torch.exp(log_nu_hat).item()
        nu_phys_val = nu_hat_val * U * L
        xa, sa, aa  = arc_force.get_params()

        history['total'].append(loss.item())
        history['data'].append(L_data.item())
        history['boundary'].append(L_bnd.item())
        history['pde'].append(L_pde.item())
        history['p_ref'].append(L_p_ref.item())
        history['symmetry'].append(L_sym.item())
        history['mirror_sym'].append(L_mirror.item())
        history['arc_prior'].append(L_arc_prior.item())
        history['energy_decay'].append(L_energy.item())
        history['temporal_smooth'].append(L_tsmooth.item())
        history['nu_hat'].append(nu_hat_val)
        history['nu_phys'].append(nu_phys_val)
        history['arc_x'].append(xa)
        history['arc_sigma'].append(sa)
        history['arc_amp'].append(aa)

        if (epoch + 1) % 100 == 0:
            arc_str = (f"  sigma={sa*L*1e3:.2f}mm"
                       if not cfg.get('disable_arc_force', False) else "")
            print(f"  Ep {epoch+1:4d}  loss={loss.item():.5f}  "
                  f"data={L_data.item():.5f}  pde={L_pde.item():.5f}  "
                  f"E_dec={L_energy.item():.5f}{arc_str}  "
                  f"nu_hat={nu_hat_val:.5f}  "
                  f"nu_phys={nu_phys_val:.2e}")

    # ── Final summary ──
    nu_hat_f  = torch.exp(log_nu_hat).item()
    nu_phys_f = nu_hat_f * U * L
    mu_phys_f = nu_phys_f * RHO
    Re        = 1.0 / nu_hat_f

    print(f"\n{'='*60}")
    print(f"LEARNED PHYSICS PARAMETERS (after {cfg['n_epochs']} epochs)")
    print(f"{'='*60}")
    print(f"  nu_hat  (dimensionless) = {nu_hat_f:.6f}")
    print(f"  nu_phys = {nu_phys_f:.4e} m^2/s")
    print(f"  mu_phys = {mu_phys_f:.4e} Pa.s  ({mu_phys_f*1e3:.3f} mPa.s)")
    print(f"  Re      = {Re:.1f}")
    print(f"  St      = {St:.4f}")
    if not cfg.get('disable_arc_force', False):
        xa, sa, aa = arc_force.get_params()
        print(f"  Arc: x_tilde={xa:.4f}  sigma_tilde={sa:.4f}  "
              f"A_tilde={aa:.4f}")
    else:
        print(f"  Arc force: DISABLED")
    print(f"{'='*60}")

    return history, log_nu_hat, St

# %% [Cell 8]
# ── Build model and train ──
model = FlowNet(CONFIG['hidden_dims']).to(DEVICE)
print(f"FlowNet: {sum(p.numel() for p in model.parameters()):,} params")
print(f"Architecture: [3] -> {CONFIG['hidden_dims']} -> [2]")

# ── Arc force initial values from image measurements ──
_sigma_init_tilde = CONFIG['arc_sigma_init_m'] / data['L']
print(f"Arc force init: x_frac={CONFIG['arc_x_frac_init']:.3f}, "
      f"sigma={CONFIG['arc_sigma_init_m']*1e3:.1f} mm (tilde={_sigma_init_tilde:.4f}), "
      f"amp={CONFIG['arc_amp_init']} (free to learn)")

arc_force = ArcForce(
    CONFIG['arc_x_frac_init'],
    _sigma_init_tilde,           # image-estimated sigma (in tilde coords)
    CONFIG['arc_amp_init'],      # arbitrary init — learned from data + PDE
    y_center=0.0,
    x_min=data['cl_x_min'],
    x_max=data['cl_x_max'],
).to(DEVICE)

history, log_nu_hat, St = train(
    model, arc_force, data,
    (bnd_pts, bnd_normals, bnd_weights), CONFIG)


# %% [Cell 9]
# ============================================================================
# Dense Prediction at Multiple Time Steps
# ============================================================================

@torch.enable_grad()
def predict_full_SI_at_time(model, data, t_hat):
    """Dense (u, v, p) over cropped mask in SI units at a given t_hat."""
    h_c, w_c = data['h_crop'], data['w_crop']
    U, L = data['U'], data['L']
    ix, iy = data['interior_ix'], data['interior_iy']
    cx, cy = data['cx_px'], data['cy_px']

    u_full = np.zeros((h_c, w_c), dtype=np.float32)
    v_full = np.zeros((h_c, w_c), dtype=np.float32)
    p_full = np.zeros((h_c, w_c), dtype=np.float32)

    batch = 5000
    for s in range(0, len(ix), batch):
        e = min(s + batch, len(ix))
        bx, by = ix[s:e], iy[s:e]

        x_t = (bx - cx) * PIXEL_SIZE_M / L
        y_t = (by - cy) * PIXEL_SIZE_M / L
        t_arr = np.full_like(x_t, t_hat)
        xyt = torch.tensor(np.stack([x_t, y_t, t_arr], axis=1),
                           dtype=torch.float32, device=DEVICE)
        xyt.requires_grad_(True)

        psi_hat, p_hat = model(xyt)
        grad = torch.autograd.grad(
            psi_hat, xyt, torch.ones_like(psi_hat))[0]

        u_full[by, bx] =  U * grad[:, 1].detach().cpu().numpy()
        v_full[by, bx] = -U * grad[:, 0].detach().cpu().numpy()
        p_full[by, bx] = RHO * U**2 * p_hat.squeeze().detach().cpu().numpy()

    return u_full, v_full, p_full


# Predict at several time steps
n_time_steps = 3
t_hat_values = np.linspace(0, 1, n_time_steps)
predictions = {}
for t_hat in t_hat_values:
    u, v, p = predict_full_SI_at_time(model, data, t_hat)
    predictions[t_hat] = {'u': u, 'v': v, 'p': p,
                          'speed': np.sqrt(u**2 + v**2)}
    print(f"t_hat={t_hat:.2f}: max speed = {predictions[t_hat]['speed'][data['mask_crop']].max():.4f} m/s")


# %% [Cell 9b]
# ============================================================================
# Reconstruction Error on Selected Frames
# ============================================================================
# For each selected frame: re-apply the same speed filtering as load_data(),
# get PINN prediction at that frame's t_hat, compute errors at filtered pixels.
# Pick 3 evenly spaced frames for visual illustration.
# ============================================================================

print("\n--- Computing reconstruction error on selected frames ---")

_rc_h_c, _rc_w_c = data['h_crop'], data['w_crop']
_rc_mask = data['mask_crop']
_rc_n_frames = data['n_frames']
_rc_speed_range_pct = 0.7  # must match the hardcoded value in load_data()

# Accumulate errors across all selected frames
all_u_errors = []
all_v_errors = []
all_speed_errors = []
all_obs_speeds = []

# Store per-frame results for visualization
per_frame_results = []

for _rc_i in tqdm(data['selected_frame_indices'], desc="Recon error per frame"):
    _rc_flow_i = data['all_flow'][_rc_i]  # (2, h_c, w_c) in m/s
    _rc_u_frame = _rc_flow_i[0]
    _rc_v_frame = _rc_flow_i[1]
    _rc_mag = np.sqrt(_rc_u_frame**2 + _rc_v_frame**2)

    # Same filtering as load_data()
    _rc_nontrivial = _rc_mask & (_rc_mag > 1e-6)
    if _rc_nontrivial.sum() == 0:
        continue
    _rc_speeds_nt = _rc_mag[_rc_nontrivial]
    _rc_speed_lo, _rc_speed_hi = _rc_speeds_nt.min(), _rc_speeds_nt.max()
    _rc_speed_thresh = _rc_speed_lo + _rc_speed_range_pct * (_rc_speed_hi - _rc_speed_lo)
    _rc_valid = _rc_nontrivial & (_rc_mag >= _rc_speed_thresh)
    if _rc_valid.sum() == 0:
        continue

    _rc_vy, _rc_vx = np.where(_rc_valid)
    _rc_t_hat_i = _rc_i / max(_rc_n_frames - 1, 1)

    # PINN prediction at this frame's t_hat
    _rc_u_pinn, _rc_v_pinn, _ = predict_full_SI_at_time(model, data, _rc_t_hat_i)

    # Errors at filtered pixels
    _rc_u_raft_sel = _rc_u_frame[_rc_vy, _rc_vx]
    _rc_v_raft_sel = _rc_v_frame[_rc_vy, _rc_vx]
    _rc_u_pinn_sel = _rc_u_pinn[_rc_vy, _rc_vx]
    _rc_v_pinn_sel = _rc_v_pinn[_rc_vy, _rc_vx]
    _rc_speed_raft_sel = _rc_mag[_rc_vy, _rc_vx]
    _rc_speed_pinn_sel = np.sqrt(_rc_u_pinn_sel**2 + _rc_v_pinn_sel**2)

    all_u_errors.append(_rc_u_pinn_sel - _rc_u_raft_sel)
    all_v_errors.append(_rc_v_pinn_sel - _rc_v_raft_sel)
    all_speed_errors.append(_rc_speed_pinn_sel - _rc_speed_raft_sel)
    all_obs_speeds.append(_rc_speed_raft_sel)

    # Per-frame maps (only at filtered pixels, NaN elsewhere)
    _rc_raft_spd = np.full((_rc_h_c, _rc_w_c), np.nan, dtype=np.float32)
    _rc_pinn_spd = np.full((_rc_h_c, _rc_w_c), np.nan, dtype=np.float32)
    _rc_err_spd  = np.full((_rc_h_c, _rc_w_c), np.nan, dtype=np.float32)
    _rc_raft_spd[_rc_vy, _rc_vx] = _rc_speed_raft_sel
    _rc_pinn_spd[_rc_vy, _rc_vx] = _rc_speed_pinn_sel
    _rc_err_spd[_rc_vy, _rc_vx]  = np.abs(_rc_speed_pinn_sel - _rc_speed_raft_sel)

    per_frame_results.append({
        'frame_idx': _rc_i,
        't_hat': _rc_t_hat_i,
        't_ms': _rc_t_hat_i * data['T_char'] * 1e3,
        'raft_speed': _rc_raft_spd,
        'pinn_speed': _rc_pinn_spd,
        'error_speed': _rc_err_spd,
        'n_valid_pts': len(_rc_vx),
        'frame_MAE': float(np.abs(_rc_speed_pinn_sel - _rc_speed_raft_sel).mean()),
        'frame_RMSE': float(np.sqrt(((_rc_speed_pinn_sel - _rc_speed_raft_sel)**2).mean())),
    })

# ── Aggregate statistics ──
if len(all_u_errors) > 0:
    all_u_errors = np.concatenate(all_u_errors)
    all_v_errors = np.concatenate(all_v_errors)
    all_speed_errors = np.concatenate(all_speed_errors)
    all_obs_speeds = np.concatenate(all_obs_speeds)

    recon_errors = {
        'reference':       'RAFT filtered observations at selected frames',
        'n_obs_points':    int(len(all_u_errors)),
        'n_selected_frames': len(data['selected_frame_indices']),
        'u_MAE_ms':        float(np.abs(all_u_errors).mean()),
        'u_RMSE_ms':       float(np.sqrt((all_u_errors**2).mean())),
        'v_MAE_ms':        float(np.abs(all_v_errors).mean()),
        'v_RMSE_ms':       float(np.sqrt((all_v_errors**2).mean())),
        'speed_MAE_ms':    float(np.abs(all_speed_errors).mean()),
        'speed_RMSE_ms':   float(np.sqrt((all_speed_errors**2).mean())),
    }
else:
    recon_errors = {'reference': 'no observed points', 'n_obs_points': 0}

print(f"\nReconstruction errors ({recon_errors.get('n_obs_points', 0)} obs points "
      f"from {recon_errors.get('n_selected_frames', 0)} selected frames):")
print(f"  u   MAE={recon_errors.get('u_MAE_ms', float('nan')):.4f}  "
      f"RMSE={recon_errors.get('u_RMSE_ms', float('nan')):.4f} m/s")
print(f"  v   MAE={recon_errors.get('v_MAE_ms', float('nan')):.4f}  "
      f"RMSE={recon_errors.get('v_RMSE_ms', float('nan')):.4f} m/s")
print(f"  spd MAE={recon_errors.get('speed_MAE_ms', float('nan')):.4f}  "
      f"RMSE={recon_errors.get('speed_RMSE_ms', float('nan')):.4f} m/s")


# %% [Cell 10]
# ============================================================================
# Visualization
# ============================================================================

import matplotlib
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

results_dir = CONFIG['results_dir']
os.makedirs(results_dir, exist_ok=True)

cnt_px = data['bnd_cnt_px']
cnt_c  = np.vstack([cnt_px, cnt_px[:1]])
h_c, w_c = data['h_crop'], data['w_crop']
U, L = data['U'], data['L']
T_char = data['T_char']
mask = data['mask_crop']

x_mm = np.arange(w_c) * PIXEL_SIZE_M * 1e3
y_mm = np.arange(h_c) * PIXEL_SIZE_M * 1e3
cnt_x_mm = cnt_c[:, 0] * PIXEL_SIZE_M * 1e3
cnt_y_mm = cnt_c[:, 1] * PIXEL_SIZE_M * 1e3
TICK_VALUES_MM = [0, 5, 10]
BOUNDARY_LW = 0.8
STREAMLINE_LW = 0.5
STREAMLINE_DENS = 1.5
STREAMLINE_ARROW = 0.4
SPEED_ALPHA = 0.6
SUBPLOT_BORDER_LW = 0.5

def _format_ax(ax, title, xlabel='$x$ (mm)', ylabel='$y$ (mm)'):
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel(ylabel, labelpad=1)
    ax.set_title(title, pad=3)
    ax.set_aspect('equal')
    ax.set_xlim(x_mm[0], x_mm[-1])
    ax.set_ylim(y_mm[-1], y_mm[0])
    ax.set_xticks(TICK_VALUES_MM)
    ax.set_yticks(TICK_VALUES_MM)
    ax.tick_params(axis='both', pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(SUBPLOT_BORDER_LW)


# ████████████████████████████████████████████████████████████████████████████
# FIGURE 0: Reconstruction Error at 3 Evenly Spaced Time Snapshots
# ████████████████████████████████████████████████████████████████████████████
# Each row: RAFT filtered obs | PINN at same t_hat | speed error
# Only filtered (sparse) pixels are shown; rest is NaN (white).
# ████████████████████████████████████████████████████████████████████████████

n_sel = len(per_frame_results)
if n_sel >= 3:
    repr_indices = [0, n_sel // 2, n_sel - 1]
elif n_sel > 0:
    repr_indices = list(range(n_sel))
else:
    repr_indices = []

n_repr = len(repr_indices)

if n_repr > 0:
    # Global color scale across representative frames
    vmax_err_fig = max(
        np.nanmax(per_frame_results[ri]['raft_speed'])
        for ri in repr_indices)

    fig_err, axes_err = plt.subplots(n_repr, 3, figsize=(7.0, 2.6 * n_repr))
    if n_repr == 1:
        axes_err = axes_err[np.newaxis, :]

    for row, ri in enumerate(repr_indices):
        pfr = per_frame_results[ri]
        t_ms = pfr['t_ms']

        ax_r = axes_err[row, 0]
        ax_p = axes_err[row, 1]
        ax_e = axes_err[row, 2]

        # (a) RAFT observation (filtered pixels only)
        im_r = ax_r.pcolormesh(x_mm, y_mm, pfr['raft_speed'],
                                cmap='jet', shading='auto',
                                vmin=0, vmax=vmax_err_fig, alpha=SPEED_ALPHA)
        ax_r.plot(cnt_x_mm, cnt_y_mm, 'k-', lw=BOUNDARY_LW)
        cb_r = plt.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)
        cb_r.set_label('m/s')
        cb_r.outline.set_visible(False)
        _format_ax(ax_r, '')

        # (b) PINN at same t_hat (filtered pixels only)
        im_p = ax_p.pcolormesh(x_mm, y_mm, pfr['pinn_speed'],
                                cmap='jet', shading='auto',
                                vmin=0, vmax=vmax_err_fig, alpha=SPEED_ALPHA)
        ax_p.plot(cnt_x_mm, cnt_y_mm, 'k-', lw=BOUNDARY_LW)
        cb_p = plt.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04)
        cb_p.set_label('m/s')
        cb_p.outline.set_visible(False)
        _format_ax(ax_p, '')

        # (c) Speed error
        im_e = ax_e.pcolormesh(x_mm, y_mm, pfr['error_speed'],
                                cmap='hot', shading='auto')
        ax_e.plot(cnt_x_mm, cnt_y_mm, 'k-', lw=BOUNDARY_LW)
        cb_e = plt.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.04)
        cb_e.set_label('m/s')
        cb_e.outline.set_visible(False)
        _format_ax(ax_e, '')

        # Clean up redundant labels
        ax_p.set_ylabel('')
        ax_e.set_ylabel('')

    plt.tight_layout(w_pad=0.4, h_pad=0.6)
    fig_err.savefig(os.path.join(results_dir, 'fig_error_illustration.png'),
                    dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig_err)
    print(f"Saved: {results_dir}/fig_error_illustration.png")
else:
    print("No selected frames available for error illustration.")


# ████████████████████████████████████████████████████████████████████████████
# FIGURE 1: Flow field at 3 time snapshots (early / mid / late)
# ████████████████████████████████████████████████████████████████████████████

# Global color scale across all times
vmax_speed = max(predictions[t]['speed'][mask].max() for t in predictions)

fig1, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
Xd_mm, Yd_mm = np.meshgrid(x_mm, y_mm)

# Arc center in mm (for marking on all subplots)
_arc_disabled_fig = CONFIG.get('disable_arc_force', False)
if not _arc_disabled_fig:
    _xa_fig, _sa_fig, _aa_fig = arc_force.get_params()
    _arc_cx_px_fig = data['cx_px'] + _xa_fig * data['L'] / PIXEL_SIZE_M
    _arc_cx_mm_fig = _arc_cx_px_fig * PIXEL_SIZE_M * 1e3
    _arc_cy_mm_fig = data['cy_px'] * PIXEL_SIZE_M * 1e3

for idx, t_hat in enumerate(t_hat_values):
    ax = axes[idx]
    pred = predictions[t_hat]
    speed_display = np.where(mask, pred['speed'], np.nan)
    t_ms = t_hat * T_char * 1e3  # convert to ms

    im = ax.pcolormesh(x_mm, y_mm, speed_display,
                       cmap='jet', shading='auto',
                       vmin=0, vmax=vmax_speed, alpha=SPEED_ALPHA)
    ax.plot(cnt_x_mm, cnt_y_mm, 'k-', lw=BOUNDARY_LW)

    # Streamlines
    u_p = np.where(mask, pred['u'], 0)
    v_p = np.where(mask, pred['v'], 0)
    ax.streamplot(Xd_mm, Yd_mm, u_p, v_p, color='k',
                  linewidth=STREAMLINE_LW, density=STREAMLINE_DENS,
                  arrowsize=STREAMLINE_ARROW, arrowstyle='->')

    # Mark arc force center
    if not _arc_disabled_fig:
        ax.plot(_arc_cx_mm_fig, _arc_cy_mm_fig, 'r*', ms=7, mew=0.8, zorder=5)

    speed_max = pred['speed'][mask].max()
    _format_ax(ax, f'$t$ = {t_ms:.1f} ms  ($v_{{max}}$ = {speed_max:.3f} m/s)')

    if idx != 0:
        ax.set_ylabel('')

# Shared colorbar
cbar_ax = fig1.add_axes([0.92, 0.15, 0.015, 0.7])
cb = fig1.colorbar(im, cax=cbar_ax)
cb.set_label('Speed (m/s)')
cb.outline.set_visible(False)

plt.tight_layout(rect=[0, 0, 0.91, 0.96])
fig1.savefig(os.path.join(results_dir, 'fig_flow_evolution.png'), dpi=300)
plt.show()
plt.close(fig1)
print(f"Saved: {results_dir}/fig_flow_evolution.png")


# ████████████████████████████████████████████████████████████████████████████
# FIGURE 2: Kinetic energy decay profile
# ████████████████████████████████████████████████████████████████████████████

t_fine = np.linspace(0, 1, 50)
KE_profile = []
for t_hat in tqdm(t_fine, desc="KE profile"):
    u, v, _ = predict_full_SI_at_time(model, data, t_hat)
    KE = 0.5 * RHO * np.mean((u[mask]**2 + v[mask]**2))
    KE_profile.append(KE)
KE_profile = np.array(KE_profile)

fig2, ax = plt.subplots(1, 1, figsize=(5, 3))
t_fine_ms = t_fine * T_char * 1e3
ax.plot(t_fine_ms, KE_profile, 'b-', lw=1.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Mean KE density (J/m³)')
ax.set_title('Kinetic energy decay after droplet impact')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig(os.path.join(results_dir, 'fig_KE_decay.png'), dpi=300)
plt.show()
plt.close(fig2)
print(f"Saved: {results_dir}/fig_KE_decay.png")


# ████████████████████████████████████████████████████████████████████████████
# FIGURE 3: Pressure field at early / mid / late times
# ████████████████████████████████████████████████████████████████████████████

fig3, axes3 = plt.subplots(1, 3, figsize=(7.0, 2.6))
time_indices = [0, len(t_hat_values)//2, len(t_hat_values)-1]

# Global symmetric pressure range: -absmax to +absmax across all panels
p_absmax = max(
    np.abs(predictions[t_hat_values[i]]['p'][mask]).max()
    for i in time_indices)
p_vmin, p_vmax = -p_absmax, p_absmax

for panel, ti in enumerate(time_indices):
    ax = axes3[panel]
    t_hat = t_hat_values[ti]
    pred = predictions[t_hat]
    t_ms = t_hat * T_char * 1e3

    p_field = pred['p']
    p_display = np.where(mask, p_field, np.nan)
    im_p = ax.pcolormesh(x_mm, y_mm, p_display,
                         cmap='RdBu_r', shading='auto',
                         vmin=p_vmin, vmax=p_vmax)
    ax.plot(cnt_x_mm, cnt_y_mm, 'k-', lw=BOUNDARY_LW)

    # Annotate per-panel pressure range
    dp = p_field[mask].max() - p_field[mask].min()
    _format_ax(ax, f'$t$ = {t_ms:.1f} ms  ($\\Delta p$ = {dp:.1f} Pa)')
    if panel > 0:
        ax.set_ylabel('')

# Shared colorbar for all three panels
cbar_ax_p = fig3.add_axes([0.92, 0.15, 0.015, 0.7])
cb_p = fig3.colorbar(im_p, cax=cbar_ax_p)
cb_p.set_label('Pa')
cb_p.outline.set_visible(False)

plt.tight_layout(rect=[0, 0, 0.91, 0.96])
fig3.savefig(os.path.join(results_dir, 'fig_pressure_evolution.png'), dpi=300)
plt.show()
plt.close(fig3)
print(f"Saved: {results_dir}/fig_pressure_evolution.png")


# ████████████████████████████████████████████████████████████████████████████
# FIGURE 3b: Arc Force Distribution (constant in time)
# ████████████████████████████████████████████████████████████████████████████
# The arc force is time-independent — the welding arc configuration doesn't
# change during the short post-impact observation window. So we plot it once.
# Format matches the quasi-static (0mm) code exactly.

_arc_disabled = CONFIG.get('disable_arc_force', False)
xa, sa, aa = arc_force.get_params()

if not _arc_disabled:
    # Compute arc force field on the spatial grid
    x_norm = np.linspace(
        (0 - data['cx_px']) * PIXEL_SIZE_M / L,
        (w_c - data['cx_px']) * PIXEL_SIZE_M / L, w_c)
    y_norm = np.linspace(
        (0 - data['cy_px']) * PIXEL_SIZE_M / L,
        (h_c - data['cy_px']) * PIXEL_SIZE_M / L, h_c)
    Xn, Yn = np.meshgrid(x_norm, y_norm)
    xy_grid = torch.tensor(np.stack([Xn.ravel(), Yn.ravel()], axis=1),
                            dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        Fx_t, Fy_t = arc_force(xy_grid)
    Fx_map = Fx_t.cpu().numpy().reshape(h_c, w_c)
    Fy_map = Fy_t.cpu().numpy().reshape(h_c, w_c)
    F_mag_tilde = np.sqrt(Fx_map**2 + Fy_map**2)
    F_scale = RHO * U**2 / L   # N/m^3 per tilde unit
    F_mag_SI = F_mag_tilde * F_scale
    Fx_SI = Fx_map * F_scale
    Fy_SI = Fy_map * F_scale

    # Arc center in mm
    arc_cx_px = data['cx_px'] + xa * L / PIXEL_SIZE_M
    arc_cx_mm = arc_cx_px * PIXEL_SIZE_M * 1e3
    arc_cy_mm = data['cy_px'] * PIXEL_SIZE_M * 1e3

    # Single-subplot figure matching the quasi-static (0mm) format exactly:
    # magnitude colormap (white-red) + quiver arrows, clipped to pool boundary
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.ticker import MaxNLocator

    fig_arc, ax_f = plt.subplots(1, 1, figsize=(8/3, 2.8))

    _cmap_arc = matplotlib.colors.LinearSegmentedColormap.from_list(
        'white_yellow_red', [(0, 'white'), (0.25, 'lightyellow'), (0.5, 'orangered'), (1.0, 'darkred')])
    f_display = np.where(mask, F_mag_SI / 1e4, np.nan)   # scale to 10^4
    im_f = ax_f.pcolormesh(x_mm, y_mm, f_display,
                            cmap=_cmap_arc, shading='auto')
    ax_f.plot(cnt_x_mm, cnt_y_mm, 'k-', lw=BOUNDARY_LW)

    # Clip polygon from pool boundary — confines color + arrows to pool
    clip_verts = np.column_stack([cnt_x_mm, cnt_y_mm])
    clip_poly = MplPolygon(clip_verts, closed=True,
                           facecolor='none', edgecolor='none',
                           transform=ax_f.transData)
    ax_f.add_patch(clip_poly)
    im_f.set_clip_path(clip_poly)

    # Quiver arrows — clipped to pool boundary
    skip = max(1, min(h_c, w_c) // 20)
    qx = x_mm[::skip]
    qy = y_mm[::skip]
    quiv = ax_f.quiver(qx, qy,
                       Fx_SI[::skip, ::skip], -Fy_SI[::skip, ::skip],
                       color='white', scale=F_mag_SI[mask].max() * 12 + 1e-8,
                       width=0.004, headwidth=3.5, alpha=0.85)
    quiv.set_clip_path(clip_poly)

    cb_f = plt.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.04)
    cb_f.set_label(r'$\times 10^4$ N/m$^3$')
    cb_f.outline.set_visible(False)
    cb_f.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _format_ax(ax_f, 'Arc force')

    plt.tight_layout()
    fig_arc.savefig(os.path.join(results_dir, 'fig_arc_force.png'), dpi=300)
    plt.show()
    plt.close(fig_arc)
    print(f"Saved: {results_dir}/fig_arc_force.png")
else:
    print("Arc force disabled — skipping arc force figure.")


# ████████████████████████████████████████████████████████████████████████████
# FIGURE 4: Training convergence
# ████████████████████████████████████████████████████████████████████████████

def smooth(y, window=5):
    y = np.array(y, dtype=np.float64)
    if len(y) < window:
        return y
    out = np.copy(y)
    hw = window // 2
    for i in range(len(y)):
        lo = max(0, i - hw)
        hi = min(len(y), i + hw + 1)
        out[i] = np.mean(y[lo:hi])
    return out

epochs = np.arange(1, len(history['total']) + 1)

fig4, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 3.5))

# (a) Losses
ax_a.semilogy(epochs, smooth(history['total']),
              'k-', lw=1.2, label=r'$\mathcal{L}_{\mathrm{total}}$')
ax_a.semilogy(epochs, smooth(history['data']),
              '-', color='#2166AC', lw=0.9,
              label=r'$\mathcal{L}_{\mathrm{data}}$')
ax_a.semilogy(epochs, smooth(history['pde']),
              '-', color='#B2182B', lw=0.9,
              label=r'$\mathcal{L}_{\mathrm{PDE}}$')
ax_a.semilogy(epochs, smooth(history['boundary']),
              '-', color='#1B7837', lw=0.9,
              label=r'$\mathcal{L}_{\mathrm{bnd}}$')
# ax_a.semilogy(epochs, smooth(history['energy_decay']),
#               '-', color='#E08214', lw=0.9,
#               label=r'$\mathcal{L}_{\mathrm{KE}}$')
ax_a.semilogy(epochs, smooth(history['symmetry']),
              '--', color='#762A83', lw=0.8,
              label=r'$\mathcal{L}_{\mathrm{sym}}$')
ax_a.semilogy(epochs, smooth(history['arc_prior']),
              '--', color='#E08214', lw=0.8,
              label=r'$\mathcal{L}_{\mathrm{arc}}$')
# ax_a.semilogy(epochs, smooth(history['mirror_sym']),
#               '--', color='#D95F02', lw=0.8,
#               label=r'$\mathcal{L}_{\mathrm{mirror}}$')
ax_a.set_xlabel('Epoch')
ax_a.set_ylabel('Loss')
ax_a.set_title('Training losses', pad=4)
ax_a.legend(loc='upper right', framealpha=0.9, edgecolor='none',
            handlelength=1.2, ncol=2, fontsize=7)
ax_a.grid(True, alpha=0.2, linewidth=0.5)

# (b) Learned parameters
L_val = data['L']
nu_phys_arr = np.array(history['nu_phys']) * 1e5
x_arc_mm    = np.array(history['arc_x']) * L_val * 1e3
sigma_mm    = np.array(history['arc_sigma']) * L_val * 1e3

ax_b.plot(epochs, smooth(nu_phys_arr), '-', color='#2166AC', lw=1.0,
          label=r'$\nu$ ($\times 10^{-5}$ m$^2$/s)')
ax_b.plot(epochs, smooth(x_arc_mm), '-', color='#762A83', lw=1.0,
          label=r'$x_{\mathrm{arc}}$ (mm)')
ax_b.set_xlabel('Epoch')
ax_b.set_ylabel('Value')
ax_b.set_title('Learned parameters', pad=4)
ax_b.grid(True, alpha=0.2, linewidth=0.5)

ax_r = ax_b.twinx()
ax_r.plot(epochs, smooth(sigma_mm), '--', color='#B2182B', lw=1.0,
          label=r'$\sigma_{\mathrm{arc}}$ (mm)')
ax_r.set_ylabel('Value')
lines_l, labels_l = ax_b.get_legend_handles_labels()
lines_r, labels_r = ax_r.get_legend_handles_labels()
ax_b.legend(lines_l + lines_r, labels_l + labels_r,
            loc='upper right', framealpha=0.9, edgecolor='none',
            handlelength=1.5, fontsize=7)

plt.tight_layout()
fig4.savefig(os.path.join(results_dir, 'fig_training_curves.png'), dpi=300)
plt.show()
plt.close(fig4)
print(f"Saved: {results_dir}/fig_training_curves.png")


# %% [Cell 11]
# ============================================================================
# Physical Property Extraction
# ============================================================================

U, L = data['U'], data['L']
T_char = data['T_char']
nu_hat_final = torch.exp(log_nu_hat).item()
xa, sa, aa   = arc_force.get_params()
_arc_disabled = CONFIG.get('disable_arc_force', False)

nu_phys = nu_hat_final * U * L
mu_phys = nu_phys * RHO
Re      = 1.0 / nu_hat_final

# Use t=0 (just after impact) for peak velocity
pred_t0 = predictions[0.0]
speed_t0 = pred_t0['speed']
u_max = speed_t0[mask].max()
u_mean = speed_t0[mask].mean()

# Use t=1 (end) for final state
pred_t1 = predictions[1.0]
speed_t1 = pred_t1['speed']
u_max_final = speed_t1[mask].max()

# Pressure range at t=0
p_t0 = pred_t0['p']
p_range = p_t0[mask].max() - p_t0[mask].min()

# Arc force
if not _arc_disabled:
    x_arc_phys = xa * L
    sigma_phys = sa * L
    A_phys = aa * RHO * U**2 / L

# Vorticity at t=0
dv_dx_fd = np.gradient(pred_t0['v'], PIXEL_SIZE_M, axis=1)
du_dy_fd = np.gradient(pred_t0['u'], PIXEL_SIZE_M, axis=0)
omega = dv_dx_fd - du_dy_fd
omega_max = np.abs(omega[mask]).max()

print("=" * 70)
print("PHYSICAL PROPERTY SUMMARY (SI Units)")
print("=" * 70)
print(f"\n--- Characteristic Scales ---")
print(f"  L (pool size)      = {L*1e3:.2f} mm")
print(f"  U (max velocity)   = {U:.4f} m/s")
print(f"  T (obs. window)    = {T_char*1e3:.2f} ms")
print(f"  St = L/(U*T)       = {St:.4f}")

print(f"\n--- Material Properties ---")
print(f"  nu_hat (dimless)     = {nu_hat_final:.6f}")
print(f"  Kinematic viscosity  = {nu_phys:.4e} m^2/s")
print(f"  Dynamic viscosity    = {mu_phys*1e3:.3f} mPa.s")
print(f"  Reynolds number      = {Re:.1f}")

print(f"\n--- Flow Characteristics ---")
print(f"  Max speed at t=0     = {u_max:.4f} m/s  ({u_max*1e3:.1f} mm/s)")
print(f"  Mean speed at t=0    = {u_mean:.4f} m/s")
print(f"  Max speed at t=T     = {u_max_final:.4f} m/s")
print(f"  Speed decay ratio    = {u_max_final/u_max:.3f}")
print(f"  Max vorticity at t=0 = {omega_max:.1f} 1/s")

if not _arc_disabled:
    print(f"\n--- Arc Force ---")
    print(f"  Position x_arc = {x_arc_phys*1e3:.2f} mm")
    print(f"  Width sigma    = {sigma_phys*1e3:.2f} mm")
    print(f"  Amplitude A    = {A_phys:.2e} N/m^3")

print("=" * 70)


# %% [Cell 12]
# ============================================================================
# Save Results
# ============================================================================

# Model checkpoint
ckpt_path = os.path.join(results_dir, 'pinn_model.pt')
torch.save({
    'model_state':     model.state_dict(),
    'arc_force_state': arc_force.state_dict(),
    'log_nu_hat':      log_nu_hat.detach().cpu(),
    'config':          CONFIG,
    'L': L, 'U': U, 'T_char': T_char, 'St': St, 'RHO': RHO,
    'history':         history,
}, ckpt_path)
print(f"Saved checkpoint: {ckpt_path}")

# Fields at all predicted time steps
for t_hat, pred in predictions.items():
    fname = f'fields_SI_t{t_hat:.2f}.npz'
    np.savez_compressed(
        os.path.join(results_dir, fname),
        u_ms=pred['u'], v_ms=pred['v'], p_Pa=pred['p'],
        mask=mask, speed_ms=pred['speed'], t_hat=t_hat,
        t_ms=t_hat * T_char * 1e3,
    )
print(f"Saved fields for {len(predictions)} time steps")

# Summary JSON
summary = {
    'characteristic_scales': {
        'L_mm': float(L * 1e3),
        'U_ms': float(U),
        'T_ms': float(T_char * 1e3),
        'St': float(St),
    },
    'material_properties': {
        'rho_kgm3': float(RHO),
        'nu_hat_dimless': float(nu_hat_final),
        'nu_phys_m2s': float(nu_phys),
        'mu_phys_mPas': float(mu_phys * 1e3),
        'Re': float(Re),
    },
    'flow_characteristics': {
        'u_max_t0_ms': float(u_max),
        'u_max_tT_ms': float(u_max_final),
        'speed_decay_ratio': float(u_max_final / u_max),
        'omega_max_1s': float(omega_max),
        'p_range_Pa': float(p_range),
    },
    'arc_force': {
        'disabled': _arc_disabled,
        **({
            'x_arc_mm': float(x_arc_phys * 1e3),
            'sigma_mm': float(sigma_phys * 1e3),
            'A_Nm3': float(A_phys),
        } if not _arc_disabled else {}),
    },
    'arc_force_prior': {
        'source': 'Image-estimated position and width; amplitude learned from data',
        'x_frac_measured': CONFIG['arc_x_frac_measured'],
        'x_frac_tolerance': CONFIG['arc_x_frac_tolerance'],
        'sigma_init_mm': CONFIG['arc_sigma_init_m'] * 1e3,
        'sigma_prior_log_std': CONFIG['arc_sigma_prior_log_std'],
        'amplitude_prior': 'log-normal, center={:.0f} N/m^3, ±50%'.format(
            CONFIG.get('arc_amp_prior_Nm3', 3e4)),
        'lambda_arc_amp_prior': CONFIG.get('lambda_arc_amp_prior', 0),
        'lambda_arc_pos_prior': CONFIG['lambda_arc_pos_prior'],
        'lambda_arc_sigma_prior': CONFIG['lambda_arc_sigma_prior'],
    },
    'reconstruction_errors': recon_errors,
    'final_losses': {k: float(history[k][-1])
                     for k in ['total', 'data', 'boundary', 'pde',
                               'p_ref', 'symmetry', 'arc_prior', 'energy_decay']},
}

json_path = os.path.join(results_dir, 'physical_properties.json')
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {json_path}")

# Training history
np.savez_compressed(
    os.path.join(results_dir, 'training_history.npz'),
    **{k: np.array(v) for k, v in history.items()}
)
print(f"Saved: {results_dir}/training_history.npz")

print(f"\nAll results saved to: {results_dir}/")
print(f"  fig_flow_evolution.png    - 1x3 flow fields at early/mid/late")
print(f"  fig_KE_decay.png          - kinetic energy decay profile")
print(f"  fig_pressure_evolution.png - pressure at early/mid/late")
print(f"  fig_arc_force.png         - learned arc force (constant in time)")
print(f"  fig_training_curves.png   - loss & parameter convergence")
print(f"  pinn_model.pt             - model checkpoint")
print(f"  fields_SI_t*.npz          - velocity/pressure fields per time step")
print(f"  physical_properties.json  - learned physical properties")
print(f"  training_history.npz      - loss curves data")
