# %% [Cell 0]
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
# ── Physical constants from config.py ──
PIXEL_SIZE_MM = 10.53 / 512             # mm per pixel
PIXEL_SIZE_M  = PIXEL_SIZE_MM * 1e-3    # m per pixel
FPS           = 1500                     # camera frame rate (Hz)
STRIDE        = 5                        # RAFT stride
DT            = STRIDE / FPS             # time between RAFT frame pairs (s)
RHO           = 7200.0                   # density of molten steel (kg/m^3)

# RAFT gives pixel displacement per stride frames
# velocity [m/s] = displacement [px] * pixel_size [m/px] / dt [s]
VEL_FACTOR = PIXEL_SIZE_M / DT          # m/s per (px/stride)

print(f"Pixel size : {PIXEL_SIZE_M:.4e} m/px  ({PIXEL_SIZE_MM:.4f} mm/px)")
print(f"Frame rate : {FPS} Hz,  stride = {STRIDE},  dt = {DT:.6f} s")
print(f"Vel factor : {VEL_FACTOR:.4e} m/s per (px/stride)")
print(f"Density    : {RHO:.0f} kg/m^3")

# ── PINN configuration ──
CONFIG = {
    'data_dir':              'output_0mm/RAFT_data',
    'results_dir':           'PINN_results_0mm_A_no_prior',
    'crop_padding':          10,
    'confidence_threshold':  0.4,

    # Boundary band
    'boundary_band_width':     50,
    'boundary_inner_fraction': 0.5,
    'boundary_outer_fraction': 0.5,

    # Sampling per epoch
    'n_obs_per_step':        2000,
    'n_pde_per_step':        1500,

    'n_boundary_points':     800,

    # Loss weights (from tex file)
    # 'lambda_data':           5.0,
    # 'lambda_boundary':       5.0,
    # 'lambda_pde':            2.0,
    # 'lambda_p_ref':          1.0,
    # 'lambda_symmetry':       3.0,

    'lambda_data':           5.0,
    'lambda_boundary':       2.0,
    'lambda_pde':            1.0,
    'lambda_p_ref':          0.5,
    'lambda_symmetry':       2.0,

    # Arc force (arc is still on during observation)
    'disable_arc_force': False,

    # ── Arc parameters estimated from high-speed images ──
    # Position: electrode center as fraction of centerline width (left to right)
    'arc_x_frac_init':       0.633,
    # Width: visible arc radius estimated from images (meters)
    'arc_sigma_init_m':      2.5e-3,     # ~2.5 mm from image

    # Amplitude: no prior — learned from data + PDE (0mm is most constrained case)
    'arc_amp_init':          1.0,

    # Soft priors on position and width
    'lambda_arc_pos_prior':  2.0,
    'arc_x_frac_measured':   0.726,
    'arc_x_frac_tolerance':  0.05,       # ±5% of centerline width
    'lambda_arc_sigma_prior': 1.0,
    'arc_sigma_prior_log_std': 0.3,      # allows ~1.35x variation

    # Training
    'n_epochs':              1000,
    'lr':                    1e-3,
    'lr_physics':            5e-3,
    'hidden_dims':           [128, 256, 256, 128],
    'lambda_peak':           0.1,     # peak matching penalty weight
}

# %% [Cell 2]
# ============================================================================
# REPLACEMENT CELL FOR SECTION 2: Data Loading and SI Unit Conversion
# ============================================================================
# Replace the entire Section 2 cell with this one.
# Changes:
#   - Per-frame sparse filtering: keep only the top 10% of the speed dynamic
#     range in each frame (positions with real signal, not near-zero RAFT noise)
#   - Build a pooled sparse observation dataset: (x_tilde, y_tilde, u_hat, v_hat)
#   - U = global max speed across ALL frames and ALL valid positions
#   - avg_flow is still computed for visualization but NOT used for training
# ============================================================================

def load_data(cfg):
    """Load symmetric RAFT data, extract sparse observations per frame,
    convert to SI, compute normalization scales."""
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

    # ── Pool center (pixel coords in cropped image) ──
    rows_m, cols_m = np.where(mask_crop)
    cx_px = (cols_m.min() + cols_m.max()) / 2.0
    cy_px = (rows_m.min() + rows_m.max()) / 2.0

    # ── Pool dimensions in meters ──
    W_px = cols_m.max() - cols_m.min()
    H_px = rows_m.max() - rows_m.min()
    W_m = W_px * PIXEL_SIZE_M
    H_m = H_px * PIXEL_SIZE_M
    L = max(W_m, H_m)   # characteristic length [m]

    print(f"Pool center (crop px): cx={cx_px:.1f}, cy={cy_px:.1f}")
    print(f"Pool size: {W_m*1e3:.2f} mm x {H_m*1e3:.2f} mm")
    print(f"Characteristic length L = {L*1e3:.2f} mm = {L:.4e} m")

    # ── Load flow frames ──
    flow_dir = os.path.join(data_dir, 'flow_symmetric')
    flow_files = sorted(glob.glob(os.path.join(flow_dir, '*.npy')))
    n_frames = len(flow_files)
    print(f"Flow frames: {n_frames}")

    # We still compute avg_flow for visualization purposes
    all_flow = np.zeros((n_frames, 2, h_c, w_c), dtype=np.float32)

    # ── Per-frame sparse extraction ──
    # Collect (ix, iy, u_ms, v_ms) from each frame's valid high-signal region
    sparse_ix_list = []
    sparse_iy_list = []
    sparse_u_list  = []   # m/s
    sparse_v_list  = []   # m/s
    global_max_speed = 0.0
    speed_range_pct = cfg.get('speed_range_percentile', 0.8)
    print(speed_range_pct)

    for i in tqdm(range(n_frames), desc="Loading"):
        f = np.load(flow_files[i])[:, r0:r1, c0:c1]   # (2, h, w) px/stride
        all_flow[i] = f * VEL_FACTOR                    # -> m/s

        # Speed in m/s within the mask
        mag = np.sqrt(all_flow[i, 0]**2 + all_flow[i, 1]**2)

        # Step 1: find non-trivial positions (mag > 1e-6 m/s) inside mask
        nontrivial = mask_crop & (mag > 1e-6)
        if nontrivial.sum() == 0:
            continue

        speeds_nt = mag[nontrivial]
        speed_lo = speeds_nt.min()
        speed_hi = speeds_nt.max()

        # Track global max
        if speed_hi > global_max_speed:
            global_max_speed = speed_hi

        # Step 2: keep only positions above the Pth percentile of dynamic range
        threshold = speed_lo + speed_range_pct * (speed_hi - speed_lo)
        valid = nontrivial & (mag >= threshold)

        if valid.sum() == 0:
            continue

        vy, vx = np.where(valid)
        sparse_ix_list.append(vx)
        sparse_iy_list.append(vy)
        sparse_u_list.append(all_flow[i, 0, vy, vx])
        sparse_v_list.append(all_flow[i, 1, vy, vx])

    # ── Concatenate all sparse observations ──
    sparse_ix = np.concatenate(sparse_ix_list)
    sparse_iy = np.concatenate(sparse_iy_list)
    sparse_u  = np.concatenate(sparse_u_list)    # m/s
    sparse_v  = np.concatenate(sparse_v_list)    # m/s

    # ── Characteristic velocity U = global max across all frames ──
    U = float(global_max_speed) if global_max_speed > 0 else 1e-5

    # ── Sparse obs in normalized coordinates and dimensionless velocity ──
    sparse_xt = (sparse_ix - cx_px) * PIXEL_SIZE_M / L   # x_tilde
    sparse_yt = (sparse_iy - cy_px) * PIXEL_SIZE_M / L   # y_tilde
    sparse_u_hat = sparse_u / U                           # dimensionless
    sparse_v_hat = sparse_v / U

    print(f"Characteristic velocity U = {U:.4f} m/s  (global max across all frames)")
    print(f"Scaling: S_psi = UL = {U*L:.4e} m^2/s, "
          f"S_p = rho*U^2 = {RHO*U**2:.2f} Pa")
    print(f"Sparse observations: {len(sparse_ix)} total from {n_frames} frames "
          f"(avg {len(sparse_ix)/n_frames:.0f} pts/frame, "
          f"threshold = {speed_range_pct*100:.0f}% of dynamic range)")

    # ── avg_flow for visualization only ──
    avg_flow = all_flow.mean(axis=0)

    # ── Interior points in normalized coordinates (unchanged) ──
    iy, ix = np.where(mask_crop)
    x_tilde = (ix - cx_px) * PIXEL_SIZE_M / L
    y_tilde = (iy - cy_px) * PIXEL_SIZE_M / L
    interior_coords = np.stack([x_tilde, y_tilde], axis=1)

    # ── Centerline points (unchanged) ──
    cl_tol = 4.0 * PIXEL_SIZE_M / L
    cl_mask = np.abs(y_tilde) < cl_tol
    cl_pts = interior_coords[cl_mask]
    cl_x_min = float(cl_pts[:, 0].min()) if len(cl_pts) > 0 else -0.5
    cl_x_max = float(cl_pts[:, 0].max()) if len(cl_pts) > 0 else  0.5
    print(f"Centerline: {cl_pts.shape[0]} points, "
          f"x_tilde in [{cl_x_min:.3f}, {cl_x_max:.3f}]")

    # ── Contour for visualization (unchanged) ──
    contours, _ = cv2.findContours(mask_crop.astype(np.uint8),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea).squeeze().astype(float)

    return {
        'mask_crop': mask_crop, 'crop': (r0, r1, c0, c1),
        'h_crop': h_c, 'w_crop': w_c,
        'cx_px': cx_px, 'cy_px': cy_px,
        'L': L, 'U': U,
        'avg_flow': avg_flow, 'n_frames': n_frames,
        # Sparse observation dataset (NEW)
        'sparse_xt': sparse_xt, 'sparse_yt': sparse_yt,
        'sparse_u_hat': sparse_u_hat, 'sparse_v_hat': sparse_v_hat,
        # Kept for PDE collocation and other losses
        'interior_coords': interior_coords,
        'interior_ix': ix, 'interior_iy': iy,
        'cl_pts': cl_pts,
        'cl_x_min': cl_x_min, 'cl_x_max': cl_x_max,
        'bnd_cnt_px': cnt,
    }

data = load_data(CONFIG)


# %% [markdown Cell 3]
# ## 3. Soft Boundary Band
# 
# Creates a weighted boundary band for soft no-penetration enforcement:
# $\mathcal{L}_{\text{bnd}} = \frac{1}{N} \sum \beta_j (\mathbf{u}_j \cdot \mathbf{n}_j)^2$

# %% [Cell 4]
def create_boundary_band(mask, band_width, inner_frac, outer_frac,
                          cx_px, cy_px, L):
    """Soft boundary band in normalized coordinates."""
    dist_inside  = distance_transform_edt(mask)
    dist_outside = distance_transform_edt(~mask)
    signed_dist  = dist_outside - dist_inside

    in_band = np.abs(signed_dist) <= band_width
    by, bx = np.where(in_band)

    # Per-point weights (1 near edge, 0 far away)
    distances = np.abs(signed_dist[by, bx])
    inner_threshold = band_width * inner_frac
    weights = np.ones(len(distances))
    outer = distances > inner_threshold
    weights[outer] = 1.0 - (distances[outer] - inner_threshold) / \
                            (band_width - inner_threshold)
    weights = np.clip(weights, 0.0, 1.0)

    # Outward normals from gradient of signed distance
    grad_y, grad_x = np.gradient(signed_dist)
    normals = np.stack([grad_x[by, bx], grad_y[by, bx]], axis=1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    normals = normals / norms

    # Normalized physical coordinates
    x_tilde = (bx - cx_px) * PIXEL_SIZE_M / L
    y_tilde = (by - cy_px) * PIXEL_SIZE_M / L
    pts = np.stack([x_tilde, y_tilde], axis=1)

    print(f"Boundary band: {len(bx)} pts, "
          f"weight mean={weights.mean():.3f}")
    return pts, normals, weights

bnd_pts, bnd_normals, bnd_weights = create_boundary_band(
    data['mask_crop'],
    CONFIG['boundary_band_width'],
    CONFIG['boundary_inner_fraction'],
    CONFIG['boundary_outer_fraction'],
    data['cx_px'], data['cy_px'], data['L'])

# %% [markdown Cell 5]
# ## 4. Neural Network Architecture
# 
# **Data flow** (from tex file Section 5):
# 
# $$
# (x,y) \xrightarrow{\div L} (\tilde{x}, \tilde{y})
# \xrightarrow{\text{MLP}} (\hat{\psi}, \hat{p})
# \xrightarrow{\text{scale}} \psi = \hat{\psi}\cdot UL,\; p = \hat{p}\cdot\rho U^2
# \xrightarrow{\partial} u = U\,\partial\hat{\psi}/\partial\tilde{y}
# $$
# 
# The PDE is solved in **dimensionless** form for numerical conditioning:
# $$\hat{u}\,\partial\hat{u}/\partial\tilde{x} + \hat{v}\,\partial\hat{u}/\partial\tilde{y}
# = -\partial\hat{p}/\partial\tilde{x}
# + \tilde{\nu}\,\tilde{\nabla}^2\hat{u}
# + \tilde{F}_x$$
# 
# where $\tilde{\nu} = \nu/(UL)$ is the learned dimensionless viscosity.

# %% [Cell 6]
class FlowNet(nn.Module):
    """MLP: (x_tilde, y_tilde) -> (psi_hat, p_hat)  [dimensionless]."""

    def __init__(self, hidden_dims):
        super().__init__()
        layers = []
        in_dim = 2
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy):
        out = self.net(xy)
        return out[:, 0:1], out[:, 1:2]   # psi_hat, p_hat


def get_velocity_hat(model, xy):
    """Dimensionless velocity from stream function.

    u_hat = d(psi_hat)/d(y_tilde)
    v_hat = -d(psi_hat)/d(x_tilde)

    Physical: u = U * u_hat,  v = U * v_hat
    """
    xy = xy.clone().requires_grad_(True)
    psi_hat, p_hat = model(xy)
    grad = torch.autograd.grad(
        psi_hat, xy, torch.ones_like(psi_hat), create_graph=True)[0]
    u_hat =  grad[:, 1:2]
    v_hat = -grad[:, 0:1]
    return u_hat, v_hat, p_hat, xy

# %% [Cell 7]
class ArcForce(nn.Module):
    """Gaussian body force in normalized coordinates.

    F_tilde_x = A_tilde * (x_tilde - x_arc) / sigma_tilde^2
                * exp(-((x_tilde-x_arc)^2 + y_tilde^2) / (2 sigma_tilde^2))

    Physical conversion:
        x_arc_phys = x_arc_tilde * L
        sigma_phys = sigma_tilde * L
        A_phys     = A_tilde * rho * U^2  [N/m^3]
    """

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

# %% [markdown Cell 8]
# ## 5. Navier-Stokes Residual (Dimensionless)
# 
# Residuals of the non-dimensionalized steady incompressible N-S:
# 
# $$\tilde{r}_x = \hat{u}\frac{\partial\hat{u}}{\partial\tilde{x}}
# + \hat{v}\frac{\partial\hat{u}}{\partial\tilde{y}}
# + \frac{\partial\hat{p}}{\partial\tilde{x}}
# - \tilde{\nu}\tilde{\nabla}^2\hat{u}
# - \tilde{F}_x$$

# %% [Cell 9]
def compute_ns_residual(model, arc_force, xy, log_nu_hat, cfg):
    """Dimensionless steady N-S residual at collocation points."""
    nu_hat = torch.exp(log_nu_hat)
    xy = xy.clone().requires_grad_(True)
    psi_hat, p_hat = model(xy)
    ones = torch.ones_like(psi_hat)

    g_psi = torch.autograd.grad(psi_hat, xy, ones, create_graph=True, retain_graph=True)[0]
    u_hat =  g_psi[:, 1:2]
    v_hat = -g_psi[:, 0:1]

    g_p = torch.autograd.grad(p_hat, xy, ones, create_graph=True, retain_graph=True)[0]
    dp_dx, dp_dy = g_p[:, 0:1], g_p[:, 1:2]

    g_u = torch.autograd.grad(u_hat, xy, ones, create_graph=True, retain_graph=True)[0]
    du_dx, du_dy = g_u[:, 0:1], g_u[:, 1:2]

    g_v = torch.autograd.grad(v_hat, xy, ones, create_graph=True, retain_graph=True)[0]
    dv_dx, dv_dy = g_v[:, 0:1], g_v[:, 1:2]

    d2u_dx2 = torch.autograd.grad(du_dx, xy, ones, create_graph=True, retain_graph=True)[0][:, 0:1]
    d2u_dy2 = torch.autograd.grad(du_dy, xy, ones, create_graph=True, retain_graph=True)[0][:, 1:2]
    d2v_dx2 = torch.autograd.grad(dv_dx, xy, ones, create_graph=True, retain_graph=True)[0][:, 0:1]
    d2v_dy2 = torch.autograd.grad(dv_dy, xy, ones, create_graph=True)[0][:, 1:2]

    if cfg.get('disable_arc_force', False):
        Fx = torch.zeros_like(u_hat)
        Fy = torch.zeros_like(u_hat)
    else:
        Fx, Fy = arc_force(xy)

    res_x = (u_hat * du_dx + v_hat * du_dy
             + dp_dx - nu_hat * (d2u_dx2 + d2u_dy2) - Fx)
    res_y = (u_hat * dv_dx + v_hat * dv_dy
             + dp_dy - nu_hat * (d2v_dx2 + d2v_dy2) - Fy)
    return res_x, res_y

# %% [markdown Cell 10]
# ## 6. Training
# 
# Five-loss training (tex file Section 7):
# 
# $$\mathcal{L} = \lambda_{\text{data}}\mathcal{L}_{\text{data}}
# + \lambda_{\text{bnd}}\mathcal{L}_{\text{bnd}}
# + \lambda_{\text{PDE}}\mathcal{L}_{\text{PDE}}
# + \lambda_p \mathcal{L}_p
# + \lambda_{\text{sym}}\mathcal{L}_{\text{sym}}$$

# %% [Cell 11]
# ============================================================================
# REPLACEMENT CELL FOR SECTION 6: Training
# ============================================================================
# Changes from previous version:
#   + Added L_arc_prior: soft priors on arc position, width, and amplitude
#   + Records 'arc_prior' in history
#   + Prints sigma_phys in mm every 100 epochs
# ============================================================================

def train(model, arc_force, data, bnd_data, cfg):
    U, L = data['U'], data['L']
    mask  = data['mask_crop']
    h_c, w_c = data['h_crop'], data['w_crop']

    # Tensors on device
    bnd_pts_t = torch.tensor(bnd_data[0], dtype=torch.float32, device=DEVICE)
    bnd_n_t   = torch.tensor(bnd_data[1], dtype=torch.float32, device=DEVICE)
    bnd_w_t   = torch.tensor(bnd_data[2], dtype=torch.float32, device=DEVICE)
    all_int   = torch.tensor(data['interior_coords'],
                             dtype=torch.float32, device=DEVICE)
    cl_pts    = torch.tensor(data['cl_pts'],
                             dtype=torch.float32, device=DEVICE)

    # ── Sparse observation dataset ──
    obs_xt    = data['sparse_xt']       # (N_total,) normalized x
    obs_yt    = data['sparse_yt']       # (N_total,) normalized y
    obs_u_hat = data['sparse_u_hat']    # (N_total,) dimensionless u
    obs_v_hat = data['sparse_v_hat']    # (N_total,) dimensionless v
    n_obs     = len(obs_xt)
    print(f"Sparse observation points: {n_obs}")

    # Pressure reference at pool centre (tilde = 0, 0)
    p_ref = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=DEVICE)

    # ── Learnable dimensionless viscosity: nu_tilde = nu / (UL) ──
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

    print(f"Arc priors:")
    print(f"  sigma_arc = {cfg['arc_sigma_init_m']*1e3:.2f} mm  ->  sigma_tilde = {sigma_target_tilde:.4f}")
    print(f"  x_arc_frac = {cfg['arc_x_frac_measured']:.3f}  ->  x_tilde = {x_arc_target_tilde:.4f}")
    print(f"  amplitude: free (learned from data + PDE)")

    opt_groups = [
        {'params': model.parameters(), 'lr': cfg['lr']},
        {'params': [log_nu_hat],       'lr': cfg['lr_physics']},
    ]
    if not cfg.get('disable_arc_force', False):
        opt_groups.append({'params': arc_force.parameters(), 'lr': cfg['lr_physics']})
    optimizer = torch.optim.Adam(opt_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg['n_epochs'])

    keys = ['total', 'data', 'boundary', 'pde', 'p_ref', 'symmetry',
            'arc_prior',
            'nu_hat', 'nu_phys', 'arc_x', 'arc_sigma', 'arc_amp']
    history = {k: [] for k in keys}

    for epoch in tqdm(range(cfg['n_epochs']), desc="Training"):
        optimizer.zero_grad()

        # ── L_data: sample from sparse observations ──
        n_s = min(cfg['n_obs_per_step'], n_obs)
        sel = np.random.choice(n_obs, n_s, replace=False)
        xy_d = torch.tensor(
            np.stack([obs_xt[sel], obs_yt[sel]], axis=1),
            dtype=torch.float32, device=DEVICE)
        up, vp, _, _ = get_velocity_hat(model, xy_d)
        ut = torch.tensor(obs_u_hat[sel], dtype=torch.float32, device=DEVICE)
        vt = torch.tensor(obs_v_hat[sel], dtype=torch.float32, device=DEVICE)
        # Speed-based weights: high-velocity obs contribute more
        speed_obs = torch.sqrt(ut**2 + vt**2)                  # dimensionless
        w_speed = speed_obs / (speed_obs.mean() + 1e-8)        # normalize so mean weight ~1

        L_data = (w_speed * (up.squeeze() - ut)**2).mean() + \
                 (w_speed * (vp.squeeze() - vt)**2).mean()
        # Peak matching penalty: penalize underestimation of max speed
        speed_pred = torch.sqrt(up.squeeze()**2 + vp.squeeze()**2)
        L_peak = (speed_obs.max() - speed_pred.max())**2
        L_data = L_data + cfg.get('lambda_peak', 1.0) * L_peak

        # ── L_boundary (soft no-penetration) — unchanged ──
        L_bnd = torch.tensor(0.0, device=DEVICE)
        if cfg['lambda_boundary'] > 0:
            nb = min(cfg['n_boundary_points'], bnd_pts_t.shape[0])
            ib = torch.randperm(bnd_pts_t.shape[0])[:nb]
            ub, vb, _, _ = get_velocity_hat(model, bnd_pts_t[ib])
            u_n = ub.squeeze() * bnd_n_t[ib, 0] + vb.squeeze() * bnd_n_t[ib, 1]
            L_bnd = (bnd_w_t[ib] * u_n**2).mean()

        # ── L_pde (dimensionless N-S residual) ──
        # Always computed with full graph for monitoring.
        # Detached when lambda_pde=0 so it doesn't affect training.
        nc = min(cfg['n_pde_per_step'], all_int.shape[0])
        ic = torch.randperm(all_int.shape[0])[:nc]
        rx, ry = compute_ns_residual(
            model, arc_force, all_int[ic], log_nu_hat, cfg)
        L_pde = (rx**2).mean() + (ry**2).mean()
        if cfg['lambda_pde'] == 0:
            L_pde = L_pde.detach()

        # ── L_p_ref (pressure gauge) — unchanged ──
        _, _, p_at_ref, _ = get_velocity_hat(model, p_ref)
        L_p_ref = p_at_ref.squeeze()**2

        # ── L_symmetry (v = 0 on centerline) — unchanged ──
        L_sym = torch.tensor(0.0, device=DEVICE)
        if cfg['lambda_symmetry'] > 0 and cl_pts.shape[0] > 0:
            ncl = min(200, cl_pts.shape[0])
            icl = torch.randperm(cl_pts.shape[0])[:ncl]
            _, vcl, _, _ = get_velocity_hat(model, cl_pts[icl])
            L_sym = (vcl**2).mean()

        # ── L_arc_prior: position and width from image, amplitude with rough prior ──
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

            # Amplitude: NO prior — learned entirely from data + PDE
            # (0mm steady-state case provides the most constrained estimate)

        loss = (cfg['lambda_data']     * L_data +
                cfg['lambda_boundary'] * L_bnd  +
                cfg['lambda_pde']      * L_pde  +
                cfg['lambda_p_ref']    * L_p_ref +
                cfg['lambda_symmetry'] * L_sym +
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
        history['arc_prior'].append(L_arc_prior.item())
        history['nu_hat'].append(nu_hat_val)
        history['nu_phys'].append(nu_phys_val)
        history['arc_x'].append(xa)
        history['arc_sigma'].append(sa)
        history['arc_amp'].append(aa)

        if (epoch + 1) % 100 == 0:
            arc_str = f"  sigma={sa*L*1e3:.2f}mm" if not cfg.get('disable_arc_force', False) else ""
            print(f"  Ep {epoch+1:4d}  loss={loss.item():.5f}  "
                  f"data={L_data.item():.5f}  pde={L_pde.item():.5f}"
                  f"{arc_str}  "
                  f"nu_hat={nu_hat_val:.5f}  "
                  f"nu_phys={nu_phys_val:.2e} m^2/s")

    # ── Final summary ──
    nu_hat_f  = torch.exp(log_nu_hat).item()
    nu_phys_f = nu_hat_f * U * L
    mu_phys_f = nu_phys_f * RHO
    Re        = 1.0 / nu_hat_f
    xa, sa, aa = arc_force.get_params()

    print(f"\n{'='*60}")
    print(f"LEARNED PHYSICS PARAMETERS (after {cfg['n_epochs']} epochs)")
    print(f"{'='*60}")
    print(f"  nu_hat  (dimensionless) = {nu_hat_f:.6f}")
    print(f"  nu_phys = {nu_phys_f:.4e} m^2/s")
    print(f"  mu_phys = {mu_phys_f:.4e} Pa.s  ({mu_phys_f*1e3:.3f} mPa.s)")
    print(f"  Re      = {Re:.1f}")
    if not cfg.get('disable_arc_force', False):
        print(f"  Arc: x_tilde={xa:.4f}  sigma_tilde={sa:.4f}  A_tilde={aa:.4f}")
        print(f"  Arc sigma_phys = {sa*L*1e3:.2f} mm  (image est: "
              f"{cfg['arc_sigma_init_m']*1e3:.1f} mm)")
    else:
        print(f"  Arc force: DISABLED")
    print(f"{'='*60}")

    return history, log_nu_hat

# %% [Cell 12]
# ── Build model and train ──
model = FlowNet(CONFIG['hidden_dims']).to(DEVICE)
print(f"FlowNet: {sum(p.numel() for p in model.parameters()):,} params")
print(f"Architecture: [2] -> {CONFIG['hidden_dims']} -> [2]")

# ── Arc force initial values from image measurements ──
_sigma_init_tilde = CONFIG['arc_sigma_init_m'] / data['L']
print(f"Arc force init: x_frac={CONFIG['arc_x_frac_init']:.3f}, "
      f"sigma={CONFIG['arc_sigma_init_m']*1e3:.1f} mm (tilde={_sigma_init_tilde:.4f}), "
      f"amp={CONFIG['arc_amp_init']} (learned from data + PDE)")

arc_force = ArcForce(
    CONFIG['arc_x_frac_init'],
    _sigma_init_tilde,           # image-estimated sigma (in tilde coords)
    CONFIG['arc_amp_init'],      # arbitrary init — learned from data + PDE
    y_center=0.0,
    x_min=data['cl_x_min'],
    x_max=data['cl_x_max'],
).to(DEVICE)

history, log_nu_hat = train(
    model, arc_force, data,
    (bnd_pts, bnd_normals, bnd_weights), CONFIG)

# %% [markdown Cell 13]
# ## 7. Dense Prediction in Physical (SI) Units
# 
# Convert network output to physical fields:
# - $u = U \cdot \hat{u}$ [m/s]
# - $p = \rho U^2 \cdot \hat{p}$ [Pa]

# %% [Cell 14]
@torch.enable_grad()
def predict_full_SI(model, data):
    """Dense (u, v, p) over cropped mask in SI units."""
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
        xy = torch.tensor(np.stack([x_t, y_t], axis=1),
                          dtype=torch.float32, device=DEVICE)
        xy.requires_grad_(True)

        psi_hat, p_hat = model(xy)
        grad = torch.autograd.grad(
            psi_hat, xy, torch.ones_like(psi_hat))[0]

        u_full[by, bx] =  U * grad[:, 1].detach().cpu().numpy()     # m/s
        v_full[by, bx] = -U * grad[:, 0].detach().cpu().numpy()     # m/s
        p_full[by, bx] = RHO * U**2 * p_hat.squeeze().detach().cpu().numpy()  # Pa

    return u_full, v_full, p_full

u_pred, v_pred, p_pred = predict_full_SI(model, data)
mask = data['mask_crop']
speed = np.sqrt(u_pred**2 + v_pred**2)

print(f"Velocity  u : [{u_pred[mask].min():.4f}, {u_pred[mask].max():.4f}] m/s")
print(f"Velocity  v : [{v_pred[mask].min():.4f}, {v_pred[mask].max():.4f}] m/s")
print(f"Speed  max  : {speed[mask].max():.4f} m/s")
print(f"Pressure    : [{p_pred[mask].min():.2f}, {p_pred[mask].max():.2f}] Pa")

# %% [markdown Cell 15]
# ## 8. Visualization

# %% [Cell 16]
# ============================================================================
# Publication-quality figures for PINN results
# ============================================================================
# Replace all visualization cells in Section 8 with these two figure blocks.
#
# Adjustable parameters are collected at the top for easy tuning.
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

# ═══════════════════════════════════════════════════════════════════════
# ── Adjustable parameters ──
# ═══════════════════════════════════════════════════════════════════════
FIG1_SIZE       = (7.0, 2.6)     # (width, height) in inches — error figure
FIG2_SIZE       = (7.0, 2.6)     # (width, height) in inches — result figure
BOUNDARY_LW     = 0.8            # pool boundary line width
STREAMLINE_LW   = 0.5            # streamline line width
STREAMLINE_DENS = 1.5            # streamline density
STREAMLINE_ARROW = 0.4           # streamline arrowhead size
SPEED_ALPHA      = 0.6           # speed colormap transparency (0=invisible, 1=opaque)
SUBPLOT_BORDER_LW = 0.5         # subplot border (spine) thickness
TICK_VALUES_MM  = [0, 5, 10]     # x and y axis ticks in mm
# ═══════════════════════════════════════════════════════════════════════

results_dir = CONFIG['results_dir']
os.makedirs(results_dir, exist_ok=True)

cnt_px = data['bnd_cnt_px']
cnt_c  = np.vstack([cnt_px, cnt_px[:1]])
h_c, w_c = data['h_crop'], data['w_crop']
U, L = data['U'], data['L']

# ── Coordinate grids in mm ──
x_mm = np.arange(w_c) * PIXEL_SIZE_M * 1e3
y_mm = np.arange(h_c) * PIXEL_SIZE_M * 1e3
cnt_x_mm = cnt_c[:, 0] * PIXEL_SIZE_M * 1e3
cnt_y_mm = cnt_c[:, 1] * PIXEL_SIZE_M * 1e3

# Helper: common axis formatting
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
# FIGURE 1: Error Illustration — (a) RAFT  (b) PINN  (c) Error
# ████████████████████████████████████████████████████████████████████████████

# ── Build per-pixel max-speed map from sparse-filtered frames ──
flow_dir = os.path.join(CONFIG['data_dir'], 'flow_symmetric')
flow_files = sorted(glob.glob(os.path.join(flow_dir, '*.npy')))
r0, r1, c0, c1 = data['crop']

max_speed_map = np.zeros((h_c, w_c), dtype=np.float32)
max_u_map     = np.zeros((h_c, w_c), dtype=np.float32)
max_v_map     = np.zeros((h_c, w_c), dtype=np.float32)
obs_count_map = np.zeros((h_c, w_c), dtype=np.int32)
speed_range_pct = CONFIG.get('speed_range_percentile', 0.9)

for i in tqdm(range(len(flow_files)), desc="Building RAFT ref"):
    f = np.load(flow_files[i])[:, r0:r1, c0:c1]
    u_frame = f[0] * VEL_FACTOR
    v_frame = f[1] * VEL_FACTOR
    mag = np.sqrt(u_frame**2 + v_frame**2)

    nontrivial = mask & (mag > 1e-6)
    if nontrivial.sum() == 0:
        continue
    speeds_nt = mag[nontrivial]
    speed_lo, speed_hi = speeds_nt.min(), speeds_nt.max()
    threshold = speed_lo + speed_range_pct * (speed_hi - speed_lo)
    valid = nontrivial & (mag >= threshold)
    if valid.sum() == 0:
        continue
    obs_count_map[valid] += 1
    better = valid & (mag > max_speed_map)
    max_speed_map[better] = mag[better]
    max_u_map[better] = u_frame[better]
    max_v_map[better] = v_frame[better]

print(f"Observed pixels: {(obs_count_map > 0).sum()} / {mask.sum()}")

# Common color scale
obs_mask = obs_count_map > 0
vmax_speed = max(max_speed_map[obs_mask].max() if obs_mask.any() else 0,
                 speed[mask].max())

# Error map
diff = np.sqrt((u_pred - max_u_map)**2 + (v_pred - max_v_map)**2)

fig1, (ax_r, ax_pinn, ax_err) = plt.subplots(1, 3, figsize=FIG1_SIZE)

# (a) RAFT sparse max speed
raft_display = np.where(obs_mask, max_speed_map, np.nan)
im_r = ax_r.pcolormesh(x_mm, y_mm, raft_display,
                        cmap='jet', shading='auto',
                        vmin=0, vmax=vmax_speed, alpha=SPEED_ALPHA)
ax_r.plot(cnt_x_mm, cnt_y_mm, 'k-', lw=BOUNDARY_LW)
cb_r = plt.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)
cb_r.set_label('m/s')
cb_r.outline.set_visible(False)
_format_ax(ax_r, 'RAFT observation')

# (b) PINN speed magnitude
pinn_display = np.where(mask, speed, np.nan)
im_pinn = ax_pinn.pcolormesh(x_mm, y_mm, pinn_display,
                              cmap='jet', shading='auto',
                              vmin=0, vmax=vmax_speed, alpha=SPEED_ALPHA)
ax_pinn.plot(cnt_x_mm, cnt_y_mm, 'k-', lw=BOUNDARY_LW)
cb_pinn = plt.colorbar(im_pinn, ax=ax_pinn, fraction=0.046, pad=0.04)
cb_pinn.set_label('m/s')
cb_pinn.outline.set_visible(False)
_format_ax(ax_pinn, 'PINN reconstruction')

# (c) Speed error (only where observations exist)
diff_display = np.where(obs_mask, diff, np.nan)
im_err = ax_err.pcolormesh(x_mm, y_mm, diff_display,
                            cmap='hot', shading='auto')
ax_err.plot(cnt_x_mm, cnt_y_mm, 'k-', lw=BOUNDARY_LW)
cb_err = plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
cb_err.set_label('m/s')
cb_err.outline.set_visible(False)
_format_ax(ax_err, 'Speed error')

plt.tight_layout(w_pad=0.4)
fig1.savefig(os.path.join(results_dir, 'fig_error_illustration.png'))
plt.show()
plt.close(fig1)
print(f"Saved: {results_dir}/fig_error_illustration.png")


# ████████████████████████████████████████████████████████████████████████████
# FIGURE 2: Result Demonstration — (a) Flow field  (b) Arc force  (c) Pressure
# ████████████████████████████████████████████████████████████████████████████

# ── Compute arc force field in SI (only when arc force is enabled) ──
_arc_disabled = CONFIG.get('disable_arc_force', False)
xa, sa, aa = arc_force.get_params()
if not _arc_disabled:
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
    arc_cx_px = data['cx_px'] + xa * L / PIXEL_SIZE_M
    arc_cx_mm = arc_cx_px * PIXEL_SIZE_M * 1e3
    arc_cy_mm = data['cy_px'] * PIXEL_SIZE_M * 1e3

fig2, (ax_v, ax_f, ax_p) = plt.subplots(1, 3, figsize=FIG2_SIZE)

# (a) Flow field: streamlines + speed color + arc center
speed_display = np.where(mask, speed, np.nan)
im_v = ax_v.pcolormesh(x_mm, y_mm, speed_display,
                        cmap='jet', shading='auto',
                        vmin=0, vmax=vmax_speed, alpha=SPEED_ALPHA)
ax_v.plot(cnt_x_mm, cnt_y_mm, 'k-', lw=BOUNDARY_LW)
u_p = np.where(mask, u_pred, 0)
v_p = np.where(mask, v_pred, 0)
Xd_mm, Yd_mm = np.meshgrid(x_mm, y_mm)
ax_v.streamplot(Xd_mm, Yd_mm, u_p, v_p, color='k',
                linewidth=STREAMLINE_LW, density=STREAMLINE_DENS,
                arrowsize=STREAMLINE_ARROW, arrowstyle='->')
if not _arc_disabled:
    ax_v.plot(arc_cx_mm, arc_cy_mm, 'r*', ms=7, mew=0.8, zorder=5)
cb_v = plt.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)
cb_v.set_label('m/s')
cb_v.outline.set_visible(False)
_format_ax(ax_v, 'Flow field')

# (b) Arc force distribution (display in units of 10^4 N/m^3)
_cmap_arc = matplotlib.colors.LinearSegmentedColormap.from_list(
    'white_yellow_red', [(0, 'white'), (0.25, 'lightyellow'), (0.5, 'orangered'), (1.0, 'darkred')])
if not _arc_disabled:
    from matplotlib.patches import Polygon as MplPolygon
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
    from matplotlib.ticker import MaxNLocator
    cb_f.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _format_ax(ax_f, 'Arc force')
else:
    ax_f.text(0.5, 0.5, 'Arc force\n(disabled)',
              ha='center', va='center', transform=ax_f.transAxes,
              fontsize=10, color='gray')
    ax_f.set_axis_off()

# (c) Pressure distribution (vmin=0 since p >= 0)
p_display = np.where(mask, p_pred, np.nan)
im_p = ax_p.pcolormesh(x_mm, y_mm, p_display,
                        cmap='RdBu_r', shading='auto',
                        vmin=0)
ax_p.plot(cnt_x_mm, cnt_y_mm, 'k-', lw=BOUNDARY_LW)
cb_p = plt.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04)
cb_p.set_label('Pa')
cb_p.outline.set_visible(False)
_format_ax(ax_p, 'Pressure')

plt.tight_layout(w_pad=0.4)
fig2.savefig(os.path.join(results_dir, 'fig_result_demonstration.png'))
plt.show()
plt.close(fig2)
print(f"Saved: {results_dir}/fig_result_demonstration.png")

# %% [Cell 17]
# ============================================================================
# REPLACEMENT CELL: Training Convergence Figure (2 subplots)
# ============================================================================
# (a) Unweighted individual loss terms (log scale)
# (b) Learned physical parameters: nu, x_arc, sigma_arc, A_arc (dual y-axis)
#
# Uses the same rcParams / font style as the other publication figures.
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

import matplotlib.pyplot as plt
import numpy as np
import os

results_dir = CONFIG['results_dir']
epochs = np.arange(1, len(history['total']) + 1)
xticks = [0, 200, 400, 600, 800]

# ── Adjustable legend positions ──────────────────────────────────────────────
# bbox_to_anchor uses (x, y) in axes fraction coordinates:
#   x: 0 = left edge, 1 = right edge
#   y: 0 = bottom edge, 1 = top edge
# Increase y to move up, decrease to move down.
LEGEND_POS_LOSS   = (0.3, 0.25)    # (a) loss panel: centered, near top
LEGEND_POS_PARAM  = (0.98, 0.95)  # (b) param panel: right side, upper-mid

# ── Smoothing helper (edge-safe moving average) ──
def smooth(y, window=1):
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

# ═════════════════════════════════════════════════════════════════════════════
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7, 2.4))

# ─────────────────────────────────────────────────────────────────────────────
# (a) Unweighted individual losses (log scale)
# ─────────────────────────────────────────────────────────────────────────────
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
ax_a.semilogy(epochs, smooth(history['symmetry']),
              '-', color='#E08214', lw=0.9,
              label=r'$\mathcal{L}_{\mathrm{sym}}$')
ax_a.semilogy(epochs, smooth(history['p_ref']),
              '-', color='#762A83', lw=0.9,
              label=r'$\mathcal{L}_{p}$')
ax_a.set_xlabel('Epoch')
ax_a.set_ylabel('Loss')
ax_a.set_title('Training losses', pad=4)
ax_a.legend(loc='upper center', bbox_to_anchor=LEGEND_POS_LOSS,
            framealpha=0.9, edgecolor='none',
            handlelength=1.2, handletextpad=0.3, borderpad=0.3,
            labelspacing=0.2, columnspacing=0.8, ncol=3)
ax_a.set_xlim(0, len(epochs))
ax_a.set_xticks(xticks)
ax_a.grid(True, alpha=0.2, linewidth=0.5)
ax_a.tick_params(axis='both', pad=2)

# ─────────────────────────────────────────────────────────────────────────────
# (b) Learned physical parameters (dual y-axis)
#     Left:  nu (× 10^-5 m²/s)  and  x_arc (mm)
#     Right: sigma_arc (mm)  and  A_tilde (dimensionless)
# ─────────────────────────────────────────────────────────────────────────────
L_val = data['L']

nu_phys_arr = np.array(history['nu_phys']) * 1e5            # × 10^-5 m²/s
x_arc_mm    = np.array(history['arc_x']) * L_val * 1e3      # mm
sigma_mm    = np.array(history['arc_sigma']) * L_val * 1e3   # mm
arc_amp_SI  = np.array(history['arc_amp']) * RHO * data['U']**2 / L_val  # N/m³
arc_amp_1e4 = arc_amp_SI * 1e-4                              # × 10⁴ N/m³

color_nu  = '#2166AC'
color_xa  = '#762A83'
color_sig = '#B2182B'
color_amp = '#1B7837'

# Left y-axis: nu and x_arc
ax_b.plot(epochs, smooth(nu_phys_arr), '-', color=color_nu, lw=1.0,
          label=r'$\nu$ ($\times 10^{-5}$ m$^2$/s)')
ax_b.plot(epochs, smooth(x_arc_mm), '-', color=color_xa, lw=1.0,
          label=r'$x_{\mathrm{arc}}$ (mm)')
ax_b.set_xlabel('Epoch')
ax_b.set_ylabel('Value')
ax_b.tick_params(axis='both', pad=2)
ax_b.set_xlim(0, len(epochs))
ax_b.set_xticks(xticks)

# Right y-axis: sigma_arc (mm) and A_tilde
ax_r = ax_b.twinx()
ax_r.plot(epochs, smooth(sigma_mm), '--', color=color_sig, lw=1.0,
          label=r'$\sigma_{\mathrm{arc}}$ (mm)')
ax_r.plot(epochs, smooth(arc_amp_1e4), '--', color=color_amp, lw=1.0,
          label=r'$A$ ($\times 10^{4}$ N/m$^3$)')
ax_r.set_ylabel('Value')
ax_r.tick_params(axis='y', pad=2)

ax_b.set_title('Learned parameters', pad=4)
ax_b.grid(True, alpha=0.2, linewidth=0.5)

# Combined legend for both axes
lines_l, labels_l = ax_b.get_legend_handles_labels()
lines_r, labels_r = ax_r.get_legend_handles_labels()
ax_b.legend(lines_l + lines_r, labels_l + labels_r,
            loc='upper right', bbox_to_anchor=LEGEND_POS_PARAM,
            framealpha=0.9, edgecolor='none',
            handlelength=1.5, handletextpad=0.4, borderpad=0.3,
            labelspacing=0.25)

# ═════════════════════════════════════════════════════════════════════════════
plt.tight_layout(w_pad=0.8)
fig.savefig(os.path.join(results_dir, 'fig_training_curves.png'), dpi=300)
plt.show()
plt.close(fig)
print(f"Saved: {results_dir}/fig_training_curves.png")

# %% [markdown Cell 18]
# ## 9. Physical Property Extraction and Validation
# 
# Convert learned parameters to physical units and compare with
# literature values for TIG welding of steel (tex file Section 9).
# 
# | Property | Formula | Expected (TIG steel) |
# |----------|---------|---------------------|
# | $\nu_{\text{phys}}$ | $\tilde{\nu} \cdot U \cdot L$ | ~ 1e-6 m^2/s |
# | $\mu$ | $\nu \cdot \rho$ | 4-7 mPa.s |
# | Re | $1/\tilde{\nu}$ | 100-500 |
# | $u_{\max}$ | $\max|\mathbf{u}|$ | 0.1-0.5 m/s |
# | $\sigma_{\text{arc}}$ | $\tilde{\sigma} \cdot L$ | 2-4 mm |
# | $A_{\text{arc}}$ | $\tilde{A} \cdot \rho U^2$ | 1e4 - 1e5 N/m^3 |

# %% [Cell 19]
# ============================================================================
# REPLACEMENT CELL FOR SECTION 9: Physical Property Extraction and Validation
# ============================================================================
# Changes:
#   1. Arc force de-normalization FIXED:
#      The dimensionless PDE is:
#        û ∂û/∂x̃ + v̂ ∂û/∂ỹ = -∂p̂/∂x̃ + ν̃ ∇̃²û + F̃_x
#      Multiply through by ρU²/L to recover physical N-S:
#        ... = ... + (ρU²/L) * F̃_x
#      So:  F_phys [N/m³] = A_tilde * (ρU²/L), NOT A_tilde * ρU²
#
#   2. Literature values updated for HIGH-TEMPERATURE TIG with filler wire:
#      - Liquid iron viscosity at ~1800-2000K: μ ≈ 4-7 mPa·s (Assael et al. 2006)
#        At TIG pool temperatures (>2000K), viscosity drops further.
#        The "4-7 mPa·s" range is near the melting point (~1811K).
#        At peak pool temperatures (2200-2800K), μ can be 2-5 mPa·s.
#      - With filler wire on surface: velocities can be significantly higher
#        than pure autogenous TIG (0.1-0.5 m/s). Wire-fed TIG can reach
#        0.2-1.2 m/s (Chen et al. 2025, various CMT/TIG-wire CFD studies).
#      - Arc force amplitude with filler wire interaction: can be higher.
#
#   3. Added diagnostic: checks self-consistency between nu_hat, U, L, Re
# ============================================================================

U, L = data['U'], data['L']
nu_hat_final = torch.exp(log_nu_hat).item()
xa, sa, aa   = arc_force.get_params()
_arc_disabled = CONFIG.get('disable_arc_force', False)

# ═══════════════════════════════════════════════════════════════════════
# De-normalization verification
# ═══════════════════════════════════════════════════════════════════════
#
# The PINN works in dimensionless variables:
#   x̃ = x/L,  ỹ = y/L               (spatial)
#   û = u/U,   v̂ = v/U               (velocity, from ψ̂ = ψ/(UL))
#   p̂ = p/(ρU²)                      (pressure)
#   ν̃ = ν/(UL) = 1/Re                (viscosity)
#
# The dimensionless steady N-S equation solved is:
#   û ∂û/∂x̃ + v̂ ∂û/∂ỹ = -∂p̂/∂x̃ + ν̃ ∇̃²û + F̃_x
#
# To recover physical units, multiply everything by U²/L:
#   (U²/L)[û ∂û/∂x̃ + ...] = (U²/L)[-∂p̂/∂x̃ + ν̃ ∇̃²û + F̃_x]
#
# Each term:
#   convection: (U²/L)(û ∂û/∂x̃) = u ∂u/∂x               ✓ [m/s²]
#   pressure:   (U²/L)(∂p̂/∂x̃)  = (1/ρ)(∂p/∂x)          ✓ [m/s²]
#   diffusion:  (U²/L)(ν̃ ∇̃²û)  = ν ∇²u                  ✓ [m/s²]
#   body force: (U²/L)(F̃_x)     = F_phys/ρ               ✓ [m/s²]
#
# Therefore: F_phys = ρ * (U²/L) * F̃                      [N/m³]
#
# The ArcForce class outputs F̃(x̃,ỹ) with amplitude A_tilde.
# The PEAK of the Gaussian envelope is A_tilde (when dx/σ² ~ 1/σ at the
# inflection point), but `get_params()` returns the raw `amp = exp(log_amp)`.
#
# The actual force formula is:
#   F̃_x = amp * (dx/σ²) * exp(-r²/(2σ²))
# Peak of |F̃_x| occurs at dx ≈ σ, giving |F̃_x|_max ≈ amp * (1/σ) * exp(-0.5)
#
# Physical amplitude:
#   A_phys = aa * ρ * U² / L    [N/m³]   ← correct
#                                          (NOT aa * ρ * U² which is missing /L)

# ── Derived physical quantities ──
nu_phys   = nu_hat_final * U * L        # m²/s
mu_phys   = nu_phys * RHO               # Pa·s
Re        = 1.0 / nu_hat_final          # = UL/ν

u_max     = speed[mask].max()            # m/s
u_mean    = speed[mask].mean()           # m/s

p_range   = p_pred[mask].max() - p_pred[mask].min()   # Pa

# Arc force in physical units  ── CORRECTED ──
if not _arc_disabled:
    x_arc_phys   = xa * L                    # m
    sigma_phys   = sa * L                    # m
    A_phys       = aa * RHO * U**2 / L       # N/m³  (was missing /L before!)
    F_peak_phys  = RHO * U**2 / L * aa / (sa * np.exp(0.5))

# ── Vorticity (1/s) ──
dv_dx_fd = np.gradient(v_pred, PIXEL_SIZE_M, axis=1)
du_dy_fd = np.gradient(u_pred, PIXEL_SIZE_M, axis=0)
omega = dv_dx_fd - du_dy_fd
omega_max = np.abs(omega[mask]).max()

# ── Self-consistency checks ──
Re_check = U * L / nu_phys  # should equal Re = 1/nu_hat
u_max_over_U = u_max / U    # should be <= 1 (dimensionless max)

print("=" * 70)
print("PHYSICAL PROPERTY SUMMARY (SI Units)")
print("=" * 70)

print(f"\n--- Characteristic Scales ---")
print(f"  L (pool size)      = {L*1e3:.2f} mm  = {L:.4e} m")
print(f"  U (max velocity)   = {U:.4f} m/s")
print(f"  S_psi = UL         = {U*L:.4e} m^2/s")
print(f"  S_p   = rho*U^2    = {RHO*U**2:.2f} Pa")

print(f"\n--- Material Properties ---")
print(f"  nu_hat (dimensionless) = {nu_hat_final:.6f}")
print(f"  Kinematic viscosity  nu = {nu_phys:.4e} m^2/s")
print(f"  Dynamic viscosity    mu = {mu_phys:.4e} Pa.s"
      f"  = {mu_phys*1e3:.3f} mPa.s")
print(f"    Literature for molten steel (near melting ~1811K): 4-7 mPa.s")
print(f"    At TIG pool peak temps (2200-2800K): ~2-5 mPa.s")
if 2.0 <= mu_phys * 1e3 <= 10:
    print(f"    -> WITHIN reasonable range for high-T molten steel")
elif mu_phys * 1e3 < 2.0:
    print(f"    -> BELOW expected range (may indicate underfitting viscosity)")
else:
    print(f"    -> ABOVE expected range: {mu_phys*1e3:.1f} mPa.s")
    print(f"       This likely indicates the PINN learned an EFFECTIVE viscosity")
    print(f"       that absorbs unresolved physics (turbulent mixing, 3D effects,")
    print(f"       Marangoni stresses not in the 2D model, etc.)")

print(f"\n--- Dimensionless Numbers ---")
print(f"  Reynolds number  Re = {Re:.1f}")
print(f"    Self-check: UL/nu = {Re_check:.1f} (should match)")
print(f"    Literature for TIG welding: Re ~ 100-2000")
print(f"    With filler wire (enhanced flow): Re can be higher")
if Re < 2300:
    print(f"    -> Laminar regime (Re < 2300)")
elif Re < 4000:
    print(f"    -> Transitional regime")
else:
    print(f"    -> Turbulent regime (Re > 4000)")

print(f"\n--- Flow Characteristics ---")
print(f"  Max velocity   |u|_max  = {u_max:.4f} m/s"
      f"  ({u_max*1e3:.1f} mm/s)")
print(f"  Mean velocity  |u|_mean = {u_mean:.4f} m/s"
      f"  ({u_mean*1e3:.1f} mm/s)")
print(f"  u_max / U (should be <= 1) = {u_max_over_U:.4f}")
print(f"    Literature for autogenous TIG: 0.05-0.5 m/s")
print(f"    With filler wire on surface: 0.1-1.2 m/s")
print(f"    (Filler wire adds momentum & Marangoni gradient)")
print(f"  Max vorticity  |omega|  = {omega_max:.1f} 1/s")

print(f"\n--- Pressure Field ---")
print(f"  Pressure range  Delta_p = {p_range:.2f} Pa")
print(f"  p_min = {p_pred[mask].min():.2f} Pa")
print(f"  p_max = {p_pred[mask].max():.2f} Pa")
print(f"    Dynamic pressure scale rho*U^2 = {RHO*U**2:.2f} Pa")

print(f"\n--- Arc Force Parameters ---")
if not _arc_disabled:
    print(f"  Position   x_arc   = {x_arc_phys*1e3:.2f} mm "
          f"(tilde={xa:.4f})")
    print(f"  Width      sigma   = {sigma_phys*1e3:.2f} mm "
          f"(tilde={sa:.4f})")
    print(f"    Literature: sigma ~ 2-4 mm for TIG arc")
    print(f"  Amplitude  A       = {A_phys:.2e} N/m^3 "
          f"(tilde={aa:.4f})")
    print(f"    [A_phys = A_tilde * rho * U^2 / L]")
    print(f"  Peak force |F|_max = {F_peak_phys:.2e} N/m^3")
    print(f"    Literature: peak body force ~ 1e3 - 1e6 N/m^3 for TIG")
    print(f"    (depends strongly on current, arc shape, and filler interaction)")
else:
    print(f"  Arc force: DISABLED (pure N-S, no body force term)")

# ── Flag potential issues ──
print(f"\n--- Diagnostic Flags ---")
issues = []
if mu_phys * 1e3 > 20:
    issues.append(f"  [!] mu = {mu_phys*1e3:.1f} mPa.s >> 7 mPa.s: "
                  f"effective/turbulent viscosity likely")
if u_max_over_U > 1.05:
    issues.append(f"  [!] u_max/U = {u_max_over_U:.3f} > 1: "
                  f"PINN predicting velocities above normalization scale")
if u_max_over_U < 0.5:
    issues.append(f"  [!] u_max/U = {u_max_over_U:.3f} << 1: "
                  f"PINN significantly under-predicting peak velocity")
if not _arc_disabled and sigma_phys * 1e3 > 6:
    issues.append(f"  [!] sigma = {sigma_phys*1e3:.1f} mm > 6 mm: "
                  f"arc force very broad (pool is only {L*1e3:.1f} mm)")
if len(issues) == 0:
    print("  No flags.")
else:
    for iss in issues:
        print(iss)

print("=" * 70)



# %% [Cell 20]
# ============================================================================
# REPLACEMENT CELL FOR FIGURE 4: Physical Comparison Bar Chart
# ============================================================================
# Changes:
#   - Updated literature ranges to reflect high-temperature TIG + filler wire
#   - Added arc force comparison bar
# ============================================================================

fig4, axes4 = plt.subplots(1, 4, figsize=(20, 5))

# Dynamic viscosity comparison
labels_mu = ['PINN', 'Lit. near\nmelting', 'Lit. TIG\npeak T']
vals_mu   = [mu_phys * 1e3, 5.5, 3.5]
errs_mu   = [0, 1.5, 1.5]  # half-range as error bars
colors_mu = ['steelblue', 'lightcoral', 'lightsalmon']
axes4[0].bar(labels_mu, vals_mu, color=colors_mu, edgecolor='k', alpha=0.8,
             yerr=errs_mu, capsize=5)
axes4[0].set_ylabel('Dynamic viscosity [mPa.s]')
axes4[0].set_title('Dynamic Viscosity μ')
axes4[0].grid(True, alpha=0.3, axis='y')

# Reynolds number comparison
labels_re = ['PINN', 'TIG\n(autogenous)', 'TIG\n(w/ wire)']
vals_re   = [Re, 300, 800]
errs_re   = [0, 200, 400]
colors_re = ['steelblue', 'lightcoral', 'lightsalmon']
axes4[1].bar(labels_re, vals_re, color=colors_re, edgecolor='k', alpha=0.8,
             yerr=errs_re, capsize=5)
axes4[1].set_ylabel('Reynolds number')
axes4[1].set_title('Reynolds Number')
axes4[1].grid(True, alpha=0.3, axis='y')

# Max velocity comparison
labels_v = ['PINN', 'TIG\n(autogenous)', 'TIG+wire\n(CFD lit.)']
vals_v   = [u_max, 0.3, 0.7]
errs_v   = [0, 0.2, 0.5]
colors_v = ['steelblue', 'lightcoral', 'lightsalmon']
axes4[2].bar(labels_v, vals_v, color=colors_v, edgecolor='k', alpha=0.8,
             yerr=errs_v, capsize=5)
axes4[2].set_ylabel('Max velocity [m/s]')
axes4[2].set_title('Maximum Velocity')
axes4[2].grid(True, alpha=0.3, axis='y')

# Arc force comparison
if not _arc_disabled:
    labels_f = ['PINN\npeak |F|', 'TIG lit.\n(low)', 'TIG lit.\n(high)']
    vals_f   = [F_peak_phys, 1e4, 1e5]
    colors_f = ['steelblue', 'lightcoral', 'lightcoral']
    axes4[3].bar(labels_f, vals_f, color=colors_f, edgecolor='k', alpha=0.8)
    axes4[3].set_ylabel('Peak force [N/m³]')
    axes4[3].set_title('Arc Force Magnitude')
    axes4[3].set_yscale('log')
    axes4[3].grid(True, alpha=0.3, axis='y')
else:
    axes4[3].text(0.5, 0.5, 'Arc force\n(disabled)',
                  ha='center', va='center', transform=axes4[3].transAxes,
                  fontsize=11, color='gray')
    axes4[3].set_title('Arc Force Magnitude')
    axes4[3].set_axis_off()

plt.tight_layout()
fig4.savefig(os.path.join(results_dir, 'physical_comparison.png'), dpi=150)
plt.show()
plt.close(fig4)
print(f"Saved: {results_dir}/physical_comparison.png")


# %% [markdown Cell 21]
# ## 10. Save All Results to `PINN_results_0mm`

# %% [Cell 22]
# ── Model checkpoint ──
ckpt_path = os.path.join(results_dir, 'pinn_model.pt')
torch.save({
    'model_state':     model.state_dict(),
    'arc_force_state': arc_force.state_dict(),
    'log_nu_hat':      log_nu_hat.detach().cpu(),
    'config':          CONFIG,
    'L': L, 'U': U, 'RHO': RHO,
    'history':         history,
}, ckpt_path)
print(f"Saved checkpoint: {ckpt_path}")

# ── Physical fields as numpy arrays ──
np.savez_compressed(
    os.path.join(results_dir, 'fields_SI.npz'),
    u_ms=u_pred, v_ms=v_pred, p_Pa=p_pred,
    mask=mask, speed_ms=speed,
    vorticity_1s=omega,
)
print(f"Saved fields: {results_dir}/fields_SI.npz")

# ── Reconstruction errors vs RAFT reference (at observed pixels) ──
if obs_mask.any():
    u_err     = u_pred[obs_mask] - max_u_map[obs_mask]
    v_err     = v_pred[obs_mask] - max_v_map[obs_mask]
    spd_err   = speed[obs_mask]  - max_speed_map[obs_mask]
    recon_errors = {
        'reference':       'RAFT max-speed map at observed pixels',
        'n_obs_pixels':    int(obs_mask.sum()),
        'u_MAE_ms':        float(np.abs(u_err).mean()),
        'u_RMSE_ms':       float(np.sqrt((u_err**2).mean())),
        'v_MAE_ms':        float(np.abs(v_err).mean()),
        'v_RMSE_ms':       float(np.sqrt((v_err**2).mean())),
        'speed_MAE_ms':    float(np.abs(spd_err).mean()),
        'speed_RMSE_ms':   float(np.sqrt((spd_err**2).mean())),
    }
else:
    recon_errors = {'reference': 'no observed pixels', 'n_obs_pixels': 0}

print(f"Reconstruction errors (vs RAFT, {recon_errors['n_obs_pixels']} obs pixels):")
print(f"  u   MAE={recon_errors.get('u_MAE_ms', float('nan')):.4f}  "
      f"RMSE={recon_errors.get('u_RMSE_ms', float('nan')):.4f} m/s")
print(f"  v   MAE={recon_errors.get('v_MAE_ms', float('nan')):.4f}  "
      f"RMSE={recon_errors.get('v_RMSE_ms', float('nan')):.4f} m/s")
print(f"  spd MAE={recon_errors.get('speed_MAE_ms', float('nan')):.4f}  "
      f"RMSE={recon_errors.get('speed_RMSE_ms', float('nan')):.4f} m/s")

# ── Physical properties summary (JSON) ──
summary = {
    'characteristic_scales': {
        'L_m': float(L),
        'L_mm': float(L * 1e3),
        'U_ms': float(U),
        'S_psi_m2s': float(U * L),
        'S_p_Pa': float(RHO * U**2),
    },
    'material_properties': {
        'rho_kgm3': float(RHO),
        'nu_hat_dimless': float(nu_hat_final),
        'nu_phys_m2s': float(nu_phys),
        'mu_phys_Pas': float(mu_phys),
        'mu_phys_mPas': float(mu_phys * 1e3),
    },
    'dimensionless_numbers': {
        'Reynolds': float(Re),
        'flow_regime': 'laminar' if Re < 2300 else (
            'transitional' if Re < 4000 else 'turbulent'),
    },
    'flow_characteristics': {
        'u_max_ms': float(u_max),
        'u_mean_ms': float(u_mean),
        'omega_max_1s': float(omega_max),
        'p_range_Pa': float(p_range),
    },
    'arc_force': {
        'disabled': _arc_disabled,
        **({
            'x_arc_mm': float(x_arc_phys * 1e3),
            'sigma_mm': float(sigma_phys * 1e3),
            'A_Nm3': float(A_phys),
            'x_tilde': float(xa),
            'sigma_tilde': float(sa),
            'A_tilde': float(aa),
        } if not _arc_disabled else {}),
    },
    'final_losses': {
        'total':       float(history['total'][-1]),
        'data':        float(history['data'][-1]),
        'boundary':    float(history['boundary'][-1]),
        'pde':         float(history['pde'][-1]),
        'p_ref':       float(history['p_ref'][-1]),
        'symmetry':    float(history['symmetry'][-1]),
        'arc_prior':   float(history['arc_prior'][-1]),
    },
    'arc_force_prior': {
        'source': 'Image-estimated position and width; amplitude free (no prior)',
        'x_frac_measured': CONFIG['arc_x_frac_measured'],
        'x_frac_tolerance': CONFIG['arc_x_frac_tolerance'],
        'sigma_init_mm': CONFIG['arc_sigma_init_m'] * 1e3,
        'sigma_prior_log_std': CONFIG['arc_sigma_prior_log_std'],
        'amplitude_prior': 'none (free — 0mm case used as reference for other cases)',
    },
    'reconstruction_errors': recon_errors,
    'literature_comparison': {
        'mu_expected_mPas': '4-7 (molten steel)',
        'Re_expected': '100-500 (TIG welding)',
        'u_max_expected_ms': '0.1-0.5 (TIG welding)',
        'sigma_arc_expected_mm': '2-4 (TIG)',
        'A_arc_expected_Nm3': '1e4-1e5 (TIG)',
    },
    'unit_conversion': {
        'pixel_size_m': float(PIXEL_SIZE_M),
        'fps': int(FPS),
        'stride': int(STRIDE),
        'dt_s': float(DT),
        'vel_factor_ms_per_pxstride': float(VEL_FACTOR),
    },
}

json_path = os.path.join(results_dir, 'physical_properties.json')
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary: {json_path}")

# ── Training history ──
np.savez_compressed(
    os.path.join(results_dir, 'training_history.npz'),
    **{k: np.array(v) for k, v in history.items()}
)
print(f"Saved history: {results_dir}/training_history.npz")

print(f"\nAll results saved to: {results_dir}/")
print(f"  training_curves.png       - loss and parameter evolution")
print(f"  field_comparison.png      - RAFT vs PINN fields")
print(f"  symmetry_check.png        - symmetry verification")
print(f"  physical_comparison.png   - comparison with literature")
print(f"  pinn_model.pt             - model checkpoint")
print(f"  fields_SI.npz             - velocity/pressure fields (SI)")
print(f"  physical_properties.json  - all physical properties")
print(f"  training_history.npz      - loss curves data")
