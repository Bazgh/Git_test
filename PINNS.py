# -*- coding: utf-8 -*-
"""
PINNS.py — GPU-mem-safe:
- All big data (mesh, inlet/wall, sparse) lives on CPU (NumPy/CPU tensors).
- Each training step: move ONLY the current batch to GPU.
- PDE autograd & optimizer steps run on GPU.
- Data loss samples a tiny subset and slices on CPU first, then moves small slices.
- Geometry latent expanded per-batch (no N-wide latent on GPU).
- AMP (autocast + GradScaler) used to cut VRAM.
"""
#for git test
import os, time, subprocess, threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

import matplotlib
matplotlib.use("Agg")  # headless plotting for Slurm
import matplotlib.pyplot as plt

import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree

# ---- heartbeat (optional; shows GPU mem/util in .out every 30s) ----
def gpu_heartbeat(interval=30):
    while True:
        try:
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() // (1024*1024)
                resv  = torch.cuda.memory_reserved() // (1024*1024)
                print(f"[GPU] torch mem alloc/res (MB): {alloc}/{resv}", flush=True)
            out = subprocess.check_output(
                ["nvidia-smi","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"]
            ).decode().strip()
            print(f"[GPU] nvidia-smi mem,util: {out}", flush=True)
        except Exception as e:
            print(f"[GPU] heartbeat error: {e}", flush=True)
        time.sleep(interval)

threading.Thread(target=gpu_heartbeat, daemon=True).start()

# ----------------------------
# Paths (relative to this file)
# ----------------------------
ROOT        = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH   = os.path.join(ROOT, "geom_pointnet_vae_k8_N400.pt")
MESH_FILE   = os.path.join(ROOT, "internal.vtu")
INLET_FILE  = os.path.join(ROOT, "inlet_velocity.vtk")
WALL_FILE   = os.path.join(ROOT, "wall.vtk")
SPARSE_FILE = os.path.join(ROOT, "sample_2.vtk")
RESULT_DIR  = os.path.join(ROOT, "Results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ----------------------------
# Device & globals
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # perf hint

# scaling/physics
X_scale = 2.0
Y_scale = 1.0
Z_scale=1.0
U_scale = 1.0
U_BC_in = 0.5
Diff    = 1e-3
rho     = 1.0

# training knobs (start conservative; scale up later)
Flag_BC_exact  = False
Lambda_BC      = 20.0
Lambda_div     = 1.0
batchsize      = 2         # tiny to avoid OOM
colloc_per_ep  = 500       # tiny to avoid OOM
epochs         = 50
use_amp        = True

# domain box (for optional exact-BC shaping)
xStart, xEnd = 0.0, 1.0
yStart, yEnd = 0.0, 1.0
zStart, zEnd = 0.0, 1.0

# LR schedule
learning_rate  = 5e-4
step_epoch     = 1200
decay_rate     = 0.1
#just for testing GIT
# ----------------------------
# I/O helpers
# ----------------------------
def read_polydata_points(path: str) -> np.ndarray:
    r = vtk.vtkPolyDataReader()
    r.SetFileName(path); r.Update()
    poly = r.GetOutput()
    n = poly.GetNumberOfPoints()
    assert n > 0, f"No points in {path}"
    pts = np.array([poly.GetPoint(i) for i in range(n)], dtype=np.float32)
    return pts

def read_unstructured_points(path: str) -> np.ndarray:
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(path); r.Update()
    grid = r.GetOutput()
    n = grid.GetNumberOfPoints()
    assert n > 0, f"No points in {path}"
    xyz = vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float32)
    return xyz

# ----------------------------
# Load data (CPU)
# ----------------------------
print("Loading mesh:", MESH_FILE, flush=True)
mesh_xyz = read_unstructured_points(MESH_FILE)
x = mesh_xyz[:, 0:1]; y = mesh_xyz[:, 1:2]; z = mesh_xyz[:, 2:3]
N = x.shape[0]
print("n_points of the mesh:", N, flush=True)

print("Loading inlet:", INLET_FILE, flush=True)
inlet_xyz = read_polydata_points(INLET_FILE)
xb_in = inlet_xyz[:, 0:1]; yb_in = inlet_xyz[:, 1:2]; zb_in = inlet_xyz[:, 2:3]
print("n_points at inlet:", xb_in.shape[0], flush=True)

print("Loading wall:", WALL_FILE, flush=True)
wall_xyz = read_polydata_points(WALL_FILE)
xb = wall_xyz[:, 0:1]; yb = wall_xyz[:, 1:2]; zb = wall_xyz[:, 2:3]
print("n_points at wall:", xb.shape[0], flush=True)

print("Loading sparse:", SPARSE_FILE, flush=True)
r = vtk.vtkPolyDataReader()
r.SetFileName(SPARSE_FILE); r.Update()
pd = r.GetOutput()
sparse_xyz = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float32)
flow_vtk = pd.GetPointData().GetArray("flow")
assert flow_vtk is not None and flow_vtk.GetNumberOfComponents() == 3, "Point-data 'flow' (3 comps) not found."
flow = vtk_to_numpy(flow_vtk).astype(np.float32)
u_sparse = flow[:, 0:1]; v_sparse = flow[:, 1:2]; w_sparse = flow[:, 2:3]
M_sparse = u_sparse.shape[0]
#for data loss
idx_sub_sparse= np.random.choice(M_sparse, size=10, replace=False)  # shape (10,)
xyz_sub_np = sparse_xyz[idx_sub_sparse]       # shape (10, 3), NumPy
xyz_sub = torch.from_numpy(xyz_sub_np).float().to(device)  # shape (10,3)
mesh_xyz_cpu = torch.from_numpy(mesh_xyz).float()  # (N,3) on CPU
# xyz_sub: (B,3) on GPU (B=10 here)
B = xyz_sub.size(0)

# holders for the best (min) squared distance and index per sparse point
dmin = torch.full((B,), float('inf'), device=device)
imin = torch.empty((B,), dtype=torch.long, device=device)

# tune chunk size to your GPU; 50k is fine on 24 GB, use 10k–25k if tight
chunk = 50000
N = mesh_xyz_cpu.size(0)

for s in range(0, N, chunk):
    e = min(s + chunk, N)
    mesh_chunk = mesh_xyz_cpu[s:e].to(device, non_blocking=True)   # (C,3) on GPU

    # broadcasting: (B,1,3) - (1,C,3) -> (B,C,3) -> squared distances (B,C)
    diff = xyz_sub[:, None, :] - mesh_chunk[None, :, :]
    d2   = (diff * diff).sum(dim=-1)

    # per sparse point: best within this chunk
    d2_min, idx_local = d2.min(dim=1)                 # both (B,)

    # compare to global best so far
    better = d2_min < dmin
    dmin[better] = d2_min[better]
    imin[better] = idx_local[better] + s              # offset back to global mesh index

    # free big temporaries early
    del mesh_chunk, diff, d2, d2_min, idx_local, better

# Slice from sparse arrays (NumPy -> torch)
u_10_sparse = torch.from_numpy(u_sparse[idx_sub_sparse]).float().to(device)
v_10_sparse = torch.from_numpy(v_sparse[idx_sub_sparse]).float().to(device)
w_10_sparse = torch.from_numpy(w_sparse[idx_sub_sparse]).float().to(device)
# inlet BC (u=v=0, w profile)
W_peak, alpha = 0.5, 1.0
xc, yc = float(xb_in.mean()), float(yb_in.mean())
a = float(0.5 * (xb_in.max() - xb_in.min()) + 1e-9)
b = float(0.5 * (yb_in.max() - yb_in.min()) + 1e-9)
rho2 = ((xb_in - xc)/a)**2 + ((yb_in - yc)/b)**2
w_in_BC = (W_peak / (1.0 + alpha * rho2)).astype(np.float32)
u_in_BC = np.zeros_like(w_in_BC, dtype=np.float32)
v_in_BC = np.zeros_like(w_in_BC, dtype=np.float32)

# wall no-slip
u_wall_BC = np.zeros_like(xb, dtype=np.float32)
v_wall_BC = np.zeros_like(yb, dtype=np.float32)
w_wall_BC = np.zeros_like(zb, dtype=np.float32)

# nearest neighbors (sparse -> mesh)

# ----------------------------
# Geometry encoder (latent z)
# ----------------------------
from Geometry_encoder_training import BoundaryEncoder

def load_encoder_and_latent(wall_xyz_: np.ndarray, k: int = 8) -> torch.Tensor:
    enc = BoundaryEncoder(k=k).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        enc_state = {k.replace("enc.", "", 1): v for k, v in ckpt.items() if k.startswith("enc.")}
        if enc_state:
            enc.load_state_dict(enc_state, strict=True)
        else:
            enc.load_state_dict(ckpt, strict=False)
    enc.eval()
    with torch.no_grad():
        coords_t = torch.from_numpy(wall_xyz_).unsqueeze(0).to(device)  # [1,N,3]
        mu, lv = enc(coords_t)  # [1,k],[1,k]
        z = mu + torch.exp(0.5*lv) * torch.randn_like(mu)
    return z.squeeze(0)  # [k]

geom_latent = load_encoder_and_latent(wall_xyz, k=8).to(device)  # [K]
K = geom_latent.numel()
input_n = 3 + K


# Nets
# ----------------------------
class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

class MySquared(nn.Module):
    def forward(self, x): return torch.square(x)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth, last_act=None):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), Swish()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        if last_act is not None: layers += [last_act]
        self.main = nn.Sequential(*layers)
    def forward(self, x): return self.main(x)

# keep models modest to save VRAM
h_nD = 32     # BC nets width
h_n  = 96     # main nets width
depth_main = 8

Net1_bc_u = lambda: MLP(input_n, h_nD, 1, 5, last_act=nn.ReLU())
Net1_bc_v = lambda: MLP(input_n, h_nD, 1, 4, last_act=nn.ReLU())
Net1_bc_w = lambda: MLP(input_n, h_nD, 1, 4, last_act=nn.ReLU())

class Net2_u(nn.Module):
    def __init__(self): super().__init__(); self.m = MLP(input_n, h_n, 1, depth_main)
    def forward(self, xin):
        out = self.m(xin)
        if Flag_BC_exact:
            x, y, z = xin[:, :1], xin[:, 1:2], xin[:, 2:3]
            out = out * (x - xStart) * (y - yStart) * (y - yEnd) + 0.0 + (y - yStart)*(y - yEnd)*(z - zStart)*(z - zEnd)
        return out

class Net2_v(nn.Module):
    def __init__(self): super().__init__(); self.m = MLP(input_n, h_n, 1, depth_main)
    def forward(self, xin):
        out = self.m(xin)
        if Flag_BC_exact:
            x, y, z = xin[:, :1], xin[:, 1:2], xin[:, 2:3]
            out = out * (x - xStart)*(x - xEnd)*(y - yStart)*(y - yEnd)*(z - zStart)*(z - zEnd) + (-0.9 * x + 1.0)
        return out

class Net2_w(nn.Module):
    def __init__(self): super().__init__(); self.m = MLP(input_n, h_n, 1, depth_main)
    def forward(self, xin):
        out = self.m(xin)
        if Flag_BC_exact:
            x, y, z = xin[:, :1], xin[:, 1:2], xin[:, 2:3]
            out = out * (x - xStart)*(x - xEnd)*(y - yStart)*(y - yEnd)*(z - zStart)*(z - zEnd) + (-0.9 * x + 1.0)
        return out

class Net2_p(nn.Module):
    def __init__(self): super().__init__(); self.m = MLP(input_n, h_n, 1, depth_main)
    def forward(self, xin):
        out = self.m(xin)
        if Flag_BC_exact:
            x, y = xin[:, :1], xin[:, 1:2]
            out = out * (x - xStart)*(x - xEnd)*(y - yStart)*(y - yEnd) + (-0.9 * x + 1.0)
        return out

# ----------------------------
# Losses (GPU)
# ----------------------------
def criterion_pde(net_u, net_v, net_w, net_p, x, y, z, geom_k):
    # x,y,z,geom_k are small batch tensors on GPU
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
    net_in = torch.cat((x, y, z, geom_k), 1)

    u = net_u(net_in).view(-1, 1)
    v = net_v(net_in).view(-1, 1)
    w = net_w(net_in).view(-1, 1)
    P = net_p(net_in).view(-1, 1)

    ones_x = torch.ones_like(x); ones_y = torch.ones_like(y); ones_z = torch.ones_like(z)

    u_x  = torch.autograd.grad(u, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    u_y  = torch.autograd.grad(u, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    u_z  = torch.autograd.grad(u, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]

    v_x  = torch.autograd.grad(v, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    v_y  = torch.autograd.grad(v, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    v_z  = torch.autograd.grad(v, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]

    w_x  = torch.autograd.grad(w, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    w_y  = torch.autograd.grad(w, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    w_z  = torch.autograd.grad(w, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]  # fixed

    P_x = torch.autograd.grad(P, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    P_y = torch.autograd.grad(P, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    P_z = torch.autograd.grad(P, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]

    XX_scale = U_scale * (X_scale ** 2)
    YY_scale = U_scale * (Y_scale ** 2)
    UU_scale = U_scale ** 2

    loss_x = u * u_x / X_scale + v * u_y / Y_scale + w * u_z / Y_scale \
         - Diff * (u_xx / XX_scale + u_yy / YY_scale + u_zz / YY_scale) \
         + (1 / rho) * (P_x / (X_scale * UU_scale))
    loss_y = u * v_x / X_scale + v * v_y / Y_scale + w * v_z / Y_scale \
         - Diff * (v_xx / XX_scale + v_yy / YY_scale + v_zz / YY_scale) \
         + (1 / rho) * (P_y / (Y_scale * UU_scale))
    loss_z = u * w_x / X_scale + v * w_y / Y_scale + w * w_z / Y_scale \
         - Diff * (w_xx / XX_scale + w_yy / YY_scale + w_zz / YY_scale) \
         + (1 / rho) * (P_z / (Y_scale * UU_scale))
    # Continuity equation with scaling
    loss_c = (u_x / X_scale) + (v_y / Y_scale) + (w_z / Z_scale)


    mse = nn.MSELoss()
    return (mse(loss_x, torch.zeros_like(loss_x)) +
            mse(loss_y, torch.zeros_like(loss_y)) +
            mse(loss_z, torch.zeros_like(loss_z)) +
            mse(loss_c, torch.zeros_like(loss_c)) * Lambda_div)

def loss_bc(net_u, net_v, net_w,
            xb, yb, zb, ub, vb, wb,
            xb_in, yb_in, zb_in, ub_in, vb_in, wb_in,
            geom_wall, geom_in):
    nin_wall = torch.cat((xb, yb, zb, geom_wall), 1)
    uw = net_u(nin_wall); vw = net_v(nin_wall); ww = net_w(nin_wall)
    nin_in   = torch.cat((xb_in, yb_in, zb_in, geom_in), 1)
    ui = net_u(nin_in);   vi = net_v(nin_in);   wi = net_w(nin_in)
    mse = nn.MSELoss()
    return (mse(uw, ub) + mse(vw, vb) + mse(ww, wb) +
            mse(ui, ub_in) + mse(vi, vb_in) + mse(wi, wb_in))

def loss_data_term(net_u, net_v, net_w,
                   x_nn, y_nn, z_nn, geom_latent_k_nn,
                   u_10_sparse, v_10_sparse, w_10_sparse):
    nin = torch.cat((x_nn, y_nn, z_nn, geom_latent_k_nn), 1)
    u_pred = net_u(nin).view(-1,1)
    v_pred = net_v(nin).view(-1,1)
    w_pred = net_w(nin).view(-1,1)
    mse = nn.MSELoss()
    return mse(u_pred, u_10_sparse) + mse(v_pred, v_10_sparse) + mse(w_pred, w_10_sparse)

# ----------------------------
# Training (CPU dataset; GPU compute)
# ----------------------------
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler(enabled=use_amp)

def train():
    # CPU tensors for dataset
    x_cpu  = torch.from_numpy(x).float()
    y_cpu  = torch.from_numpy(y).float()
    z_cpu  = torch.from_numpy(z).float()
    dataset = TensorDataset(x_cpu, y_cpu, z_cpu)
    #sampler = RandomSampler(dataset, replacement=True, num_samples=colloc_per_ep)
    loader  = DataLoader(dataset, batch_size=batchsize,
                         num_workers=0, pin_memory=True, drop_last=True)

    # small BC/sparse tensors — prepare GPU copies ONCE
    xb_t = torch.from_numpy(xb).float().to(device)
    yb_t = torch.from_numpy(yb).float().to(device)
    zb_t = torch.from_numpy(zb).float().to(device)
    ub_t = torch.from_numpy(u_wall_BC).float().to(device)
    vb_t = torch.from_numpy(v_wall_BC).float().to(device)
    wb_t = torch.from_numpy(w_wall_BC).float().to(device)

    xbi_t = torch.from_numpy(xb_in).float().to(device)
    ybi_t = torch.from_numpy(yb_in).float().to(device)
    zbi_t = torch.from_numpy(zb_in).float().to(device)
    ubi_t = torch.from_numpy(u_in_BC).float().to(device)
    vbi_t = torch.from_numpy(v_in_BC).float().to(device)
    wbi_t = torch.from_numpy(w_in_BC).float().to(device)

    #us_t = torch.from_numpy(u_sparse).float().to(device)
    #vs_t = torch.from_numpy(v_sparse).float().to(device)
    #ws_t = torch.from_numpy(w_sparse).float().to(device)
    
    #nn_index_t = imin.long().to(device)
    
    # nets & optims
    net_u, net_v, net_w, net_p = Net2_u().to(device), Net2_v().to(device), Net2_w().to(device), Net2_p().to(device)
    def init_kaiming(m):
        if isinstance(m, nn.Linear): nn.init.kaiming_normal_(m.weight)
    for net in (net_u, net_v, net_w, net_p): net.apply(init_kaiming)

    opt_u = optim.Adam(net_u.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-15)
    opt_v = optim.Adam(net_v.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-15)
    opt_w = optim.Adam(net_w.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-15)
    opt_p = optim.Adam(net_p.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-15)

    sch_u = torch.optim.lr_scheduler.StepLR(opt_u, step_size=step_epoch, gamma=decay_rate)
    sch_v = torch.optim.lr_scheduler.StepLR(opt_v, step_size=step_epoch, gamma=decay_rate)
    sch_w = torch.optim.lr_scheduler.StepLR(opt_w, step_size=step_epoch, gamma=decay_rate)
    sch_p = torch.optim.lr_scheduler.StepLR(opt_p, step_size=step_epoch, gamma=decay_rate)

    t0 = time.time()
    for ep in range(epochs):
        eq_tot = bc_tot = dat_tot = 0.0; nb = 0

        for (x_in_cpu, y_in_cpu, z_in_cpu) in loader:
            # move ONLY the batch to GPU
            x_in  = x_in_cpu.to(device, non_blocking=True)
            y_in  = y_in_cpu.to(device, non_blocking=True)
            z_in  = z_in_cpu.to(device, non_blocking=True)
            geomN = geom_latent.view(1,-1).to(device).expand(x_in.shape[0], -1)

            opt_u.zero_grad(); opt_v.zero_grad(); opt_w.zero_grad(); opt_p.zero_grad()

            with autocast(enabled=use_amp):
                # PDE on collocation batch
                leq = criterion_pde(net_u, net_v, net_w, net_p, x_in.float(), y_in.float(), z_in.float(), geomN.float())

                # BC on small sets (GPU)
                gW = geom_latent.view(1,-1).to(device).expand(xb_t.size(0), -1)
                gI = geom_latent.view(1,-1).to(device).expand(xbi_t.size(0), -1)
                lbc = loss_bc(net_u, net_v, net_w,
                              xb_t, yb_t, zb_t, ub_t, vb_t, wb_t,
                              xbi_t, ybi_t, zbi_t, ubi_t, vbi_t, wbi_t,
                              gW, gI)
                # result: imin is (B,) long tensor of mesh indices on GPU print("nearest mesh indices for the 10 sparse points:", imin.tolist())
                mesh_nn_idx_cpu = imin.detach().cpu().numpy()      # (B,)
                x_nn = torch.from_numpy(mesh_xyz[mesh_nn_idx_cpu, 0:1]).float().to(device)
                y_nn = torch.from_numpy(mesh_xyz[mesh_nn_idx_cpu, 1:2]).float().to(device)
                z_nn = torch.from_numpy(mesh_xyz[mesh_nn_idx_cpu, 2:3]).float().to(device)
                k = geom_latent.size(-1)  # latent dim# build latent batch for the selected points
                if geom_latent.dim() == 1:
                    # (k,) -> (1,k) -> (B,k)
                    geom_latent_k_nn = geom_latent.view(1, -1).to(device).expand(imin.size(0), -1)
                elif geom_latent.dim() == 2 and geom_latent.size(0) == 1:
                    # (1,k) -> (B,k)
                    geom_latent_k_nn = geom_latent.to(device).expand(imin.size(0), -1)
                else:
                    # if you ever have per-node latents (N,k), gather by imin
                    geom_latent_k_nn = geom_latent.index_select(0, imin.long())

                # single, clean call (no duplicates, no stray characters)
                ldat = loss_data_term(
                    net_u, net_v, net_w,
                    x_nn, y_nn, z_nn, geom_latent_k_nn,
                    u_10_sparse, v_10_sparse, w_10_sparse
                )
        loss = leq + Lambda_BC * lbc + ldat
        scaler.scale(loss).backward()
        scaler.step(opt_u); scaler.step(opt_v); scaler.step(opt_w); scaler.step(opt_p)

        eq_tot += leq.item(); bc_tot += lbc.item(); dat_tot += ldat.item(); nb += 1

        sch_u.step(); sch_v.step(); sch_w.step(); sch_p.step()
        print(f"Epoch {ep:04d} | Loss eqn {eq_tot/nb:.3e}  Loss BC {bc_tot/nb:.3e}  "
              f"Loss data {dat_tot/nb:.3e}  lr {opt_u.param_groups[0]['lr']:.2e}", flush=True)

    print("Elapsed (s):", time.time() - t0, flush=True)

    # save weights
    torch.save(net_p.state_dict(), os.path.join(RESULT_DIR, "sten_p.pt"))
    torch.save(net_u.state_dict(), os.path.join(RESULT_DIR, "sten_u.pt"))
    torch.save(net_v.state_dict(), os.path.join(RESULT_DIR, "sten_v.pt"))
    torch.save(net_w.state_dict(), os.path.join(RESULT_DIR, "sten_w.pt"))
    print("Saved models in", RESULT_DIR, flush=True)

    # evaluate in chunks (avoid OOM)
    with torch.no_grad():
        net_u.eval(); net_v.eval(); net_w.eval()
        B = 50000  # chunk size
        u_pred = np.empty((N,1), dtype=np.float32)
        v_pred = np.empty((N,1), dtype=np.float32)
        w_pred = np.empty((N,1), dtype=np.float32)
        g1 = geom_latent.view(1,-1).to(device)
        for i in range(0, N, B):
            j = min(i+B, N)
            xi = torch.from_numpy(x[i:j]).float().to(device)
            yi = torch.from_numpy(y[i:j]).float().to(device)
            zi = torch.from_numpy(z[i:j]).float().to(device)
            gi = g1.expand(j-i, -1)
            nin = torch.cat((xi, yi, zi, gi), 1)
            u_pred[i:j] = net_u(nin).float().cpu().numpy()
            v_pred[i:j] = net_v(nin).float().cpu().numpy()
            w_pred[i:j] = net_w(nin).float().cpu().numpy()

    # save quick PNGs
    def save_scatter(vals, name):
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x[:,0], y[:,0], z[:,0], s=1, c=vals[:,0])
        fig.colorbar(sc); ax.set_title(name); fig.tight_layout()
        fig.savefig(os.path.join(RESULT_DIR, f"{name}.png"), dpi=150); plt.close(fig)

    save_scatter(u_pred, "u_pred")
    save_scatter(v_pred, "v_pred")
    save_scatter(w_pred, "w_pred")
    print("Saved figures in", RESULT_DIR, flush=True)
    
    # --- Write predictions to VTU for ParaView ---
    from vtk.util.numpy_support import numpy_to_vtk

    # Re-read the original unstructured grid so we keep the same connectivity
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(MESH_FILE)
    r.Update()
    grid = r.GetOutput()
    assert grid.GetNumberOfPoints() == N, "Point count mismatch between predictions and mesh."

    # Stack velocity to 3-component vector (float32)
    vel = np.hstack([u_pred, v_pred, w_pred]).astype(np.float32)  # shape (N,3)
    vel_vtk = numpy_to_vtk(vel, deep=True)
    vel_vtk.SetNumberOfComponents(3)
    vel_vtk.SetName("flow")

    # Optionally also save scalar components
    u_vtk = numpy_to_vtk(u_pred.astype(np.float32), deep=True); u_vtk.SetName("u")
    v_vtk = numpy_to_vtk(v_pred.astype(np.float32), deep=True); v_vtk.SetName("v")
    w_vtk = numpy_to_vtk(w_pred.astype(np.float32), deep=True); w_vtk.SetName("w")

    pd = grid.GetPointData()
    pd.AddArray(vel_vtk)
    pd.SetActiveVectors("flow")
    pd.AddArray(u_vtk); pd.AddArray(v_vtk); pd.AddArray(w_vtk)

    # Write VTU
    out_vtu = os.path.join(RESULT_DIR, "pinns_result.vtu")
    wtr = vtk.vtkXMLUnstructuredGridWriter()
    wtr.SetFileName(out_vtu)
    wtr.SetInputData(grid)
    wtr.Write()

    print("Saved VTU:", out_vtu, flush=True)
    
# ----------------------------
# Main
# ---------------------------- 
if __name__ == "__main__":
    # tip: in your sbatch add:
    #   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    train()
