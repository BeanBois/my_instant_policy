import torch
from torch import Tensor
from typing import List, Tuple
import math

# standalone functions 
def next_proceeding_strict(curr_hyp, demo_hyp, attn_same, c, delta: int = 1, topk: int = 1):
    """
    curr_hyp : [B, d]
    demo_hyp : [B, N, L, d]
    attn_same: [B, N, L]  (same-agent Euclid attention already reduced over agents)
    c        : Hyperbolic curvature (float)
    delta    : step forward size (1 by default)
    Enforces candidate indices be (n, l+delta). If l+delta >= L, they are masked out.

    Returns:
      cand:    [B, topk, d]     candidate hyperbolic points
      idx_n:   [B, topk]        demo indices
      idx_l:   [B, topk]        time indices (already +delta)
      weights: [B, topk]        combined score weights (for inspection)
    """
    B, N, L, d = demo_hyp.shape
    assert curr_hyp.shape == (B, d), f"curr_hyp must be [B,d], got {tuple(curr_hyp.shape)}"
    assert attn_same.shape == (B, N, L), f"attn_same must be [B,N,L], got {tuple(attn_same.shape)}"
    device = demo_hyp.device

    # valid time window where l+delta < L
    valid_L = L - delta
    if valid_L <= 0:
        raise ValueError(f"delta ({delta}) must be < sequence length L ({L})")

    # restrict attention and normalize over time per (B,N)
    attn = attn_same[:, :, :valid_L]                           # [B,N,valid_L]
    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

    # candidates at (n, l+delta)
    y = demo_hyp[:, :, delta:, :]                               # [B,N,valid_L,d]

    # compare in tangent at 0
    x_tan = logmap0(curr_hyp, c)                                # [B,d]
    y_tan = logmap0(y, c)                                       # [B,N,valid_L,d]

    # alignment score: negative squared Euclidean distance
    diff = x_tan[:, None, None, :] - y_tan                      # [B,N,valid_L,d]
    align = - (diff * diff).sum(dim=-1)                         # [B,N,valid_L]

    # combine with attention
    score = align * attn                                        # [B,N,valid_L]

    # pick top-k across (N, valid_L)
    score_flat = score.view(B, -1)                              # [B, N*valid_L]
    k = min(topk, score_flat.shape[1])
    topv, topi = torch.topk(score_flat, k=k, dim=1)             # [B,k]

    idx_n = topi // valid_L                                     # [B,k]
    idx_l = topi %  valid_L                                     # [B,k]

    # gather candidates from y (still in hyperbolic space)
    b_idx = torch.arange(B, device=device)[:, None]
    cand  = y[b_idx, idx_n, idx_l, :]                           # [B,k,d]

    # restore original timeline indices
    idx_l = idx_l + delta

    return cand, idx_n, idx_l, topv


def time_posterior(attn_same, reduce_over_agents: bool = True):
    """
    attn_same: [B, A, N, L]
    returns:
      p_t: [B, N, L] posterior over time per demo (agent-averaged if reduce_over_agents)
      t_soft: [B, N] expected timestep (float)
      t_star: [B, N] argmax timestep (long)
    """
    if reduce_over_agents:
        p = attn_same.mean(dim=1)  # [B,N,L]
    else:
        p = attn_same              # [B,A,N,L] (use a later)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
    idx = torch.arange(p.size(-1), device=p.device).float()
    t_soft = (p * idx).sum(dim=-1)                 # [B,N]
    t_star = torch.argmax(p, dim=-1)               # [B,N]
    return p, t_soft, t_star

# graph aux
def fourier_embed_2d(delta: Tensor, num_freqs: int = 10) -> Tensor:
    """Δ=(dx,dy) -> [sin/cos at powers-of-2 frequencies] with shape [E, 4*num_freqs]."""
    freqs = (2.0 ** torch.arange(num_freqs, device=delta.device, dtype=delta.dtype)) * torch.pi
    ang = delta.unsqueeze(-1) * freqs  # [E,2,F]
    sin_x, cos_x = torch.sin(ang[:, 0, :]), torch.cos(ang[:, 0, :])
    sin_y, cos_y = torch.sin(ang[:, 1, :]), torch.cos(ang[:, 1, :])
    return torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)  # [E, 4F]

# point care aux
def _sqrt_c(c):
    return c ** 0.5

def expmap0(v: torch.Tensor, c: float, eps: float = 1e-6) -> torch.Tensor:
    # exp_0(v) = tanh( sqrt(c) ||v|| / 2 ) * v / (sqrt(c) ||v||)
    sqc = _sqrt_c(c)
    v_norm = torch.clamp(v.norm(dim=-1, keepdim=True), min=eps)
    coef = torch.tanh(sqc * v_norm / 2.0) / (sqc * v_norm)
    return coef * v

def logmap0(x: torch.Tensor, c: float, eps: float = 1e-6) -> torch.Tensor:
    # log_0(x) = (2 / sqrt(c)) * artanh( sqrt(c) ||x|| ) * x / ||x||
    sqc = _sqrt_c(c)
    x_norm = torch.clamp(x.norm(dim=-1, keepdim=True), min=eps)
    arg = torch.clamp(sqc * x_norm, max=1 - 1e-6)  # stay inside ball
    coef = (2.0 / sqc) * torch.atanh(arg) / x_norm
    return coef * x

def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float, eps: float = 1e-6) -> torch.Tensor:
    # Not strictly needed for mean below, but handy if you extend.
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c**2 * x2 * y2
    return num / torch.clamp(den, min=eps)

def mobius_scalar_mul(t, x, c, eps=1e-6):
    nx = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    rn = torch.tanh(t * torch.atanh(_sqrt_c(c)*nx)) * x / ( _sqrt_c(c) * nx )
    return rn

def geodesic_segment(x, y, c, T: int):
    """
    x, y: [B, d] points in the Poincaré ball
    returns: [B, T+1, d] geodesic samples from t=0..1
    """
    # Δ = (-x) ⊕ y
    delta = mobius_add(-x, y, c)                                   # [B, d]

    # time samples t ∈ [0,1]
    ts = torch.linspace(0, 1, T + 1, device=x.device, dtype=x.dtype).view(1, T + 1, 1)  # [1, T+1, 1]

    # γ(t) = x ⊕ (t ⊗ Δ)
    path = mobius_add(x.unsqueeze(1),                               # [B, 1, d]
                      mobius_scalar_mul(ts, delta.unsqueeze(1), c), # [B, T+1, d]
                      c)                                            # [B, T+1, d]
    return path

    return path  # geodesic points from x to y

def poincare_weighted_mean(x: torch.Tensor, w: torch.Tensor, c: float, eps: float = 1e-6) -> torch.Tensor:
    """
    x: [..., S, d_h], w: [..., S] with sum w = 1 along S.
    Compute Möbius (Fréchet) barycenter via log_0 / exp_0.
    """
    # tangent-space mean at 0
    v = logmap0(x, c)                               # [..., S, d_h]
    m = torch.sum(w.unsqueeze(-1) * v, dim=-2)      # [..., d_h]
    y = expmap0(m, c)                               # [..., d_h]
    # clamp to inside ball for stability
    max_rad = (1.0 / _sqrt_c(c)) - 1e-5
    norm = torch.norm(y, dim=-1, keepdim=True) + eps
    scale = torch.clamp(max_rad / norm, max=1.0)
    return y * scale

def poincare_dist(x, y, c, eps=1e-6):
    # x,y: [...,d]
    x2 = (x*x).sum(-1, keepdim=True)
    y2 = (y*y).sum(-1, keepdim=True)
    num = 2 * ((x - y)**2).sum(-1, keepdim=True) * c
    den = torch.clamp((1 - c*x2)*(1 - c*y2), min=eps)
    z = 1 + num/den
    d = torch.log(z + torch.sqrt(torch.clamp(z*z - 1, min=0.0))) / _sqrt_c(c)
    return d.squeeze(-1)

def proj_ball(x, c, eps=1e-5):
    # project to open ball radius 1/√c
    max_norm = (1.0 / (c**0.5)) - eps
    n = x.norm(dim=-1, keepdim=True)
    scale = (max_norm / n).clamp(max=1.0)
    return x * scale


# Geometric Encoder Aux 
def furthest_point_sampling_2d(P: torch.Tensor, M: int) -> torch.Tensor:
    """
    FPS over 2D points.
    P: [N, 2] float tensor
    returns indices of chosen centroids: [M]
    """
    device = P.device
    N = P.shape[0]
    M = min(M, N)
    idxs = torch.zeros(M, dtype=torch.long, device=device)

    # Pick a random start
    idxs[0] = torch.randint(0, N, (1,), device=device)
    dist = torch.full((N,), float("inf"), device=device)

    last = P[idxs[0]]
    for i in range(1, M):
        d = torch.sum((P - last) ** 2, dim=-1)
        dist = torch.minimum(dist, d)
        idxs[i] = torch.argmax(dist)
        last = P[idxs[i]]
    return idxs[:M]

def knn_indices(P: torch.Tensor, q: torch.Tensor, k: int) -> torch.Tensor:
    """
    Return indices of k nearest neighbors of q among P
    P: [N,2], q: [2], returns [k] (with replacement if N<k)
    """
    N = P.shape[0]
    k = min(k, N)
    d2 = torch.sum((P - q) ** 2, dim=-1)
    return torch.topk(d2, k, largest=False).indices

def fourier_feats_2d(delta: torch.Tensor, L: int) -> torch.Tensor:
    """
    NeRF-like Fourier features over 2D offsets.
    delta: [K,2]; returns [K, 4*L] = [sin(wx),cos(wx), sin(wy),cos(wy)]_l
    """
    outs = []
    for l in range(L):
        w = (2.0 ** l) * math.pi
        x = delta[:, 0:1] * w
        y = delta[:, 1:2] * w
        outs.extend([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)])
    return torch.cat(outs, dim=-1) if outs else torch.zeros(delta.shape[0], 0, device=delta.device)


# action aux
def se2_from_kp(P, Q, eps=1e-6):
    """
    P, Q: (B, K, 2) keypoints at t and t+1 in Euclidean chart (after unwrapping hyp part if any).
    Returns: dx, dy, dtheta (B,)
    """
    B, K, _ = P.shape
    Pc = P.mean(dim=1, keepdim=True)
    Qc = Q.mean(dim=1, keepdim=True)
    P0, Q0 = P - Pc, Q - Qc
    H = torch.matmul(P0.transpose(1,2), Q0)    # (B,2,2)
    U, S, Vt = torch.linalg.svd(H)
    R = torch.matmul(Vt.transpose(1,2), U.transpose(1,2))
    # enforce det=+1
    det = torch.linalg.det(R).unsqueeze(-1).unsqueeze(-1)
    Vt_adj = torch.cat([Vt[:,:,:1], Vt[:,:,1:]*det], dim=2)
    R = torch.matmul(Vt_adj.transpose(1,2), U.transpose(1,2))
    t = (Qc - torch.matmul(Pc, R.transpose(1,2))).squeeze(1)  # (B,2)
    dtheta = torch.atan2(R[:,1,0], R[:,0,0])
    return t[:,0], t[:,1], dtheta


# batching aux
def split_by_batch(x: Tensor, batch_vec: Tensor) -> List[Tensor]:
    """
    x: [N, D] node embeddings
    batch_vec: [N] graph ids in 0..B-1 (from data['agent'].batch)
    -> list of length B with tensors of shape [N_i, D]
    """
    B = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 1
    outs: List[Tensor] = []
    for b in range(B):
        outs.append(x[batch_vec == b])
    return outs

def reshape_fixed_count(x: Tensor, batch_vec: Tensor, count_per_graph: int) -> Tensor:
    """
    Fast path when every graph has the SAME number of agent nodes (e.g., 4 keypoints).
    x: [N, D], batch_vec: [N], count_per_graph: int
    -> [B, count_per_graph, D]
    """
    N, D = x.shape
    B = int(N // count_per_graph)
    # sanity check
    counts = torch.bincount(batch_vec, minlength=B)
    if not torch.all(counts == count_per_graph):
        raise ValueError(f"Not all graphs have {count_per_graph} nodes: counts={counts.tolist()}")
    return x.view(B, count_per_graph, D)

def pad_by_batch(x: Tensor, batch_vec: Tensor) -> Tuple[Tensor, Tensor]:
    """
    General (ragged) path: returns a padded tensor and a mask.
    x: [N, D], batch_vec: [N]
    -> padded: [B, max_len, D], mask: [B, max_len] with True for valid positions
    """
    parts = split_by_batch(x, batch_vec)        # list of [N_i, D]
    B = len(parts)
    D = x.shape[-1]
    max_len = max(p.shape[0] for p in parts) if parts else 0
    padded = x.new_zeros(B, max_len, D)
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=x.device)
    for b, p in enumerate(parts):
        n = p.shape[0]
        if n > 0:
            padded[b, :n] = p
            mask[b, :n] = True
    return padded, mask

def split_into_horizons(seq, pred_horizon: int):
    """
    seq: [B, L, ...]  (Kth demo sequence of length L)
    returns list of tensors, each [B, pred_horizon, ...]
    last chunk dropped if incomplete
    """
    B, L = seq.shape[0], seq.shape[1]
    T = pred_horizon
    num = (L - 1) // T   # -1 so we always have next frames
    chunks = []
    for i in range(num):
        s = i*T
        e = s + T         # [t=s .. s+T-1] predicts [s+1 .. s+T]
        chunks.append(seq[:, s:e, ...])
    return chunks