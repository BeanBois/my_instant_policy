import torch
import torch.nn as nn
import torch.nn.functional as F

# ========= Poincaré ball utilities (curvature c > 0) =========

def _safe_norm(x, dim=-1, keepdim=False, eps=1e-15):
    return torch.clamp(torch.norm(x, dim=dim, keepdim=keepdim), min=eps)

def _project_to_ball(x, c, eps=1e-5):
    # Ensure ||x|| < 1/sqrt(c)
    sqrt_c = c ** 0.5
    norm = _safe_norm(x, dim=-1, keepdim=True)
    max_norm = (1. - eps) / sqrt_c
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale

def lambda_x(x, c):
    # Conformal factor λ_x^c = 2 / (1 - c ||x||^2)
    x = _project_to_ball(x, c)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    return 2.0 / (1.0 - c * x2).clamp_min(1e-15)

def mobius_add(x, y, c):
    """
    Möbius addition on the Poincaré ball (Ganea et al. 2018).
    """
    x = _project_to_ball(x, c)
    y = _project_to_ball(y, c)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return num / den.clamp_min(1e-15)

def poincare_distance_sq(x, y, c):
    """
    d_c(x,y)^2 = ((2/√c) atanh( √c ||(-x) ⊕ y|| ))^2
    """
    sqrt_c = c ** 0.5
    x = _project_to_ball(x, c)
    y = _project_to_ball(y, c)
    diff = mobius_add(-x, y, c)               # (-x) ⊕ y
    norm = _safe_norm(diff, dim=-1)
    arg = torch.clamp(sqrt_c * norm, max=1 - 1e-7)
    dist = (2.0 / sqrt_c) * torch.atanh(arg)
    return dist * dist

def log_map_x(x, y, c):
    """
    log_x(y) on the Poincaré ball.
    log_x(y) = (2 / (λ_x sqrt(c))) * atanh( sqrt(c) ||(-x) ⊕ y|| ) * u / ||u||
               where u = (-x) ⊕ y
    """
    sqrt_c = c ** 0.5
    lam = lambda_x(x, c)                      # [*, 1]
    u = mobius_add(-x, y, c)
    unorm = _safe_norm(u, dim=-1, keepdim=True)
    # avoid 0 direction
    u_dir = u / unorm
    arg = torch.clamp(sqrt_c * unorm, max=1 - 1e-7)
    scale = (2.0 / (lam * sqrt_c)) * torch.atanh(arg)  # [*,1]
    return scale * u_dir

def exp_map_x(x, v, c):
    """
    exp_x(v) on the Poincaré ball.
    exp_x(v) = x ⊕ ( tanh( (λ_x sqrt(c)/2) ||v|| ) * v / (sqrt(c) ||v||) )
    """
    sqrt_c = c ** 0.5
    lam = lambda_x(x, c)
    vnorm = _safe_norm(v, dim=-1, keepdim=True)
    # direction
    v_dir = v / vnorm
    # scale
    factor = torch.tanh((lam * sqrt_c * vnorm) / 2.0) / (sqrt_c)
    y = v_dir * factor
    out = mobius_add(x, y, c)
    return _project_to_ball(out, c)

# ========= Product-manifold attention layer =========

class ProductManifoldAttention(nn.Module):
    """
    Product-space attention (H × E):
      - Scores from product metric: s = -(λ_H d_H^2 + λ_E ||.||^2) / τ
      - Euclidean aggregation: weighted average
      - Hyperbolic aggregation: weighted Karcher mean (log/exp at current point)

    Forward signature (as requested):
        def forward(self,
            curr_rho_batch,  # [B, A, de]
            curr_hyp_emb,    # [B, dh]
            demo_rho_batch,  # [B, N, L, A, de]
            demo_hyp_emb     # [B, N, L, dh]
        ) -> torch.Tensor:    # returns [B, A, z_dim]
    """
    def __init__(
        self,
        de: int,              # Euclidean feature dim (de)
        dh: int,              # Hyperbolic dim (dh)
        z_dim: int,           # output latent dim
        curvature: float = 1.0,
        tau: float = 1.0,
        lambda_euc: float = 1.0,
        lambda_hyp: float = 1.0,
        use_layernorm: bool = True,
        dropout: float = 0.0,
        proj_hidden: int = 0,   # 0 = direct proj to z; >0 = small MLP per factor
    ):
        super().__init__()
        assert curvature > 0, "Poincaré ball curvature c must be > 0."
        self.c = curvature
        self.tau = tau
        self.le = lambda_euc
        self.lh = lambda_hyp

        # Factor-wise projections → latent
        # Split z into two halves (E and H); if odd, Euclidean gets the extra unit.
        z_e = z_dim // 2 + (z_dim % 2)
        z_h = z_dim // 2
        def make_head(in_dim, out_dim):
            if proj_hidden and proj_hidden > 0:
                return nn.Sequential(
                    nn.Linear(in_dim, proj_hidden, bias=False),
                    nn.GELU(),
                    nn.Linear(proj_hidden, out_dim, bias=False),
                )
            else:
                return nn.Linear(in_dim, out_dim, bias=False)

        self.proj_e = make_head(de, z_e)
        self.proj_h = make_head(dh, z_h)

        self.norm = nn.LayerNorm(z_dim) if use_layernorm else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _flatten_demos(demo_rho_batch, demo_hyp_emb):
        """
        Flatten over (N, L) → M = N*L.
        demo_rho_batch: [B,N,L,A,de] → [B,M,A,de]
        demo_hyp_emb:   [B,N,L,dh]   → [B,M,dh]
        """
        B, N, L, A, de = demo_rho_batch.shape
        M = N * L
        demo_rho_flat = demo_rho_batch.view(B, M, A, de)
        demo_hyp_flat = demo_hyp_emb.view(B, M, -1)  # dh
        return demo_rho_flat, demo_hyp_flat  # [B,M,A,de], [B,M,dh]

    def forward(self,
                curr_rho_batch: torch.Tensor,  # [B, A, de]
                curr_hyp_emb: torch.Tensor,    # [B, dh]
                demo_rho_batch: torch.Tensor,  # [B, N, L, A, de]
                demo_hyp_emb: torch.Tensor     # [B, N, L, dh]
                ) -> torch.Tensor:             # [B, A, z_dim]
        device = curr_rho_batch.device
        B, A, de = curr_rho_batch.shape
        _, dh = curr_hyp_emb.shape

        demo_rho_flat, demo_hyp_flat = self._flatten_demos(demo_rho_batch, demo_hyp_emb)
        # Shapes:
        #   demo_rho_flat: [B, M, A, de]
        #   demo_hyp_flat: [B, M, dh]
        M = demo_rho_flat.shape[1]

        # ---------- Product-metric attention scores ----------
        # Euclidean squared distances per agent: ||curr_e - demo_e||^2
        # curr_e: [B, 1, A, de] vs demo_e: [B, M, A, de] -> [B, M, A]
        curr_e = curr_rho_batch.unsqueeze(1).expand(B, M, A, de)
        e_diff_sq = ((curr_e - demo_rho_flat) ** 2).sum(dim=-1)  # [B, M, A]
        e_diff_sq = e_diff_sq.permute(0, 2, 1).contiguous()      # [B, A, M]

        # Hyperbolic squared distances (independent of A, then broadcast):
        # curr_h: [B, dh], demo_h: [B, M, dh] -> [B, M]
        dH2 = poincare_distance_sq(
            curr_hyp_emb.unsqueeze(1).expand(B, M, dh),  # [B,M,dh]
            demo_hyp_flat,                                # [B,M,dh]
            self.c
        )  # [B, M]
        dH2 = dH2.unsqueeze(1).expand(B, A, M).contiguous()  # [B, A, M]

        # scores: s = -(λ_H d_H^2 + λ_E ||.||^2) / τ
        scores = -(self.lh * dH2 + self.le * e_diff_sq) / max(self.tau, 1e-8)  # [B, A, M]

        # ---------- Attention weights ----------
        alpha = torch.softmax(scores, dim=-1)  # [B, A, M]

        # ---------- Euclidean aggregation (weighted mean) ----------
        # demo_rho_flat: [B, M, A, de] → [B, A, M, de]
        demo_e_for_agg = demo_rho_flat.permute(0, 2, 1, 3).contiguous()
        e_out = torch.sum(alpha.unsqueeze(-1) * demo_e_for_agg, dim=2)  # [B, A, de]

        # ---------- Hyperbolic aggregation (Karcher mean via log/exp at curr_h) ----------
        # log_{x}(y_m): x = curr_hyp_emb; y_m = demo_hyp_flat
        # Produce logs for each (B, M, dh), then weight per agent with alpha[B,A,M].
        x = curr_hyp_emb  # [B, dh]
        y = demo_hyp_flat # [B, M, dh]
        # Compute log vectors once (no A), then combine with per-agent alpha
        log_vecs = log_map_x(
            x.unsqueeze(1).expand(B, M, dh),  # [B,M,dh]
            y,                                # [B,M,dh]
            self.c
        )  # [B, M, dh]
        # Weight per-agent: alpha [B,A,M] -> [B,A,M,1]
        v = torch.sum(alpha.unsqueeze(-1) * log_vecs.unsqueeze(1), dim=2)  # [B, A, dh]
        # Exp back at x (per agent): base x is same across A; broadcast to [B,A,dh]
        x_ba = x.unsqueeze(1).expand(B, A, dh)
        h_out = exp_map_x(x_ba, v, self.c)  # [B, A, dh]

        # ---------- Project factors and combine ----------
        z_e = self.proj_e(e_out)            # [B, A, z_e]
        z_h = self.proj_h(h_out)            # [B, A, z_h]
        z = torch.cat([z_e, z_h], dim=-1)   # [B, A, z_dim]

        z = self.drop(self.norm(z))         # optional LN/Dropout
        return z
