import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- small Fourier encoder for relative (dx,dy,theta) edges ---
class FourierEdgeEmbed(nn.Module):
    def __init__(self, num_freqs: int = 6, out_dim: int = 64):
        super().__init__()
        self.num_freqs = num_freqs
        self.proj = nn.Linear(3 * (2 * num_freqs) + 2 , out_dim)  # dx,dy,theta -> 2*num_freqs each + sin/cos(theta)

    def forward(self, dxyth: torch.Tensor):  # [B, T, A, 3 or 4]; last can include state_action
        # Use dx,dy,theta (ignore state_action for the edge encoding)
        dx = dxyth[..., 0:1]
        dy = dxyth[..., 1:2]
        th = dxyth[..., 2:3]
        # sin/cos(theta) (low-freq)
        trig = torch.cat([torch.sin(th), torch.cos(th)], dim=-1)

        # Fourier for dx,dy,theta
        feats = [trig]
        for k in range(self.num_freqs):
            f = (2.0 ** k) * math.pi
            feats += [torch.sin(f * dx), torch.cos(f * dx),
                      torch.sin(f * dy), torch.cos(f * dy),
                      torch.sin(f * th), torch.cos(f * th)]
        x = torch.cat(feats, dim=-1)  # [B,T,A, 4*(2*num_freqs)+2]
        return self.proj(x)           # [B,T,A,out_dim]


class PsiForesight(nn.Module):
    """
    Edge-aware cross-attn from Z_current (keys/values) --> Z_pred (queries),
    with edge features derived from (dx,dy,theta) like IP's key+edge formulation (Eq.3). :contentReference[oaicite:1]{index=1}
    """
    def __init__(self, z_dim: int, edge_dim: int = 64, n_heads: int = 8, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        assert z_dim % n_heads == 0, "z_dim must be divisible by n_heads"
        self.z = z_dim
        self.h = n_heads
        self.hd = z_dim // n_heads

        # projections
        self.q_proj = nn.Linear(z_dim, z_dim, bias=False)
        self.k_proj = nn.Linear(z_dim, z_dim, bias=False)
        self.v_proj = nn.Linear(z_dim, z_dim, bias=False)
        self.e_proj = nn.Linear(edge_dim, z_dim, bias=False)  # add to K, like W5 * e_ij in Eq.(3). :contentReference[oaicite:2]{index=2}

        self.out = nn.Linear(z_dim, z_dim, bias=False)

        # norm + ffn
        self.ln_q = nn.LayerNorm(z_dim)
        self.ln_o = nn.LayerNorm(z_dim)
        self.ff = nn.Sequential(
            nn.Linear(z_dim, ff_mult * z_dim),
            nn.GELU(),
            nn.Linear(ff_mult * z_dim, z_dim),
        )
        self.drop = nn.Dropout(dropout)

        self.edge_enc = FourierEdgeEmbed(num_freqs=6, out_dim=edge_dim)

        # optional learned gate (lets the layer decide how much to overwrite Z_pred)
        self.gate = nn.Sequential(nn.Linear(2 * z_dim, z_dim), nn.GELU(), nn.Linear(z_dim, 1))

    def _split_heads(self, x):  # [B,*,A,z] -> [B,*,A,h,hd]
        B = x.shape[0]
        *rest, A, _ = x.shape[:-2], x.shape[-2], x.shape[-1]
        x = x.view(B, *x.shape[1:-1], self.h, self.hd)
        return x

    def _merge_heads(self, x):  # [B,*,A,h,hd] -> [B,*,A,z]
        B = x.shape[0]
        *rest, A, h, hd = x.shape[1:]
        return x.reshape(B, *rest,A,h*hd)

    @torch.no_grad()
    def _safe_softmax(self, logits, dim):
        return torch.softmax(logits, dim=dim)

    def forward(self,
                z_current: torch.Tensor,          # [B, A, z]
                z_pred: torch.Tensor,             # [B, T, A, z]
                actions: torch.Tensor,            # [B, T, 4] -> (dx,dy,dtheta,state_action)
                state_gate_from_action: bool = False  # if True, use the 4th action channel to gate updates
                ) -> torch.Tensor:
        B, T, A, _ = z_pred.shape
        device = z_pred.device
        dtype = z_pred.dtype

        # Expand Z_current across time to match [B,T,A,z]
        zc = z_current.unsqueeze(1).expand(B, T, A, self.z)

        # Pre-norm queries (Z_pred)
        q = self.q_proj(self.ln_q(z_pred))   # [B,T,A,z]
        k = self.k_proj(zc)                  # [B,T,A,z]
        v = self.v_proj(zc)                  # [B,T,A,z]

        # Edge encoding from (dx,dy,theta) # fix from here?
        # e = self.edge_enc(actions[..., :3]).unsqueeze(2).expand(B, T, A, A, -1)  # broadcast per (i <- j); simple A-to-A same edge
        # # We only have a single relative per-time-step transform from "current->action".
        # # For per-agent edges, we share the same edge code across j (keeps compute light).
        # # Fold edge into K (Eq.3 style). :contentReference[oaicite:3]{index=3}
        # e_k = self.e_proj(actions[..., :3].new_zeros(B, T, A, self.z))  # placeholder
        # Simpler: add the same edge bias to every K_j at that t
        e_bias = self.e_proj(self.edge_enc(actions[..., :3]))  # [B,T,edge_z]
        e_bias = e_bias.unsqueeze(2).expand(B, T, A, self.z)   # [B,T,A,z]
        k = k + e_bias

        # Split heads: [B,T,A,h,hd]
        qh = self._split_heads(q)
        kh = self._split_heads(k)
        vh = self._split_heads(v)

        # attention over j (current agents) for each i (pred agents), per t
        # logits: [B,T,A,i,h,hd] x [B,T,A,j,h,hd] -> [B,T,h,A_i,A_j]
        logits = torch.einsum('btaih,btajh->btahij',
                              qh.reshape(B,T,A,self.h,self.hd),
                              kh.reshape(B,T,A,self.h,self.hd)) / math.sqrt(self.hd)
        # We want [B,T,h,A_i,A_j]; einsum above produced [B,T,h,A_i,A_j] already with dims ordering
        # (If your PyTorch version complains, you can do: (qh * kh).sum(-1) with broadcasting.)

        attn = self._safe_softmax(logits, dim=-1)  # softmax over j

        # weighted sum of values: [B,T,h,A_i,A_j] @ [B,T,A_j,h,hd] -> [B,T,h,A_i,hd]
        out_h = torch.einsum('btahij,btajh->btaih',
                             attn,
                             vh.reshape(B,T,A,self.h,self.hd))
        
        out = self._merge_heads(out_h)  # [B,T,A,z]
        out = self.out(out)
        out = self.drop(out)

        # learned residual gate (optionally modulated by action's state channel)
        gate_inp = torch.cat([out, z_pred], dim=-1)  # [B,T,A,2z]
        g = torch.sigmoid(self.gate(gate_inp))       # [B,T,A,1]

        if state_gate_from_action and actions.shape[-1] >= 4:
            # Use state_action (0..1) as an extra multiplicative gate per (t).
            sa = actions[..., 3:4].clamp(0, 1).unsqueeze(2).expand(B, T, A, 1)
            g = g * sa
        # Residual + FFN
        z_mid = z_pred + g * out
        z_mid = self.ln_o(z_mid)
        z_out = z_mid + self.ff(z_mid)
        return z_out
