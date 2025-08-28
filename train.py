import os
import math
import json
import random
from dataclasses import dataclass
from turtle import forward
from typing import List, Tuple, Optional, Dict

from agent import Agent
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np




# ---------------------------
# Small hyperbolic utilities
# (Poincaré ball; curvature c>0)
# ---------------------------
class Poincare:
    @staticmethod
    def mobius_add(x, y, c):
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
        den = 1 + 2*c*xy + c*c*x2*y2
        return num / torch.clamp(den, min=1e-15)

    @staticmethod
    def lambda_x(x, c):
        return 2.0 / torch.clamp(1 - c*(x*x).sum(dim=-1, keepdim=True), min=1e-15)

    @staticmethod
    def poincare_dist(x, y, c, eps=1e-15):
        # d_c(x,y) = arcosh(1 + 2c||x-y||^2 / ((1 - c||x||^2)(1 - c||y||^2)))
        x2 = torch.clamp((x*x).sum(dim=-1), max=(1.0 - 1e-6)/c)  # keep inside ball
        y2 = torch.clamp((y*y).sum(dim=-1), max=(1.0 - 1e-6)/c)
        diff2 = ((x - y)**2).sum(dim=-1)
        num = 2*c*diff2
        den = (1 - c*x2) * (1 - c*y2)
        z = 1 + torch.clamp(num / torch.clamp(den, min=eps), min=1+1e-7)
        return torch.acosh(z)

    @staticmethod
    def project_to_ball(x, c, eps=1e-5):
        # keep inside radius 1/sqrt(c)
        r = 1.0 / math.sqrt(c)
        norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
        scale = torch.clamp((r - eps)/norm, max=1.0)
        return x * scale


# ---------------------------
# Data interface (stub)
# Replace with your real dataset that returns tensors
# ---------------------------
@dataclass
class Item:
    # Shapes must match your policy.forward signature
    curr_agent_info: torch.Tensor       # [B, A, 6]
    curr_object_pos: torch.Tensor       # [B, M, 2]
    clean_actions: torch.Tensor         # [B, T, 3] (or your action dim)
    demo_agent_info: torch.Tensor       # [B, N, L, A, 6]
    demo_object_pos: torch.Tensor       # [B, N, L, M, 2]
    demo_agent_action: torch.Tensor     # [B, N, L-1, 3]
    # Optional/time channels you may already export:
    demo_time: Optional[torch.Tensor] = None  # [B, N, L] monotonically increasing
    curr_time: Optional[torch.Tensor] = None  # [B] or [B, A] if per-node
 
from data import PseudoDemoGenerator

class PseudoDemoDataset(Dataset):
    agent_kp = PseudoDemoGenerator.agent_keypoints
    kp_order = ["front", "back-left", "back-right", "center"]
    def __init__(self, length=10000, device="cpu",
                 B=4, A=4, M=64, N=2, L=10, T=8, action_dim=3):
        self.length = length
        self.device = device
        self.B, self.A, self.M, self.N, self.L, self.T = B, A, M, N, L, T
        self.action_dim = action_dim
        self.data_gen = PseudoDemoGenerator(device, num_demos = self.N + 1, demo_length = self.L, pred_horizon = self.T)

    def __len__(self):
        return self.length

    def __getitem__(self, idx) -> Item:
        B, A, M, N, L, T, ad = self.B, self.A, self.M, self.N, self.L, self.T, self.action_dim
        # Dummy random sample — replace with your real loading logic.
        curr_agent_info = torch.randn(B, A, 6)
        curr_object_pos = torch.randn(B, M, 2)
        clean_actions   = torch.randn(B, T, ad)

        demo_agent_info   = torch.randn(B, N, L, A, 6)
        demo_object_pos   = torch.randn(B, N, L, M, 2)
        demo_agent_action = torch.randn(B, N, L-1, ad)

        curr_obs, context, _clean_actions = self.data_gen.get_batch_samples(self.B)
        curr_agent_info, curr_object_pos = self._process_obs(curr_obs)
        demo_agent_info, demo_object_pos, demo_agent_action = self._process_context(context)
        clean_actions = self._process_actions(_clean_actions)

        # Monotone times for each demo traj
        base = torch.arange(L).float()[None, None, :].expand(B, N, L)
        noise = 0.01*torch.randn_like(base)
        demo_time = base + noise
        curr_time = (L-1) * torch.ones(B)  # treat current as “after” the last demo waypoint if you prefer

        return Item(
            curr_agent_info, curr_object_pos, clean_actions,
            demo_agent_info, demo_object_pos, demo_agent_action,
            demo_time=demo_time, curr_time=curr_time
        )

    def _process_obs(self, curr_obs: List[Dict]):
        """
        curr_obs: list length B. Each element is a list of observation dicts
                  (from PDGen._get_ground_truth: 'curr_obs_set').
        We take the FIRST obs of each sample as the 'current' one and turn it
        into tensors:
          - curr_agent_info: [B, A=4, 6] with [x,y,orientation,state,time,done]
          - curr_object_pos: [B, M, 2]  sampled coords
        """
        B, A, M = self.B, self.A, self.M
        device = self.device

        # fixed keypoint order (matches 4 nodes expected by A=4)
        kp_local = [PseudoDemoGenerator.agent_keypoints[k] for k in PseudoDemoDataset.kp_order]  # local-frame offsets
        kp_local = torch.tensor(kp_local, dtype=torch.float32, device=device)  # [4,2]

        agent_infos = []
        obj_coords_all = []

        for b in range(B):
            # Use the first "current" obs for this sample
            ob = curr_obs[b][0] if isinstance(curr_obs[b], list) else curr_obs[b]

            # Scalars
            cx, cy = float(ob["agent-pos"][0][0]), float(ob["agent-pos"][0][1])
            ori_deg = float(ob["agent-orientation"])
            ori_rad = math.radians(ori_deg)
            st = ob["agent-state"]
            st_val = float(getattr(st, "value", st))  # enum -> int if needed
            t_val = float(ob["time"])
            done_val = float(bool(ob["done"]))

            # Rotate local KPs to world and translate by agent center
            c, s = math.cos(ori_rad), math.sin(ori_rad)
            R = torch.tensor([[c, -s],
                              [s,  c]], dtype=torch.float32, device=device)     # [2,2]
            kp_world = (kp_local @ R.T) + torch.tensor([cx, cy], device=device)  # [4,2]

            # Pack [x,y,orientation,state,time,done] per keypoint
            o = torch.full((A, 1), ori_deg, dtype=torch.float32, device=device)
            stt = torch.full((A, 1), st_val, dtype=torch.float32, device=device)
            tt = torch.full((A, 1), t_val, dtype=torch.float32, device=device)
            dd = torch.full((A, 1), done_val, dtype=torch.float32, device=device)
            agent_info = torch.cat([kp_world, o, stt, tt, dd], dim=1)  # [4,6]
            agent_infos.append(agent_info)

            # Object coords → pick exactly M 2D points
            coords_np = ob["coords"]  # numpy array [K,2] (possibly K != M)
            K = int(coords_np.shape[0])
            if K == 0:
                # nothing detected → zeros
                sel = torch.zeros((M, 2), dtype=torch.float32, device=device)
            elif K >= M:
                idx = np.random.choice(K, size=M, replace=False)
                sel = torch.tensor(coords_np[idx], dtype=torch.float32, device=device)
            else:
                # not enough points → repeat with replacement
                idx = np.random.choice(K, size=M, replace=True)
                sel = torch.tensor(coords_np[idx], dtype=torch.float32, device=device)
            obj_coords_all.append(sel)

        curr_agent_info = torch.stack(agent_infos, dim=0)  # [B,4,6]
        curr_object_pos = torch.stack(obj_coords_all, dim=0)  # [B,M,2]
        return curr_agent_info, curr_object_pos

    def _process_actions(self, _clean_actions: List[List[torch.Tensor]]):
        """
        _clean_actions: list length B; each element is a LIST of length >=1,
                        where each entry is a [T, 10] tensor:
                          9 numbers = row-major SE(2) (3x3), then 1 gripper/state.
        We return [B, T, 3] with (tx, ty, theta) for the FIRST horizon sequence.
        """
        B, T = self.B, self.T
        device = self.device

        def mat_to_vec(m9: torch.Tensor) -> torch.Tensor:
            # m9 [9] row-major -> tx,ty,theta(rad)
            M = m9.view(3, 3)
            tx = M[0, 2]
            ty = M[1, 2]
            theta = torch.atan2(M[1, 0], M[0, 0])
            return torch.stack([tx, ty, theta], dim=0)  # [3]

        out = []
        for b in range(B):
            # take the first pred-horizon sequence for this sample
            seq = _clean_actions[b][0]  # [T, 10] on same device as generator set
            # Robustness: pad/truncate to T if needed
            Tb = seq.shape[0]
            if Tb < T:
                pad = torch.zeros((T - Tb, seq.shape[1]), dtype=seq.dtype, device=seq.device)
                seq = torch.cat([seq, pad], dim=0)
            elif Tb > T:
                seq = seq[:T]

            # Convert each step
            vecs = []
            for t in range(T):
                m9 = seq[t, :9]  # first 9 entries are SE(2)
                state_action = seq[t,-1].view(1)
                _vec = mat_to_vec(m9)
                vec = torch.concat([_vec,state_action])
                vecs.append(vec)
            vecs = torch.stack(vecs, dim=0).to(device)  # [T,3]
            out.append(vecs)

        return torch.stack(out, dim=0)  # [B,T,3]

    def _process_context(self, context: List[Tuple]):
        """
        context: list length B; each element is a LIST of N demos.
                 Each demo is (observations, actions) where:
                   - observations: list length L of obs dicts (already downsampled in PDGen)
                   - actions: tensor [L-1, 10] (accumulated, already downsampled in PDGen)
        Returns:
          demo_agent_info  : [B, N, L, A=4, 6]
          demo_object_pos  : [B, N, L, M, 2]
          demo_agent_action: [B, N, L-1, 3]  (tx, ty, theta)
        """
        B, N, L, A, M = self.B, self.N, self.L, self.A, self.M
        device = self.device

        kp_local = [PseudoDemoGenerator.agent_keypoints[k] for k in PseudoDemoDataset.kp_order]
        kp_local = torch.tensor(kp_local, dtype=torch.float32, device=device)  # [4,2]

        # Containers
        all_demo_agent_info = []
        all_demo_obj = []
        all_demo_act = []

        def obs_to_agent_info(ob):
            cx, cy = float(ob["agent-pos"][0][0]), float(ob["agent-pos"][0][1])
            ori_deg = float(ob["agent-orientation"])
            ori_rad = math.radians(ori_deg)
            st = ob["agent-state"]
            st_val = float(getattr(st, "value", st))
            t_val = float(ob["time"])
            done_val = float(bool(ob["done"]))

            c, s = math.cos(ori_rad), math.sin(ori_rad)
            R = torch.tensor([[c, -s],
                              [s,  c]], dtype=torch.float32, device=device)
            kp_world = (kp_local @ R.T) + torch.tensor([cx, cy], dtype=torch.float32, device=device)  # [4,2]

            o = torch.full((A, 1), ori_deg, dtype=torch.float32, device=device)
            stt = torch.full((A, 1), st_val, dtype=torch.float32, device=device)
            tt = torch.full((A, 1), t_val, dtype=torch.float32, device=device)
            dd = torch.full((A, 1), done_val, dtype=torch.float32, device=device)
            return torch.cat([kp_world, o, stt, tt, dd], dim=1)  # [4,6]

        def mat_to_vec(m9: torch.Tensor) -> torch.Tensor:
            M = m9.view(3, 3)
            tx = M[0, 2]
            ty = M[1, 2]
            theta = torch.atan2(M[1, 0], M[0, 0])
            return torch.stack([tx, ty, theta], dim=0)  # [3]

        for b in range(B):
            demos = context[b]  # list of N demos
            assert len(demos) == N, f"Expected {N} demos, got {len(demos)}"

            demo_infos = []
            demo_objs = []
            demo_acts = []

            for n in range(N):
                observations, actions = demos[n]  # observations: list L; actions: [L-1,10] torch
                # Observations → [L, A, 6] and [L, M, 2]
                agent_info_steps = []
                obj_steps = []
                for l in range(L):
                    if l >= len(observations):
                        agent_info_steps.append(torch.zeros((A, 6), dtype=torch.float32, device=device))
                        obj_steps.append(torch.zeros((M, 2), dtype=torch.float32, device=device))
                        continue
                    ob = observations[l]
                    agent_info_steps.append(obs_to_agent_info(ob))

                    coords_np = ob["coords"]
                    K = int(coords_np.shape[0])
                    if K == 0:
                        sel = torch.zeros((M, 2), dtype=torch.float32, device=device)
                    elif K >= M:
                        idx = np.random.choice(K, size=M, replace=False)
                        sel = torch.tensor(coords_np[idx], dtype=torch.float32, device=device)
                    else:
                        idx = np.random.choice(K, size=M, replace=True)
                        sel = torch.tensor(coords_np[idx], dtype=torch.float32, device=device)
                    obj_steps.append(sel)

                demo_infos.append(torch.stack(agent_info_steps, dim=0))  # [L,4,6]
                demo_objs.append(torch.stack(obj_steps, dim=0))          # [L,M,2]

                # Actions → [L-1, 3]
                act = actions  # [L-1,10]
                # robust pad/truncate to L-1 in case
                if act.shape[0] < L - 1:
                    pad = torch.zeros((L - 1 - act.shape[0], act.shape[1]),
                                      dtype=act.dtype, device=act.device)
                    act = torch.cat([act, pad], dim=0)
                elif act.shape[0] > L - 1:
                    act = act[:L - 1]

                vecs = []
                for i in range(act.shape[0]):
                    vecs.append(mat_to_vec(act[i, :9].to(device)))  # [3]
                demo_acts.append(torch.stack(vecs, dim=0))  # [L-1,3]

            all_demo_agent_info.append(torch.stack(demo_infos, dim=0))  # [N,L,4,6]
            all_demo_obj.append(torch.stack(demo_objs, dim=0))          # [N,L,M,2]
            all_demo_act.append(torch.stack(demo_acts, dim=0))          # [N,L-1,3]

        demo_agent_info = torch.stack(all_demo_agent_info, dim=0)  # [B,N,L,4,6]
        demo_object_pos = torch.stack(all_demo_obj, dim=0)         # [B,N,L,M,2]
        demo_agent_action = torch.stack(all_demo_act, dim=0)       # [B,N,L-1,3]
        return demo_agent_info, demo_object_pos, demo_agent_action



def collate_items(batch: List[Item]) -> Item:
    # Here each dataset __getitem__ already returns batched B samples;
    # if yours returns single samples, stack along dim 0 here.
    assert len(batch) == 1, "This stub returns batch-already tensors; adjust as needed."
    return batch[0]

# ---------------------------
# Losses
# ---------------------------
class PerNodeDenoisingMSELoss(nn.Module):
    """
    ------
    clean_actions : [B, T, 4]  -> (dx, dy, dtheta_rad, state)
    noisy_actions : [B, T, 4]  -> (dx, dy, dtheta_rad, state)
    pred_eps      : [B, T, A, 5]  predicted per-node denoising (Δt_x, Δt_y, Δr_x, Δr_y, Δs)
    keypoints     : [A, 2] (optional) agent/gripper keypoints in its local frame, centred.

    Returns
    -------
    loss : scalar tensor (mean MSE over B*T*A*5)
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction="mean")  # we average at the very end

    @staticmethod
    def _default_keypoints(device, dtype):
        # 6 simple, centred keypoints (units arbitrary; you can swap for your real gripper KP set).
        # A small star/hex pattern gives some spread so rotation is observable.
        kp = [PseudoDemoDataset.agent_kp[k] for k in PseudoDemoDataset.kp_order]
        pts = torch.tensor([kp], device=device, dtype=dtype)
        return pts  # [A,2]

    def forward(
        self,
        clean_actions: torch.Tensor,     # [B,T,4]
        noisy_actions: torch.Tensor,     # [B,T,4]
        pred_eps: torch.Tensor,          # [B,T,A,5]
        keypoints: torch.Tensor = None,  # [A,2]
    ) -> torch.Tensor:

        assert clean_actions.dim() == 3 and clean_actions.size(-1) == 4, "clean_actions [B,T,4]"
        assert noisy_actions.dim() == 3 and noisy_actions.size(-1) == 4, "noisy_actions [B,T,4]"
        assert pred_eps.dim() == 4 and pred_eps.size(-1) == 5, "pred_eps [B,T,A,5]"

        B, T, _ = clean_actions.shape
        _, _, A, _ = pred_eps.shape
        device = clean_actions.device
        dtype  = clean_actions.dtype

        if keypoints is None:
            keypoints = self._default_keypoints(device, dtype)  # [A,2]
        else:
            assert keypoints.shape == (A, 2), f"keypoints must be [A,2], got {tuple(keypoints.shape)}"
            keypoints = keypoints.to(device=device, dtype=dtype)

        # Split components
        dx_c, dy_c, th_c, s_c = clean_actions[..., 0], clean_actions[..., 1], clean_actions[..., 2], clean_actions[..., 3]
        dx_n, dy_n, th_n, s_n = noisy_actions[..., 0], noisy_actions[..., 1], noisy_actions[..., 2], noisy_actions[..., 3]

        # --- Translation delta (same for all nodes) --------------------------
        # Δt = t_clean - t_noisy
        dt_x = dx_c - dx_n     # [B,T]
        dt_y = dy_c - dy_n     # [B,T]

        # --- Rotation-around-centre delta (node-dependent) ------------------
        # Δr(node) = R_clean p_kp - R_noisy p_kp, for each node
        # Build R(θ) for clean and noisy: [B,T,2,2]
        cos_c, sin_c = torch.cos(th_c), torch.sin(th_c)
        cos_n, sin_n = torch.cos(th_n), torch.sin(th_n)

        R_c = torch.stack([
            torch.stack([ cos_c, -sin_c], dim=-1),
            torch.stack([ sin_c,  cos_c], dim=-1)
        ], dim=-2)  # [B,T,2,2]

        R_n = torch.stack([
            torch.stack([ cos_n, -sin_n], dim=-1),
            torch.stack([ sin_n,  cos_n], dim=-1)
        ], dim=-2)  # [B,T,2,2]

        # Apply to keypoints
        # keypoints: [A,2] -> broadcast to [B,T,A,2]
        kp = keypoints.view(1, 1, A, 2).expand(B, T, A, 2)  # [B,T,A,2]

        # (R @ p): [B,T,2,2] @ [B,T,A,2] -> [B,T,A,2]
        # do via einsum for clarity
        Rp_c = torch.einsum('btij,btaj->btai', R_c, kp)  # [B,T,A,2]
        Rp_n = torch.einsum('btij,btaj->btai', R_n, kp)  # [B,T,A,2]

        dr = Rp_c - Rp_n  # [B,T,A,2] -> per-node rotation component (Δr_x, Δr_y)

        # --- Gripper state delta --------------------------------------------
        ds = (s_c - s_n)  # [B,T]

        # --- Assemble GT epsilon per node -----------------------------------
        # ε*_k[node] = [Δt_x, Δt_y, Δr_x(node), Δr_y(node), Δs]
        # Broadcast translation and state deltas across nodes:
        dt = torch.stack([dt_x, dt_y], dim=-1).unsqueeze(2).expand(B, T, A, 2)  # [B,T,A,2]
        ds_exp = ds.unsqueeze(-1).unsqueeze(-1).expand(B, T, A, 1)              # [B,T,A,1]

        eps_gt = torch.cat([dt, dr, ds_exp], dim=-1)  # [B,T,A,5]

        # --- MSE -------------------------------------------------------------
        # NOTE: The paper normalises components to [-1,1] during training to balance magnitudes.
        # If you already normalise actions/flow elsewhere, plain MSE here is correct (ε-targets vs ε-preds) :contentReference[oaicite:3]{index=3}.
        loss = self.mse(pred_eps, eps_gt)

        return loss


# ---------------------------
# Trainer
# ---------------------------
@dataclass
class TrainConfig:
    device: str = "cpu"
    batch_size: int = 1      # Each dataset item already contains an internal B; keep 1 here for the stub
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_steps: int = 50000
    log_every: int = 50
    ckpt_every: int = 1000
    out_dir: str = "./checkpoints"
    grad_clip: float = 1.0
    amp: bool = True
    align_temp: float = 0.07
    hyp_curvature: float = 1.0
    hyp_margin: float = 0.2
    hyp_neg_w: float = 0.5
    time_window: float = 1.5   # for alignment positives
    lookahead: int = 1      # next-search step size
    num_sampled_pc = 8
    num_att_heads = 4
    euc_head_dim = 16
    hyp_dim = 2
    in_dim_agent = 9
    tau=0.5
    pred_horizon = 5
    demo_length = 20
    max_translation = 1000
    max_diffusion_steps = 1000
    beta_start = 1e-4
    beta_end = 0.02

    # flags
    train_geo_encoder = False




if __name__ == "__main__":
    from agent import GeometryEncoder, fulltrain_geo_enc2d
    import torch
    from contextlib import nullcontext

    
    cfg = TrainConfig()
    geometry_encoder = GeometryEncoder(M = cfg.num_sampled_pc, out_dim=cfg.num_att_heads * cfg.euc_head_dim)
    if cfg.train_geo_encoder:  
        geometry_encoder.impl = fulltrain_geo_enc2d(feat_dim=cfg.num_att_heads * cfg.euc_head_dim, num_sampled_pc= cfg.num_sampled_pc, save_path=f"geometry_encoder_2d")
    else:
        state = torch.load("geometry_encoder_2d_frozen.pth", map_location="cpu")
        geometry_encoder.impl.load_state_dict(state)
    os.makedirs(cfg.out_dir, exist_ok=True)
    

    # --- Data
    ds = PseudoDemoDataset(B=cfg.batch_size, T=cfg.pred_horizon, L = cfg.demo_length)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_items, num_workers=0)

    # --- Model
    agent = Agent(
        geometric_encoder=geometry_encoder,
        max_translation=cfg.max_translation,
        max_diff_timesteps=cfg.max_diffusion_steps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        num_att_heads=cfg.num_att_heads,
        euc_head_dim=cfg.euc_head_dim,
        pred_horizon=cfg.pred_horizon,
        in_dim_agent=cfg.in_dim_agent,
        curvature=cfg.hyp_curvature,
        tau=cfg.tau

    ).to(cfg.device)  # your policy encapsulates rho, PCA alignment, and dynamics

    # --- Losses
    pnn_loss = PerNodeDenoisingMSELoss()

    # --- Optim
    optim = AdamW([p for p in agent.parameters() if p.requires_grad],
                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cpu.amp.GradScaler(enabled=cfg.amp)

    step = 0
    agent.train()

    while step < cfg.max_steps:
        for item in dl:
            step += 1
            if step > cfg.max_steps:
                break

            # Move to device
            def dev(x): return None if x is None else x.to(cfg.device)
            curr_agent_info = dev(item.curr_agent_info)
            curr_object_pos = dev(item.curr_object_pos)
            clean_actions   = dev(item.clean_actions)
            demo_agent_info = dev(item.demo_agent_info)
            demo_object_pos = dev(item.demo_object_pos)
            demo_agent_action = dev(item.demo_agent_action)
            demo_time = dev(item.demo_time)
            curr_time = dev(item.curr_time)

            optim.zero_grad(set_to_none=True)
            use_amp = cfg.amp and torch.cuda.is_available()
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            with torch.cuda.amp.autocast(enabled=use_amp):

                pred_pn_denoising_dir, noisy_actions = agent.forward(
                    curr_agent_info,
                    curr_object_pos,
                    demo_agent_info,
                    demo_object_pos,
                    clean_actions
                )


                loss = pnn_loss(
                    clean_actions,
                    noisy_actions, 
                    pred_pn_denoising_dir
                )

            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

            if step % cfg.log_every == 0:
                print(f"[step {step:6d}] loss={loss.item():.4f}")

            if step % cfg.ckpt_every == 0:
                ckpt = {
                    "step": step,
                    "model": agent.state_dict(),
                    "optim": optim.state_dict(),
                    "cfg": cfg.__dict__,
                }
                path = os.path.join(cfg.out_dir, f"ckpt_{step:07d}.pt")
                torch.save(ckpt, path)
                print(f"Saved checkpoint to {path}")



