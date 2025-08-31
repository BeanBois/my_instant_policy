import torch 
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from utilities import se2_exp, se2_log, _wrap_to_pi
import math

from .policy import Policy
class Agent(nn.Module):

    def __init__(self, 
                geometric_encoder,
                max_translation = 500,
                max_rotation_deg = 30,
                max_diff_timesteps = 1000,
                beta_start = 1e-4,
                beta_end = 0.02,
                num_att_heads = 4,
                euc_head_dim = 16,
                pred_horizon = 5,
                in_dim_agent = 9,             
                ):

        super().__init__()
        self.policy = Policy(
            geometric_encoder,
                num_att_heads,
                euc_head_dim,
                pred_horizon,
                in_dim_agent,              
                )
        self.max_translation = max_translation
        self.max_rotation_rad = math.radians(max_rotation_deg)
        self.max_diff_timesteps = max_diff_timesteps
        betas = torch.linspace(beta_start, beta_end, max_diff_timesteps)  # linear; swap for cosine if you like
        self.register_buffer("betas", betas)                     # [K]
        alphas = 1.0 - betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))  # [K]
    
    def forward(self,
                curr_agent_info, # [B x self.num_agent_nodes x 6] x, y, theta, state, time, done
                curr_object_pos, # [B x M x 2] x,y
                demo_agent_info, # [B x N x L x self.num_agent_nodes x 6] x, y, theta, state, time, done
                demo_object_pos, # [B x N x L x M x 2]
                actions,         # [B, T, 4]
                ): 
        
        B, _, _ = actions.shape 
        device = actions.device 
        timesteps = torch.randint(0,self.max_diff_timesteps, (B,), device = device)
        noisy_actions, _, _ = self.add_action_noise(actions, timesteps) # [B, T, 4]

        denoising_directions_normalised = self.policy(
            curr_agent_info, # [B x self.num_agent_nodes x 6] x, y, theta, state, time, done
            curr_object_pos, # [B x M x 2] x,y
            demo_agent_info, # [B x N x L x self.num_agent_nodes x 6] x, y, theta, state, time, done
            demo_object_pos, # [B x N x L x M x 2]
            noisy_actions # [B, T, 4]
        ) # [B, T, 4, 5]
        # denoising_directions = self._unnormalise_denoising_directions(denoising_directions_normalised)

        return denoising_directions_normalised, noisy_actions

    @torch.no_grad()
    def plan_actions(
        self,
        curr_agent_info: Tensor,   # [B, A, ...]
        curr_object_pos: Tensor,   # [B, M, 2]
        demo_agent_info: Tensor,   # [B, N, L, A, ...]
        demo_object_pos: Tensor,   # [B, N, L, M, 2]
        actions: Tensor = None,    # [B, T, 4] start guess; if None, zeros
        K: int = 5,                # number of refinement/DDIM steps
        keypoints: Tensor = None,
        use_ddim: bool = True,
        ddim_steps: int = None     # override number of DDIM steps; default=K
    ) -> Tensor:
        """
        If use_ddim:
        Interprets each refinement pass as a DDIM step in Lie-action space.
        Uses self.alphas_cumprod (registered 1D tensor of length D) for schedule.
        Else:
        Falls back to your original K-pass rigid refinement.
        Returns: [B, T, 4]
        """
        device = curr_agent_info.device
        dtype  = curr_agent_info.dtype
        B = curr_agent_info.shape[0]

        # T (horizon) from actions or model attr
        if actions is None:
            T = getattr(self, "pred_horizon", None)
            assert T is not None, "Provide actions or set self.pred_horizon"
            actions = torch.zeros(B, T, 4, device=device, dtype=dtype)
        else:
            T = actions.shape[1]

        if not use_ddim:
            # ---- your original loop here (unchanged) ----
            for _ in range(K):
                denoise = self.policy(curr_agent_info, curr_object_pos,
                                    demo_agent_info, demo_object_pos, actions)  # [B,T,A,5]
                a_ref   = self._svd_refine_once(actions, denoise, keypoints)
                actions = a_ref
            actions[..., 2] = _wrap_to_pi(actions[..., 2])
            actions[..., 3:4] = actions[..., 3:4].clamp(0.0, 1.0)
            return actions

        # ----------------- DDIM mode -----------------
        ab = self.alphas_cumprod.to(device=device, dtype=dtype)   # [D]
        D  = ab.numel()
        steps = int(ddim_steps or K)
        assert steps >= 1, "DDIM requires at least 1 step"

        # Build t-schedule: D-1 -> 0 in 'steps' hops
        t_sched = torch.linspace(D - 1, 0, steps=steps + 1, device=device).round().long()  # [steps+1]

        # Start from current actions (interpreted as x_t); if you prefer pure noise, set actions ~ N
        x_t = torch.cat([se2_log(actions[..., :3]), actions[..., 3:4]], dim=-1)  # [B,T,4]

        for s in range(steps):
            t      = t_sched[s].item()
            t_prev = t_sched[s + 1].item()
            ab_t       = ab[t]
            ab_prev    = ab[t_prev]
            sqrt_ab_t  = ab_t.sqrt()
            sqrt1m_t   = (1 - ab_t).clamp_min(1e-12).sqrt()

            # current noisy actions for conditioning
            a_t = torch.cat([se2_exp(x_t[..., :3]), x_t[..., 3:4]], dim=-1)  # [B,T,4]

            # one-step denoise to get a0_hat
            denoise = self.policy(curr_agent_info, curr_object_pos,
                                demo_agent_info, demo_object_pos, a_t)      # [B,T,A,5]
            a0_hat  = self._svd_refine_once(a_t, denoise, keypoints)          # [B,T,4]
            x0_hat  = torch.cat([se2_log(a0_hat[..., :3]), a0_hat[..., 3:4]], dim=-1)  # [B,T,4]

            # ε̂ and DDIM update
            eps_hat = (x_t - sqrt_ab_t * x0_hat) / (sqrt1m_t + 1e-12)
            x_t     = ab_prev.sqrt() * x0_hat + (1 - ab_prev).clamp_min(1e-12).sqrt() * eps_hat

        # back to vector actions
        a0 = se2_exp(x_t[..., :3])
        s0 = x_t[..., 3:4].clamp(0.0, 1.0)
        out = torch.cat([a0, s0], dim=-1)
        out[..., 2] = _wrap_to_pi(out[..., 2])
        return out

    # get unnormalised noise and project to SE(2) to add noise 
    def add_action_noise(self, actions: torch.Tensor, t: torch.Tensor):
        """
        actions: [B,T,4] = (dx,dy,theta,state)
        t:       [B,T] int diffusion step
        returns: (noisy_actions [B,T,4], epsilon_target [B,T,3], state_eps [B,T])
                epsilon_target is the Lie noise xi you sampled (vx,vy,omega)
        """
        B,T,_ = actions.shape
        dev, dt = actions.device, actions.dtype

        transrot = actions[...,:3]
        state    = actions[...,-1]

        # clean SE(2)
        T_clean = self._se2_from_vec(transrot)         # [B,T,3,3]

        betas = self._betas_lookup(t).to(dev, dt)    # [B,T]
        sigma = torch.sqrt(betas).unsqueeze(-1)      # [B,T,1]

        # sample Lie noise ε ~ N(0, I) then scale by σ_t
        eps_body = torch.randn(B,T,3, device=dev, dtype=dt) * sigma  # [B,T,3]
        Xi = self._se2_exp(eps_body)                                       # [B,T,3,3]

        # left-invariant noising
        T_noisy = self._se2_compose(Xi, T_clean)                           # [B,T,3,3]

        noisy_vec = self._se2_to_vec(T_noisy)                               # [B,T,3]
        # wrap angle to (-pi,pi]
        noisy_vec[...,2] = ((noisy_vec[...,2] + torch.pi) % (2*torch.pi)) - torch.pi

        # state/gripper channel in R
        state_eps = torch.randn_like(state) * torch.sqrt(betas)       # [B,T]
        state_noisy = state + state_eps
        state_noisy = state_noisy.clamp(0.,1.)

        noisy_actions = torch.cat([noisy_vec, state_noisy.unsqueeze(-1)], dim=-1)
        return noisy_actions, eps_body, state_eps

    def _actions_vect_to_SE2_flat(self, actions):
        # (x,y,theta_rad,state) => SE(2).flatten() | state 
        B, T, _ = actions.shape
        device = actions.device 
        dtype = actions.dtype 

        dx = actions[..., 0]
        dy = actions[..., 1]
        th = actions[..., 2]
        st = actions[..., 3]

        c, s = torch.cos(th), torch.sin(th)
        # T_clean: [B,T,3,3]
        SE2 = torch.zeros(B, T, 3, 3, device=device, dtype=dtype)
        SE2[..., 0, 0] = c
        SE2[..., 0, 1] = -s
        SE2[..., 1, 0] = s
        SE2[..., 1, 1] = c
        SE2[..., 0, 2] = dx
        SE2[..., 1, 2] = dy
        SE2[..., 2, 2] = 1.

        SE2_flat = SE2.view(B,T,9)
        SE2_flat_final = torch.concat([SE2_flat, st], dim = -1)
        return SE2_flat_final

    def _se2_exp(self, xi: torch.Tensor):
        """
        xi: [...,3] -> (vx,vy,omega)  (body-frame twist)
        returns SE(2) matrix [...,3,3]
        """
        vx, vy, w = xi[...,0], xi[...,1], xi[...,2]
        eps = 1e-6
        sw, cw = torch.sin(w), torch.cos(w)
        w_safe  = torch.where(torch.abs(w) < eps, torch.ones_like(w), w)
        a = sw / w_safe
        b = (1. - cw) / w_safe

        # V @ v
        tx = a*vx - b*vy
        ty = b*vx + a*vy
        tx = torch.where(torch.abs(w) < eps, vx, tx)
        ty = torch.where(torch.abs(w) < eps, vy, ty)

        T = torch.zeros(*xi.shape[:-1], 3, 3, dtype=xi.dtype, device=xi.device)
        T[...,0,0] = cw;  T[...,0,1] = -sw; T[...,0,2] = tx
        T[...,1,0] = sw;  T[...,1,1] =  cw; T[...,1,2] = ty
        T[...,2,2] = 1.0
        return T

    def _se2_from_vec(self, vec: torch.Tensor):
        """
        vec: [...,3] -> (dx,dy,theta) to matrix
        """
        dx, dy, th = vec[...,0], vec[...,1], vec[...,2]
        c, s = torch.cos(th), torch.sin(th)
        T = torch.zeros(*vec.shape[:-1], 3, 3, dtype=vec.dtype, device=vec.device)
        T[...,0,0] = c;  T[...,0,1] = -s; T[...,0,2] = dx
        T[...,1,0] = s;  T[...,1,1] =  c; T[...,1,2] = dy
        T[...,2,2] = 1.0
        return T

    def _se2_to_vec(self, T: torch.Tensor):
        """
        T: [...,3,3] -> (dx,dy,theta)
        """
        dx = T[...,0,2]
        dy = T[...,1,2]
        th = torch.atan2(T[...,1,0], T[...,0,0])
        return torch.stack([dx,dy,th], dim=-1)

    def _se2_compose(self, A: torch.Tensor, B: torch.Tensor):
        """matrix compose with batch broadcasting: A @ B"""
        return A @ B

    def _betas_lookup(self, timesteps):
        return self.betas[timesteps]

    def _unnormalise_denoising_directions(self, x):
        # scale translation + per-node disp by length; keep state as-is
        return torch.cat([x[..., :2] * self.max_translation,
                        x[..., 2:4] * self.max_translation *torch.tensor([torch.cos(self.max_rotation_rad), torch.sin(self.max_rotation_rad)]),
                        x[..., 4:5]], dim=-1)

    def _svd_refine_once(self, actions: Tensor, denoise: Tensor, keypoints: Tensor = None) -> Tensor:
        """
        actions: [B,T,4]  (dx,dy,theta,state)
        denoise: [B,T,A,5] (tx,ty, dx_i,dy_i, ds)  -- your head’s layout
        keypoints: [A,2] (if None, uses self.keypoints or a small cross)
        returns: refined actions a0_hat [B,T,4]
        """
        device, dtype = actions.device, actions.dtype
        B, T, _ = actions.shape
        A = denoise.shape[2]

        # keypoints
        if keypoints is None:
            if hasattr(self, "keypoints") and self.keypoints is not None:
                kp0 = self.keypoints.to(device=device, dtype=dtype)[:A]
            else:
                s = torch.tensor(1.0, device=device, dtype=dtype)
                kp0 = torch.stack([
                    torch.tensor([0., 0.], device=device, dtype=dtype),
                    torch.tensor([ s, 0.], device=device, dtype=dtype),
                    torch.tensor([-s, 0.], device=device, dtype=dtype),
                    torch.tensor([0.,  s], device=device, dtype=dtype),
                    torch.tensor([0., -s], device=device, dtype=dtype),
                ], dim=0)[:A]
        kp = kp0.view(1,1,A,2).expand(B,T,A,2)

        dxdy = actions[..., :2]           # [B,T,2]
        th   = actions[..., 2:3]          # [B,T,1]
        st   = actions[..., 3:4]          # [B,T,1]

        c = torch.cos(th); s = torch.sin(th)
        kx, ky = kp[..., 0], kp[..., 1]
        Rx = c * kx - s * ky
        Ry = s * kx + c * ky
        P  = torch.stack([Rx, Ry], dim=-1) + dxdy.unsqueeze(2)  # [B,T,A,2]

        # denoise split
        denoise = self._unnormalise_denoising_directions(denoise)  # make sure this scales tx/ty & disp
        t_bias  = denoise[..., :2].mean(dim=2, keepdim=True)       # [B,T,1,2]
        disp    = denoise[..., 2:4]                                # [B,T,A,2]
        Q       = P + disp + t_bias

        # Procrustes (vectorized)
        muP = P.mean(dim=2, keepdim=True)
        muQ = Q.mean(dim=2, keepdim=True)
        X, Y = P - muP, Q - muQ
        H00 = (X[...,0]*Y[...,0]).sum(dim=2); H01 = (X[...,0]*Y[...,1]).sum(dim=2)
        H10 = (X[...,1]*Y[...,0]).sum(dim=2); H11 = (X[...,1]*Y[...,1]).sum(dim=2)
        H = torch.stack([torch.stack([H00,H01],dim=-1),
                        torch.stack([H10,H11],dim=-1)], dim=-2)        # [B,T,2,2]
        U, S, Vh = torch.linalg.svd(H)
        # det correction
        Rtmp = U @ Vh
        det  = torch.det(Rtmp)
        sign = torch.where(det < 0, torch.tensor(-1., device=device, dtype=dtype),
                                torch.tensor( 1., device=device, dtype=dtype))
        Sfix = torch.diag_embed(torch.stack([torch.ones_like(sign), sign], dim=-1))  # [B,T,2,2]
        R = U @ Sfix @ Vh

        dtheta = torch.atan2(R[...,1,0], R[...,0,0]).unsqueeze(-1)    # [B,T,1]
        # translation t = muQ - R*muP
        Rp = torch.einsum('btij,btaj->bta i', R, muP.expand_as(P))    # [B,T,A,2]
        t  = (muQ - Rp).mean(dim=2)                                   # [B,T,2]

        dxdy_hat = dxdy + t
        th_hat   = _wrap_to_pi(th.squeeze(-1) + dtheta.squeeze(-1)).unsqueeze(-1)
        s_hat    = (st + denoise[..., 4:5].mean(dim=2)).clamp(0.0, 1.0)
        return torch.cat([dxdy_hat, th_hat, s_hat], dim=-1)           # [B,T,4]
