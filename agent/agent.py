import torch 
import torch.nn as nn
import torch.nn.functional as F

from .policy import Policy
class Agent(nn.Module):

    def __init__(self, 
                geometric_encoder,
                max_translation = 500,
                max_diff_timesteps = 1000,
                beta_start = 1e-4,
                beta_end = 0.02,
                num_att_heads = 4,
                euc_head_dim = 32,
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
        denoising_directions = self._unnormalise_denoising_directions(denoising_directions_normalised)

        return denoising_directions, noisy_actions

    @torch.no_grad()
    def plan_actions(self,
                    curr_agent_info,   # [B,A_nodes,6]
                    curr_object_pos,   # [B,M,2]
                    demo_agent_info,   # [B,N,L,A_nodes,6]
                    demo_object_pos,   # [B,N,L,M,2]
                    T: int,            # horizon
                    K: int = 10,       # refinement steps
                    keypoints: torch.Tensor = None,  # [A,2] same KP set used in training loss
                    init: str = "gauss"  # or "repeat_prev_zero"
                    ):
        """
        Returns clean action sequence [B,T,4] in your (dx,dy,theta,state) parameterisation.
        """
        device = curr_agent_info.device
        B = curr_agent_info.shape[0]

        # --- initialise noisy actions x^{(0)} -----------------------------------
        if init == "gauss":
            dxdy = torch.randn(B, T, 2, device=device) * (self.max_translation / 10.0)
            theta = (torch.rand(B, T, 1, device=device) - 0.5) * (2*torch.pi/6)  # ~±30°
            state = torch.full((B,T,1), 0.5, device=device)
        else:
            dxdy = torch.zeros(B, T, 2, device=device)
            theta = torch.zeros(B, T, 1, device=device)
            state = torch.full((B,T,1), 0.5, device=device)

        actions = torch.cat([dxdy, theta, state], dim=-1)  # [B,T,4]

        if keypoints is None:
            kp = torch.tensor([(20, 0), 
                 (-14.747874310824908, 13.509263611023021), 
                 (-14.747874310824908, -13.509263611023021), 
                 (0, 0)]
                , device=device, dtype=actions.dtype
            )  # [A,2]
        else:
            kp = keypoints.to(device=device, dtype=actions.dtype)  # [A,2]
        A = kp.shape[0]

        for _ in range(K):
            
            # 1) predict per-node denoising directions ε_pred: [B,T,A,5]
            eps_pred_norm = self.policy(
                curr_agent_info, curr_object_pos,
                demo_agent_info, demo_object_pos,
                actions
            )
            eps_pred = self._unnormalise_denoising_directions(eps_pred_norm)

            # split components
            dt = eps_pred[..., 0:2]                   # [B,T,A,2] (same across A ideally)
            dr = eps_pred[..., 2:4]                   # [B,T,A,2]
            ds = eps_pred[..., 4]                     # [B,T,A]

            # 2) Build current (noisy) node positions under actions
            Rn = self._rot2d(actions[..., 2])         # [B,T,2,2]
            tn = actions[..., 0:2]                    # [B,T,2]
            Pn = torch.einsum("btij,aj->btai", Rn, kp) + tn.unsqueeze(-2)  # [B,T,A,2]

            # 3) Predicted clean node positions = current + predicted residuals
            #    also add the (shared) translation residual mean for stability
            dt_mean = dt.mean(dim=-2)                 # [B,T,2]
            Q = Pn + dr + dt_mean.unsqueeze(-2)       # [B,T,A,2]

            # 4) Recover SE(2) update via SVD (Pn -> Q)
            R_upd, t_upd = self._svd_align_2d(Pn, Q)  # [B,T,2,2], [B,T,2]

            # 5) Apply update to (dx,dy,theta) – left-compose
            #    Compose angles via atan2 from R_upd
            dth = torch.atan2(R_upd[...,1,0], R_upd[...,0,0]).unsqueeze(-1)  # [B,T,1]
            theta = (theta + dth + torch.pi) % (2*torch.pi) - torch.pi

            dxdy = dxdy + t_upd  # translation in world frame
            actions = torch.cat([dxdy, theta, state], dim=-1)

            # 6) Update gripper state with a small step & clamp
            ds_mean = ds.mean(dim=-1, keepdim=True)   # [B,T,1]
            state = (state + ds_mean).clamp(0., 1.)
            actions = torch.cat([dxdy, theta, state], dim=-1)

        # final clean actions
        return actions  # [B,T,4]

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

    def _unnormalise_denoising_directions(self, denoising_directions_normalised):
        
        denoising_directions_normalised = denoising_directions_normalised[:4] * self.max_translation 
        return denoising_directions_normalised 

    def _rot2d(self, theta):
        c, s = torch.cos(theta), torch.sin(theta)
        return torch.stack([torch.stack([c, -s], -1),
                            torch.stack([s,  c], -1)], -2)  # [...,2,2]

    @torch.no_grad()
    def _svd_align_2d(self, P, Q):
        """
        R,t that best aligns P->Q (Procrustes).
        P,Q: [B,T,A,2]
        Returns R:[B,T,2,2], t:[B,T,2]
        """
        muP = P.mean(dim=-2, keepdim=True)  # [B,T,1,2]
        muQ = Q.mean(dim=-2, keepdim=True)
        Pc, Qc = P - muP, Q - muQ
        # H = Pc^T Qc
        H = torch.einsum("btai,btaj->btij", Pc, Qc)  # [B,T,2,2]
        U, S, Vh = torch.linalg.svd(H)
        R = torch.einsum("btik,btkj->btij", Vh, U.transpose(-2,-1))
        # fix reflection
        det = torch.det(R).unsqueeze(-1).unsqueeze(-1)  # [B,T,1,1]
        Vh_fix = Vh.clone()
        mask = (det < 0)
        if mask.any():
            Vh_fix[mask, :, -1] *= -1
            R = torch.einsum("btik,btkj->btij", Vh_fix, U.transpose(-2,-1))
        t = (muQ - torch.einsum("btij,btaj->btai", R, muP)).squeeze(-2)  # [B,T,2]
        return R, t  # aligns P -> Q


