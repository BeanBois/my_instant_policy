import torch
import torch.nn as nn 


class SimpleActionHead(torch.nn.Module):

    def __init__(self, hidden_dim, in_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.pred_head_p = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim,2),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        self.pred_head_rot = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        self.pred_head_g = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim,1),
            nn.Sigmoid(),  # Output in [0, 1] range for binary gripper actions
        )
    
    def forward(self, 
                embeddings # [B, T, self.in_dim]
                ):
        p = self.pred_head_p(embeddings)
        r = self.pred_head_rot(embeddings)
        g = self.pred_head_g(embeddings)
        return torch.concat([p,r,g], dim =-1)


def mobius_sqnorm(x):                  # ||x||^2
    return (x * x).sum(-1, keepdim=True)

def safe_ball_norm2(x, eps=1e-6):
    # clip radius to open ball
    r2 = mobius_sqnorm(x)
    return torch.clamp(r2, max=1 - 1e-6)

def hyp_distance_poincare(x, y, eps=1e-6):
    # x,y: [...,2], inside unit ball
    x2 = safe_ball_norm2(x, eps)
    y2 = safe_ball_norm2(y, eps)
    diff2 = ((x - y) ** 2).sum(-1)
    num = 2 * diff2
    den = (1 - x2.squeeze(-1)) * (1 - y2.squeeze(-1)) + eps
    arg = 1 + num / den
    # arcosh(u) = log(u + sqrt(u^2 - 1))
    return torch.log(arg + torch.sqrt(torch.clamp(arg * arg - 1, min=eps)))

def k_hyp(x, y, ell_h):
    # x:[N,2], y:[M,2]
    # returns [N,M]
    d = hyp_distance_poincare(x[:,None,:], y[None,:,:])  # [N,M]
    return torch.exp(-(d**2) / (2 * ell_h * ell_h))

def k_rbf(zx, zy, ell_z):
    # zx:[N,Z], zy:[M,Z] -> [N,M]
    diff = zx[:,None,:] - zy[None,:,:]
    d2 = (diff*diff).sum(-1)
    return torch.exp(-d2 / (2 * ell_z * ell_z))

def k_prod(hx, hy, zx, zy, ell_h, ell_z, sigma2_jitter=1e-5):
    Kh = k_hyp(hx, hy, ell_h)  # [N,M]
    Kz = k_rbf(zx, zy, ell_z)  # [N,M]
    K  = Kh * Kz               # [N,M]
    if hx.shape[0] == hy.shape[0] and torch.all(hx.eq(hy)) and torch.all(zx.eq(zy)):
        K = K + sigma2_jitter * torch.eye(hx.size(0), device=hx.device)
    return K

class ProductManifoldGPHead(torch.nn.Module):
    """
    Maps (h_{t+1}, z_t) -> a_t with a sparse GP:
      q(f(X)) ~ N( A K_uu^{-1} K_uX,  ... )
    We use a deterministic posterior mean for inference-time decode
    and train by NLL with a Gaussian noise model (can extend to VI).
    """
    def __init__(self, action_dim, z_dim, M=128, ell_h=0.7, ell_z=1.0, noise=1e-2):
        super().__init__()
        self.action_dim = action_dim
        self.z_dim = z_dim

        # Inducing points (learnable): on ball for h and Euclidean for z
        self.u_h = torch.nn.Parameter(0.1 * torch.randn(M, 2))     # keep in ball during step
        self.u_z = torch.nn.Parameter(0.1 * torch.randn(M, z_dim))

        # Kernel hyperparams (log-space so they stay positive)
        self.log_ell_h = torch.nn.Parameter(torch.log(torch.tensor(ell_h)))
        self.log_ell_z = torch.nn.Parameter(torch.log(torch.tensor(ell_z)))
        self.log_noise = torch.nn.Parameter(torch.log(torch.tensor(noise)))

        # Linear readout W: maps GP latent f_u (size M) to action_dim
        # We’ll compute f(X) = K_Xu K_uu^{-1} f_u, with f_u as free params
        self.f_u = torch.nn.Parameter(torch.zeros(M, action_dim))

    def project_ball(self, h, eps=1e-6):
        # keep h in open ball
        r = h.norm(dim=-1, keepdim=True)
        mask = (r >= 1 - 1e-5).float()
        h = h / torch.clamp(r, min=1.0) * (1 - 1e-5) * mask + h * (1 - mask)
        return h

    def _Kuu(self):
        ell_h = self.log_ell_h.exp()
        ell_z = self.log_ell_z.exp()
        Ku = k_prod(self.project_ball(self.u_h), self.project_ball(self.u_h),
                    self.u_z, self.u_z, ell_h, ell_z)  # [M,M] + jitter inside k_prod
        return Ku

    def forward(self, h_tp1, z_t):
        """
        h_tp1: [B,T,2]   (Poincaré ball)
        z_t  : [B,T,Z]
        returns: actions [B,T,A]
        """
        B,T,_ = h_tp1.shape
        Xh = h_tp1.reshape(B*T, 2)
        Xz = z_t.reshape(B*T, -1)

        ell_h = self.log_ell_h.exp()
        ell_z = self.log_ell_z.exp()

        Ku = self._Kuu()                              # [M,M]
        Ku_inv = torch.inverse(Ku)                    # [M,M]
        Kxu = k_prod(self.project_ball(Xh), self.project_ball(self.u_h),
                     Xz, self.u_z, ell_h, ell_z)      # [BT,M]

        # posterior mean at X: m(X) = K_xu K_uu^{-1} f_u
        mean = Kxu @ (Ku_inv @ self.f_u)              # [BT, A]
        actions = mean.reshape(B, T, self.action_dim)
        return actions

    def nll(self, h_tp1, z_t, a_target):
        pred = self.forward(h_tp1, z_t)               # [B,T,A]
        noise = self.log_noise.exp()
        return ((pred - a_target)**2).mean() / (2*noise) + 0.5*torch.log(noise)
