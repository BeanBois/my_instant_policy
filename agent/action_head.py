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


