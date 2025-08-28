import torch
import torch.nn as nn

from utilities import * 

# aux
from .demo_handler import DemoHandler
from .graph import build_local_heterodata_batch, build_action_graph_batch, build_context_graph_batch
from .rho import Rho 
from .psi import Psi
from .phi import Phi
from .action_head import ProductManifoldGPHead, SimpleActionHead


class Policy(nn.Module):

    def __init__(self,
                geometric_encoder,
                num_att_heads,
                euc_head_dim,
                pred_horizon,
                in_dim_agent = 9):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.euc_dim = num_att_heads * euc_head_dim

        

        self.geometric_encoder = geometric_encoder 
        self.rho = Rho(
            in_dim_agent = in_dim_agent, # default by construction, 4 onehot + 5 scalars (sin,cos,state,time,done)
            in_dim_scene = num_att_heads * euc_head_dim,     
            edge_dim     = num_att_heads * euc_head_dim,        
            hidden_dim   = num_att_heads * euc_head_dim,
            out_dim      = num_att_heads * euc_head_dim,
            num_layers   = 2,
            heads        = num_att_heads,
            dropout      = 0.1,
            use_agent_agent = False
        )

        self.psi = Phi(
            dim=self.euc_dim
        )

        self.psi = Psi(
            dim=self.euc_dim
        )

        self.action_head = SimpleActionHead(
            in_dim=self.euc_dim,
            hidden_dim=self.euc_dim // 2
        )

              
    def forward(self,
                curr_agent_info, # [B x self.num_agent_nodes x 6] x, y, theta, state, time, done
                curr_object_pos, # [B x M x 2] x,y
                demo_agent_info, # [B x N x L x self.num_agent_nodes x 6] x, y, theta, state, time, done
                demo_object_pos, # [B x N x L x M x 2]
                actions # [B, T, 4]
                ):
        B, N, L, num_agent_nodes, agent_dim = demo_agent_info.shape
        _, _, _, num_object_nodes, obj_pos_dim = demo_object_pos.shape 

        ############################ First process demos into embeddings ############################

        ### get rho(G) for each demo
        flat_demo_agent_info = demo_agent_info.view(B * N * L, num_agent_nodes, agent_dim)
        flat_demo_object_pos = demo_object_pos.view(B * N * L, num_object_nodes, obj_pos_dim)
        
        ### first embed obj in demos
        F_list, C_list = [], []
        for i in range(B*N*L):
            # one cloud Pi: [M_raw, 2]  (use your raw per-frame points here)
            Pi = flat_demo_object_pos[i]           # [M_raw, 2]
            Fi, Ci = self.geometric_encoder(Pi)    # Fi: [M, D], Ci: [M, 2]
            F_list.append(Fi)
            C_list.append(Ci)
        flat_demo_scene_feats_batch = torch.stack(F_list, dim=0)  # [B*N*L, M, D]
        flat_demo_scene_pos_batch   = torch.stack(C_list, dim=0)  # [B*N*L, M, 2]

        flat_demo_local_graphs = build_local_heterodata_batch(
            agent_pos_b = flat_demo_agent_info,
            scene_pos_b = flat_demo_scene_pos_batch,
            scene_feats_b = flat_demo_scene_feats_batch,
            num_freqs = self.euc_dim // 4,
            include_agent_agent = False # no agent-agent edges 
        ) # returns HeteroBatch[B*N*L]

        ### get rho(G) for demos 
        demo_node_emb, _ = self.rho(flat_demo_local_graphs)
        ##### indv emb
        flat_demo_rho_batch = demo_node_emb['agent']                  # [B*N*L, A, euc_emb]        
        demo_rho_batch = flat_demo_rho_batch.view(B, N, L, num_agent_nodes, -1)    # [B,N,L,A,euc_emb]


        ############################ Now for current observation ###################################
        ### first get obj embeddings 
        F_list, C_list = [], []
        for i in range(B):
            Fi, Ci = self.geometric_encoder(curr_object_pos[i])  # [M,D], [M,2]
            F_list.append(Fi); C_list.append(Ci)
        curr_scene_feats = torch.stack(F_list, dim=0)  # [B, M, D]
        curr_scene_pos   = torch.stack(C_list, dim=0)  # [B, M, 2]

        ### build local graph
        curr_local_graph_batch = build_local_heterodata_batch(
            agent_pos_b = curr_agent_info,      # [B, A, 6]
            scene_pos_b = curr_scene_pos,       # [B, M, 2]
            scene_feats_b = curr_scene_feats,   # [B, M, D]
            num_freqs = self.euc_dim //4,
            include_agent_agent=False
        )

        ### get rho 
        curr_node_emb, _ = self.rho(curr_local_graph_batch)
        ##### indv emb 
        curr_rho_batch = curr_node_emb['agent']     # [B, A, De]

        ### impromptu shape fixes 
        if len(curr_rho_batch.shape) == 2:
            temp = curr_rho_batch.shape
            curr_rho_batch = curr_rho_batch.view(1,*temp)

        
        ############################ Then align context ###################################
        context_graph = build_context_graph_batch(
            curr_agent_emb = curr_rho_batch,
            demo_agent_emb = demo_rho_batch
        ) # like in instant policy
        curr_agent_emb_context_aligned_batch = self.phi(context_graph)  # [B,A,de]
        curr_agent_emb_context_aligned_batch = curr_agent_emb_context_aligned_batch.view(-1, num_agent_nodes, self.euc_dim)       # because each graph has exactly A 'curr' nodes
       
        ############################ Then Actions ############################ 
        pred_obj_info, pred_agent_info = self._perform_reverse_action(actions, curr_object_pos, curr_agent_info)
        
        # with pred info flatten, then make hetero graph 
        B,T,M, _ = pred_obj_info.shape  # T = self.pred_horizon
        flat_pred_obs_info = pred_obj_info.view(B*T, M, -1)
        flat_pred_agent_info = pred_agent_info.view(B*T, num_agent_nodes, -1)
        F_list, C_list = [], []
        for i in range(B*T):
            # one cloud Pi: [M_raw, 2]  (use your raw per-frame points here)
            Pi = flat_pred_obs_info[i]           # [M_raw, 2]
            Fi, Ci = self.geometric_encoder(Pi)    # Fi: [M, D], Ci: [M, 2]
            F_list.append(Fi)
            C_list.append(Ci)
        flat_pred_feats_batch = torch.stack(F_list, dim=0)  # [B*N*L, M, D]
        flat_pred_scene_pos_batch   = torch.stack(C_list, dim=0)  # [B*N*L, M, 2]


        flat_pred_local_graphs = build_local_heterodata_batch(
            agent_pos_b = flat_pred_agent_info,
            scene_pos_b = flat_pred_scene_pos_batch,
            scene_feats_b = flat_pred_feats_batch,
            num_freqs=self.euc_dim // 4,
            include_agent_agent=False 
        )

        ### get pred rho and hyp emb
        pred_node_emb, _ = self.rho(flat_pred_local_graphs)
        flat_pred_rho_batch = pred_node_emb['agent'] # [B*T, num_agent_nodes, self.euc_dim]
        pred_rho_batch = flat_pred_rho_batch.view(B,T, num_agent_nodes,-1) # [B, T, A, de]

        action_graph = build_action_graph_batch(
            curr_agent_emb = curr_agent_emb_context_aligned_batch, # [B, A, de]
            pred_agent_emb = pred_rho_batch # [B, T, A, de]
        )
        final_embd = self.psi(action_graph)  # [B, T, A, de]
        final_embd = final_embd.view(B, self.pred_horizon, num_agent_nodes, self.euc_dim)  
        denoising_direction = self.action_head(final_embd) # [B,T,5] tran_x, tran_y, rot_x, rot_y, state_change

        return denoising_direction

    def _perform_reverse_action(self,
                                actions: torch.Tensor,         # [B, T, 4] -> (dx, dy, dtheta_rad, state_action)
                                curr_object_pos: torch.Tensor, # [B, M, 2]
                                curr_agent_info: torch.Tensor  # [B, A, 6]: [ax, ay, cos, sin, grip, state_gate?]
                                ):
        """
        Apply inverse SE(2) to OBJECTS ONLY to simulate agent motion.
        Agents do not move here. Their info only changes in the gripper/state
        channel when state_action != current grip (per time step, per agent).
        """
        B, T, _ = actions.shape
        _, M, _ = curr_object_pos.shape
        _, A, C = curr_agent_info.shape
        device  = curr_object_pos.device
        dtype   = curr_object_pos.dtype

        # ---- split action channels ----
        dxdy   = actions[..., 0:2]            # [B,T,2]
        dtheta = actions[..., 2]              # [B,T]
        sa     = actions[..., 3].clamp(0, 1)  # [B,T] desired gripper/state command (0/1)

        # ---- OBJECTS: inverse SE(2) applied unconditionally ----
        # p_prev = R(-dθ) @ (p_curr - [dx,dy])
        dxdy_bt12 = dxdy.view(B, T, 1, 2)           # [B,T,1,2]
        obj_b1m2  = curr_object_pos.view(B, 1, M, 2)
        delta     = obj_b1m2 - dxdy_bt12            # [B,T,M,2]

        c = torch.cos(-dtheta).view(B, T, 1, 1)     # [B,T,1,1]
        s = torch.sin(-dtheta).view(B, T, 1, 1)

        x = delta[..., 0:1]
        y = delta[..., 1:2]
        x_rot =  c * x + s * y
        y_rot = -s * x + c * y
        pred_object_pos = torch.cat([x_rot, y_rot], dim=-1)  # [B,T,M,2]

        # ---- AGENTS: positions/orientations stay fixed; only grip may toggle ----
        # Broadcast static agent info across time
        ax   = curr_agent_info[..., 0].unsqueeze(1).expand(B, T, A)  # [B,T,A]
        ay   = curr_agent_info[..., 1].unsqueeze(1).expand(B, T, A)
        cth  = curr_agent_info[..., 2].unsqueeze(1).expand(B, T, A)
        sth  = curr_agent_info[..., 3].unsqueeze(1).expand(B, T, A)
        grip = curr_agent_info[..., 4].unsqueeze(1).expand(B, T, A)

        # Optional pass-through gate channel
        if C >= 6:
            gate = curr_agent_info[..., 5].unsqueeze(1).expand(B, T, A)
        else:
            gate = torch.ones(B, T, A, device=device, dtype=dtype)

        # Compare desired state to current grip; update ONLY when they differ
        sa_bta = sa.unsqueeze(-1).expand(B, T, A)   # [B,T,A]
        change_mask = (sa_bta.round() != grip.round())  # bool

        grip_out = torch.where(change_mask, sa_bta, grip)  # set to command when different, else keep

        # Repack without moving agents
        pred_agent_info = torch.stack([ax, ay, cth, sth, grip_out, gate], dim=-1)  # [B,T,A,6]

        return pred_object_pos, pred_agent_info

