from data.game_aux import PlayerState
from data.interface import GameInterface
import torch 
import torch.nn as nn 

import numpy as np
import math 

from typing import List,Dict, Tuple

from data import PseudoDemoGenerator, GameMode, Action 
from train import PseudoDemoDataset  
from math import sqrt

AGENT_KEYPOINTS = [PseudoDemoGenerator.agent_keypoints[k] for k in PseudoDemoDataset.kp_order]   

def collect_demos(game_interface, num_demos, manual=True, max_demo_length = 20):
    provided_demos = []
    print(f"Collecting {num_demos} demonstrations...")
    if manual:
        for n in range(num_demos):
            game_interface.start_game()
            while game_interface.running:
                game_interface.step()
            
            # Process demo to extract relevant observations
            demo = game_interface.observations
            if len(demo) > 1:
                observations = downsample_obs(demo, max_demo_length)
                provided_demos.append(observations)
            
            game_interface.reset()
        return provided_demos
    
    print(f"Collected {len(provided_demos)} demonstrations")
    pass

def downsample_obs(observations, target_length):
        """
        Downsample observations to demo_length items.
        """
        
        if len(observations) <= target_length:
            return observations
        
        if target_length <= 0:
            return []
        
        if target_length == 1:
            # If we only want 1 observation, take the first one
            return [observations[0]]
        
        result = [observations[0]]  # Always include first observation
        
        if target_length > 1:
            result.append(observations[-1])  # Always include last observation
        
        # Fill in the middle observations
        if target_length > 2:
            middle_indices = []
            for i in range(1, target_length - 1):
                # Calculate position in the original sequence
                pos = 1 + (i - 1) * (len(observations) - 2) / (target_length - 2)
                actual_index = int(round(pos))
                actual_index = min(actual_index, len(observations) - 1)  # Clamp to valid range
                middle_indices.append(actual_index)
            
            # Insert middle observations in the correct order
            for i, idx in enumerate(middle_indices):
                result.insert(i + 1, observations[idx])
        
        return result

def process_obs(curr_obs: List[Dict], B,A,M,device):
    """
    curr_obs: list length B. Each element is a list of observation dicts
                (from PDGen._get_ground_truth: 'curr_obs_set').
    We take the FIRST obs of each sample as the 'current' one and turn it
    into tensors:
        - curr_agent_info: [B, A=4, 6] with [x,y,orientation,state,time,done]
        - curr_object_pos: [B, M, 2]  sampled coords
    """
    device = device

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

def process_context(context: List[Dict], B, N, L, M, A, device):
    """
    context: list length B; each element is a LIST of N demos.
                Each demo is (observations, actions) where:
                - observations: list length L of obs dicts (already downsampled in PDGen)
                - actions: tensor [L-1, 10] (accumulated, already downsampled in PDGen)
    Returns:
        demo_agent_info  : [B, N, L, A=4, 6]
        demo_object_pos  : [B, N, L, M, 2]
    """

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

        for n in range(N):
            observations = demos[n]  # observations: list L; actions: [L-1,10] torch
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

        all_demo_agent_info.append(torch.stack(demo_infos, dim=0))  # [N,L,4,6]
        all_demo_obj.append(torch.stack(demo_objs, dim=0))          # [N,L,M,2]

    demo_agent_info = torch.stack(all_demo_agent_info, dim=0)  # [B,N,L,4,6]
    demo_object_pos = torch.stack(all_demo_obj, dim=0)         # [B,N,L,M,2]

    return demo_agent_info, demo_object_pos

def action_from_vec(action): #x,y,theta,state_change [4] vector  
    state_action = None 
    for state in PlayerState:
        if round(torch.clamp(action[-1],0,1)) == state.value:
            state_action = state
            break 
    action_obj = Action(
        forward_movement = int(math.sqrt(action[0]**2 + action[1]**2)),
        rotation_deg = torch.deg(action[2]),
        state_change = state_action
    )
    return action_obj 

def rollout_once(game_interface, agent, num_demos = 2, max_demo_length = 20, 
                max_iter = 50, refine=10, keypoints=AGENT_KEYPOINTS, manual = True):
    
    demos = collect_demos(game_interface, num_demos, manual, max_demo_length)
    horizon = agent.pred_horizon
    done = False
    _t = 0
    # start game
    game_interface.change_mode(GameMode.AGENT_MODE)
    curr_obs = game_interface.start_game()
    demo_agent_info, demo_object_pos = process_context(demos)
    won = False 
    while not done and _t < max_iter:
        curr_agent_info, curr_object_pos = process_obs(curr_obs)
        for _ in range(horizon):
            actions = agent.plan_actions(
                curr_agent_info = curr_agent_info,         # shape as in training
                curr_object_pos = curr_object_pos,
                demo_agent_info = demo_agent_info,
                demo_object_pos = demo_object_pos,
                T = horizon,
                K = refine,
                keypoints = keypoints
            )  # [B,T,4]
            a0 = actions[:, 0]  # execute first step
            action_obj = action_from_vec(a0)
            curr_obs = game_interface.step(action_obj)
            done = curr_obs['done']
            if done:
                won = curr_obs['won']
                break
    return won 
        


if __name__ == "__main__":
    from train import TrainConfig
    from agent import GeometryEncoder, Agent
    import os 

    cfg = TrainConfig()
    geometry_encoder = GeometryEncoder(M = cfg.num_sampled_pc, out_dim=cfg.num_att_heads * cfg.euc_head_dim)

    state = torch.load("geometry_encoder_2d_frozen.pth", map_location="cpu")
    geometry_encoder.impl.load_state_dict(state)
    os.makedirs(cfg.out_dir, exist_ok=True)
    

    # --- Data

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
    agent_state_dict = torch.load('/checkpoints/...', map_location="cpu")
    agent.load_state_dict(state)

    print('Start evaluating')
    num_rollouts = 10
    wins = 0
    for _ in range(num_rollouts):
        game_interface = GameInterface(
            mode=GameMode.DEMO_MODE,
            # num_edibles=1,
            # num_obstacles=1,
        )
        wins += int(rollout_once(game_interface, agent))
    print(f'Won {wins} / {num_rollouts}!')
    


    

