from audioop import bias
import torch
import numpy as np
import random
from typing import Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .pseudo_game import PseudoGame 
from .pseudo_configs import SCREEN_HEIGHT as  PSEUDO_SCREEN_HEIGHT
from .pseudo_configs import SCREEN_WIDTH as  PSEUDO_SCREEN_WIDTH
import math 


# TODO: 
# deal with demo length i think?
# also the padding is a problem 


class PseudoDemoGenerator:
    agent_keypoints = PseudoGame.agent_keypoints

    def __init__(self, device, num_demos=5, min_num_waypoints=2, max_num_waypoints=6, 
                 num_threads=2, demo_length = 10, pred_horizon = 5, biased_odds = 0.5, augmented_odds = 0.1):
        self.num_demos = num_demos
        self.pred_horizon = pred_horizon
        self.min_num_waypoints = min_num_waypoints
        self.max_num_waypoints = max_num_waypoints
        self.device = device
        self.translation_scale = 500
        self.demo_length = demo_length

        self.player_speed = 5 
        self.player_rot_speed = 5
        self.num_threads = num_threads
        self.biased_odds = biased_odds
        self.augmented_odds = augmented_odds
        
        # Thread-local storage for agent keypoints
        self._thread_local = threading.local()

    def get_batch_samples(self, batch_size: int, biased = None, augmented = None) -> Tuple[torch.Tensor, List, torch.Tensor]:
        """
        Generate a batch of samples in parallel
        Returns:
            curr_obs_batch: List of batch_size current observations
            context_batch: List of batch_size contexts (each context is a list of demos)
            clean_actions_batch: Tensor of shape [batch_size, pred_horizon, 4]
        """
        # Use ThreadPoolExecutor to generate samples in parallel
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all sample generation tasks

            futures = [executor.submit(self._generate_single_sample, biased, augmented) for _ in range(batch_size)]
            
            # Collect results as they complete
            curr_obs_batch = []
            context_batch = []
            clean_actions_list = []

            
            for future in as_completed(futures):
                curr_obs, context, clean_actions = future.result()
                curr_obs_batch.append(curr_obs)
                context_batch.append(context)
                clean_actions_list.append(clean_actions)

        
        # Stack clean actions into a single tensor [batch_size, pred_horizon, 4]
        # clean_actions_batch = torch.stack(clean_actions_list, dim=0)
        clean_actions_batch = clean_actions_list
        
        return curr_obs_batch, context_batch, clean_actions_batch

    def _generate_single_sample(self, biased, augmented) -> Tuple[dict, List, torch.Tensor]:
        """Generate a single training sample (thread-safe)"""
        if biased is None:
            biased = random.random() < self.biased_odds
        if augmented is None:
            augmented = random.random() < self.augmented_odds
        pseudo_game = self._make_game(biased, augmented)
        context = self._get_context(pseudo_game)   
        curr_obs, clean_actions = self._get_ground_truth(pseudo_game)
        return curr_obs, context, clean_actions

    def get_agent_keypoints(self):
    
        agent_keypoints = torch.zeros((len(self.agent_key_points), 2), device=self.device)
        agent_keypoints[0] = torch.tensor(self.agent_key_points['front'], device=self.device)
        agent_keypoints[1] = torch.tensor(self.agent_key_points['back-left'], device=self.device)
        agent_keypoints[2] = torch.tensor(self.agent_key_points['back-right'], device=self.device)
        agent_keypoints[3] = torch.tensor(self.agent_key_points['center'], device=self.device)
        return agent_keypoints
    
    def _make_game(self, biased,augmented):
        player_starting_pos =(random.randint(0,PSEUDO_SCREEN_WIDTH), random.randint(0,PSEUDO_SCREEN_HEIGHT))
        return PseudoGame(
                    player_starting_pos=player_starting_pos,
                    max_num_sampled_waypoints=self.max_num_waypoints, 
                    min_num_sampled_waypoints=self.min_num_waypoints, 
                    biased=biased, 
                    augmented=augmented
                )

    def _run_game(self, pseudo_demo):
        max_retries = 1000
        # player_starting_pos =(random.randint(0,PSEUDO_SCREEN_WIDTH), random.randint(0,PSEUDO_SCREEN_HEIGHT))
        for attempt in range(max_retries):
            try: 
                # first reset 
                pseudo_demo.reset_game(shuffle=True) # config stays, but game resets (player, obj change positions)
                pseudo_demo.run()
                if len(pseudo_demo.actions) < 2:
                    continue
                return pseudo_demo
            except Exception as e:
                if attempt == max_retries-1:
                    raise 
                continue

    def _get_context(self, pseudo_game):
        context = []
        for _ in range(self.num_demos - 1):
            pseudo_demo = self._run_game(pseudo_game)
            observations = pseudo_demo.observations
            pd_actions = pseudo_demo.get_actions(mode='se2')
            se2_actions = np.array([action[0].flatten() for action in pd_actions]).reshape(-1, 9) # n x 9
            state_actions = np.array([action[1] for action in pd_actions]) # n x 1
            state_actions = state_actions.reshape(-1,1)
            actions = np.concatenate([se2_actions, state_actions], axis=1)
            actions = torch.tensor(
                actions, 
                dtype=torch.float, 
                device=self.device
            )          
            actions = self._accumulate_actions(actions)
            actions = self._downsample_actions(actions)
            observations = self._downsample_obs(observations)
            context.append((observations,actions))
        return context
            
    def _get_ground_truth(self, pseudo_game):
        pseudo_game.set_augmented(False) # actual test path never augmented, bias remains 
        pseudo_demo = self._run_game(pseudo_game)
        pd_actions = pseudo_demo.get_actions(mode='se2')
        se2_actions = np.array([action[0].flatten() for action in pd_actions]).reshape(-1, 9) # n x 9
        state_actions = np.array([action[1] for action in pd_actions]) # n x 1
        state_actions = state_actions.reshape(-1,1)
        actions = np.concatenate([se2_actions, state_actions], axis=1)
        actions = torch.tensor(
            actions, 
            dtype=torch.float, 
            device=self.device
        )          
        # actions = self._accumulate_actions(actions)
        observations = pseudo_demo.observations

        actions = self._downsample_actions_accumulate(actions)
        observations = self._downsample_obs(observations)
        curr_obs_set = []
        action_set = []
        # now i want to map self.prediciton_horizon => 1 obs
        for i in range(0, len(observations), self.pred_horizon):
            curr_obs = observations[i]
            curr_actions = actions[i:i + self.pred_horizon]  # Fix: start from i, take pred_horizon actions
            if len(curr_actions) > 0:
                curr_obs_set.append(curr_obs)
                action_set.append(curr_actions)

        # handle remainder
        return curr_obs_set, action_set

    # def _accumulate_actions(self, actions):
    #     n = actions.shape[0]
        
    #     # Extract and reshape SE(2) matrices
    #     se2_matrices = actions[:, :9].view(n, 3, 3)
    #     state_actions = actions[:, 9:]
        
    #     # Compute cumulative matrix products
    #     cumulative_matrices = torch.zeros_like(se2_matrices)
    #     cumulative_matrices[0] = se2_matrices[0]
        
    #     for i in range(1, n):
    #         cumulative_matrices[i] = torch.matmul(cumulative_matrices[i-1], se2_matrices[i])
        
    #     # Flatten back and concatenate with state actions
    #     cumulative_se2_flat = cumulative_matrices.view(n, 9)
    #     cumulative_actions = torch.cat([cumulative_se2_flat, state_actions], dim=1)
        
    #     return cumulative_actions

    # def _downsample_actions(self, actions):
    #     """
    #     Downsample actions to demo_length-1 items to match observations correspondence.
    #     If observations are downsampled to demo_length, actions should be demo_length-1.
    #     """
    #     target_length = self.demo_length - 1  # Actions should be one less than observations
        
    #     if actions.shape[0] <= target_length:
    #         return actions
        
    #     if target_length <= 0:
    #         return torch.empty((0, actions.shape[1]), device=actions.device)
        
    #     if target_length == 1:
    #         # If we only want 1 action, take the first one
    #         return actions[0:1]
        
    #     result = torch.zeros((target_length, actions.shape[1]), device=actions.device)
        
    #     # Always include first action
    #     result[0] = actions[0]
        
    #     if target_length > 1:
    #         # Always include last action
    #         result[-1] = actions[-1]
        
    #     # Fill in the middle actions
    #     if target_length > 2:
    #         for i in range(1, target_length - 1):
    #             # Calculate position in the original sequence
    #             pos = 1 + (i - 1) * (actions.shape[0] - 2) / (target_length - 2)
    #             actual_index = int(round(pos))
    #             actual_index = min(actual_index, actions.shape[0] - 1)  # Clamp to valid range
    #             result[i] = actions[actual_index]
        
    #     return result
    
    def _compose_se2_window(self, win: torch.Tensor) -> torch.Tensor:
        """
        Compose a sequence of SE(2) increments (tx, ty, theta) into a single transform.
        win: [W, 3]  (tx, ty, theta), increments applied in order.
        Returns: [3] net (tx, ty, theta).
        """
        if win.numel() == 0:
            return torch.zeros(3, dtype=win.dtype, device=win.device)

        # running pose
        TX = torch.zeros(2, dtype=win.dtype, device=win.device)  # accumulated translation in world frame
        TH = torch.zeros((), dtype=win.dtype, device=win.device) # accumulated heading (rad)

        for i in range(win.shape[0]):
            tx, ty, dth = win[i]
            c = torch.cos(TH); s = torch.sin(TH)
            # rotate local increment into world, then add
            dT_world = torch.stack([c*tx - s*ty, s*tx + c*ty])
            TX = TX + dT_world
            TH = TH + dth

        # wrap angle to [-pi, pi]
        pi = math.pi
        TH = (TH + pi) % (2 * pi) - pi
        return torch.stack([TX[0], TX[1], TH])

    def _downsample_actions_accumulate(self, actions: torch.Tensor, state_mode: str = "last"):
        """
        Downsample by composing actions inside windows.
        actions: [N, D] with columns (tx, ty, theta, [state...])
        Returns: [demo_length-1, D]
        """
        N, D = actions.shape
        target_length = self.demo_length - 1
        device, dtype = actions.device, actions.dtype

        if target_length <= 0:
            return torch.empty((0, D), device=device, dtype=dtype)
        if N == 0:
            return torch.zeros((target_length, D), device=device, dtype=dtype)
        if N == target_length:
            return actions  # already aligned 1:1 windows of size 1

        # Window edges (inclusive-exclusive) over action indices [0..N)
        # Use evenly spaced edges, then round to ints and fix monotonicity.
        edges_f = torch.linspace(0, N, steps=target_length + 1, device=device)
        edges = torch.round(edges_f).to(torch.long).clamp_(0, N)
        # ensure strictly increasing by forcing end >= start+1 where possible
        # If two edges collapse, we'll still produce something sensible (single best index).
        result = torch.zeros((target_length, D), device=device, dtype=dtype)

        has_state = (D >= 4)

        for i in range(target_length):
            start = edges[i].item()
            end   = edges[i+1].item()

            if end <= start:
                # degenerate window → pick nearest valid index
                idx = min(start, N-1)
                net_xyz = actions[idx, :3]
                if has_state:
                    if state_mode == "last":
                        st = actions[idx, 3]
                    elif state_mode == "sum":
                        st = actions[idx, 3]
                    elif state_mode == "any":
                        st = actions[idx, 3]
                    else:
                        st = actions[idx, 3]
                    result[i, :3] = net_xyz
                    result[i, 3]  = st
                else:
                    result[i, :3] = net_xyz
                continue

            window = actions[start:end]  # [W, D]
            net_xyz = self._compose_se2_window(window[:, :3])  # (tx,ty,theta) composed

            result[i, :3] = net_xyz

            if has_state:
                if state_mode == "last":
                    st = window[-1, 3]
                elif state_mode == "sum":
                    st = window[:, 3].sum().clamp(-1.0, 1.0)
                elif state_mode == "any":
                    # treat state as command; any nonzero -> 1
                    st = (window[:, 3] != 0).any().to(dtype)
                else:
                    st = window[-1, 3]
                result[i, 3] = st

            # If there are extra columns beyond 4, carry the last sample's values through
            if D > 4:
                result[i, 4:] = window[-1, 4:]

        return result

    def _downsample_obs(self, observations):
        """
        Downsample observations to demo_length items.
        """
        target_length = self.demo_length
        T = len(observations)
        if len(observations) <= target_length:
            return observations
        
        if target_length <= 0:
            return []
        
        if target_length == 1:
            # If we only want 1 observation, take the first one
            return [observations[0]]
        gripper = np.array([1 if observations[i]['agent-state'] else 0 for i in range(T)], dtype=np.int32)

        # indices where state changes (use the index of the NEW state)
        toggles = np.nonzero(np.diff(gripper) != 0)[0] + 1    # e.g., [5, 12, ...]
        candidates = [0, T-1] + toggles.tolist()
        keep = sorted(set(candidates))
        
        # --- prune or pad to reach target_length ---
        if len(keep) > target_length:
            # too many points: keep endpoints, choose a subset of toggles uniformly
            # separate endpoints and toggles to preserve ends
            ends = [0, T-1]
            toggle_only = [i for i in keep if i not in ends]

            if target_length <= 2:
                # only room for endpoints
                chosen = ends[:target_length]
            else:
                k = target_length - 2
                # pick k toggle indices uniformly spaced over toggle_only
                # (works even if there are many toggles)
                sel = np.round(np.linspace(0, len(toggle_only)-1, num=k)).astype(int)
                chosen_toggles = [toggle_only[j] for j in sel]
                chosen = sorted(set(ends + chosen_toggles))

            keep = chosen

        elif len(keep) < target_length:
            # not enough points: backfill with evenly spaced frames
            need = target_length - len(keep)
            keep_set = set(keep)

            # propose evenly spaced indices across [0, T-1]
            # avoid endpoints because they’re already in keep
            proposals = np.round(np.linspace(1, T-2, num=max(need*3, need))).astype(int).tolist()

            # add proposals that aren’t already kept, until we reach target_length
            for idx in proposals:
                if idx not in keep_set:
                    keep_set.add(int(idx))
                    if len(keep_set) >= target_length:
                        break

            # if still short (rare), fill by linear scan
            i = 1
            while len(keep_set) < target_length and i < T-1:
                if i not in keep_set:
                    keep_set.add(i)
                i += 1

            keep = sorted(keep_set)

        # --- build the result in temporal order ---
        return [observations[i] for i in keep]

    def _pad_to_length(self, arr: torch.Tensor, T: int, dim: int = 0) -> torch.Tensor:
        """
        Pad `arr` along `dim` with zeros so its size at `dim` is exactly T.
        """
        curr_len = arr.size(dim)
        if curr_len >= T:
            return arr[:T]  # truncate if too long
        pad_shape = list(arr.shape)
        pad_shape[dim] = T - curr_len
        pad = torch.zeros(pad_shape, dtype=arr.dtype, device=arr.device)
        return torch.cat([arr, pad], dim=dim)

