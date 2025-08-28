# local_graph_builder.py
from typing import Optional, List
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.data import Batch as HeteroBatch

from utilities import fourier_embed_2d

def _fully_connected_edges(n_src: int, n_dst: int) -> Tensor:
    src = torch.arange(n_src).repeat_interleave(n_dst)
    dst = torch.arange(n_dst).repeat(n_src)
    return torch.stack([src, dst], dim=0)  # [2, n_src*n_dst]

def _complete_graph_edges(n: int) -> Tensor:
    idx = torch.arange(n)
    src = idx.repeat_interleave(n)
    dst = idx.repeat(n)
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)  # [2, n*(n-1)]

def build_local_heterodata_single(
    agent_pos: Tensor,          # [4, 6] = [x,y,theta,state,time,done] per keypoint row
    scene_pos: Tensor,          # [M, 2]
    num_freqs: int = 10,
    include_agent_agent: bool = False,
    scene_feats: Optional[Tensor] = None,  # [M, d_s] or None
) -> HeteroData:
    assert agent_pos.ndim == 2 and agent_pos.size(0) > 0 and agent_pos.size(1) == 6, "agent_pos must be [4,6]"
    assert scene_pos.ndim == 2 and scene_pos.size(1) == 2, "scene_pos must be [M,2]"

    n_a = agent_pos.size(0)      # 4 (keypoints)
    n_s = scene_pos.size(0)      # M

    # split agent tensor
    pos_a = agent_pos[:, :2]     # [4,2]
    theta = agent_pos[0, 2]      # global fields (take from first row)
    state = agent_pos[0, 3]
    time  = agent_pos[0, 4]
    done  = agent_pos[0, 5]

    # per-node features for agent:
    # - one-hot node id (identity of each keypoint)
    # - broadcasted globals: sin(theta), cos(theta), state, time, done
    one_hot = torch.eye(n_a, device=agent_pos.device, dtype=agent_pos.dtype)           # [4,4]
    theta_sin = torch.sin(theta).expand(n_a, 1)
    theta_cos = torch.cos(theta).expand(n_a, 1)
    state_b   = state.expand(n_a, 1)
    time_b    = time.expand(n_a, 1)
    done_b    = done.expand(n_a, 1)
    x_agent = torch.cat([one_hot, theta_sin, theta_cos, state_b, time_b, done_b], dim=-1)  # [4, 4+5]

    data = HeteroData()
    # nodes
    data['agent'].pos = pos_a
    data['agent'].x   = x_agent
    data['scene'].pos = scene_pos
    if scene_feats is None:
        data['scene'].x = torch.zeros(n_s, 1, device=scene_pos.device, dtype=scene_pos.dtype)
    else:
        assert scene_feats.shape[0] == n_s
        data['scene'].x = scene_feats

    # edges: agent -> scene
    ei_as = _fully_connected_edges(n_a, n_s)
    delta_as = scene_pos[ei_as[1]] - pos_a[ei_as[0]]        # [E,2]
    eattr_as = fourier_embed_2d(delta_as, num_freqs=num_freqs)
    data[('agent', 'to', 'scene')].edge_index = ei_as
    data[('agent', 'to', 'scene')].edge_attr  = eattr_as

    # edges: scene -> agent
    ei_sa = _fully_connected_edges(n_s, n_a)
    delta_sa = pos_a[ei_sa[1]] - scene_pos[ei_sa[0]]
    eattr_sa = fourier_embed_2d(delta_sa, num_freqs=num_freqs)
    data[('scene', 'to', 'agent')].edge_index = ei_sa
    data[('scene', 'to', 'agent')].edge_attr  = eattr_sa

    # optional agent -> agent (no self loops)
    if include_agent_agent and n_a > 1:
        ei_aa = _complete_graph_edges(n_a)
        delta_aa = pos_a[ei_aa[1]] - pos_a[ei_aa[0]]
        eattr_aa = fourier_embed_2d(delta_aa, num_freqs=num_freqs)
        data[('agent', 'to', 'agent')].edge_index = ei_aa
        data[('agent', 'to', 'agent')].edge_attr  = eattr_aa

    return data

def build_local_heterodata_batch(
    agent_pos_b: Tensor,         # [B, 4, 6]
    scene_pos_b: Tensor,         # [B, M, 2]
    num_freqs: int = 10,
    include_agent_agent: bool = False,
    scene_feats_b: Optional[Tensor] = None,  # [B, M, d_s] or None
) -> HeteroBatch:
    assert agent_pos_b.ndim == 3 and agent_pos_b.size(1) == 4 and agent_pos_b.size(2) == 6
    assert scene_pos_b.ndim == 3 and scene_pos_b.size(2) == 2
    B, M, _ = scene_pos_b.shape

    data_list: List[HeteroData] = []
    for b in range(B):
        sf = None if scene_feats_b is None else scene_feats_b[b]
        d = build_local_heterodata_single(
            agent_pos=agent_pos_b[b],
            scene_pos=scene_pos_b[b],
            num_freqs=num_freqs,
            include_agent_agent=include_agent_agent,
            scene_feats=sf,
        )
        data_list.append(d)
    return HeteroBatch.from_data_list(data_list)
