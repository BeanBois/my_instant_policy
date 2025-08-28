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

def build_context_heterodata_single(
    curr_agent_emb: Tensor,       # [A, D]
    curr_agent_info: Tensor,       # [A,6]
    demo_agent_emb: Tensor,       # [N, L, A, D]
    demo_agent_info: Tensor,       # [N,L,A,6]
    num_freqs: int = 10,
) -> HeteroData:
    """
    Nodes:
      - 'curr': A nodes with features = curr_agent_emb
      - 'demo': N*L*A nodes with features = flattened demo_agent_emb
    Edges:
      - ('demo','temporal','demo'): within each demo & keypoint, connect l -> l+1
      - ('demo','to','curr') and ('curr','to','demo'): full bipartite (structured cross-attn)
    """
    A, D = curr_agent_emb.shape
    N, L, A2, D2 = demo_agent_emb.shape
    assert A == A2 and D == D2

    data = HeteroData()
    # node features
    curr_agent_pos = curr_agent_info[..., :2]
    demo_agent_pos = demo_agent_info[..., :2]

    data['curr'].x = curr_agent_emb          # [A, D]
    data['curr'].pos = curr_agent_pos.float()          # [A, 2]
    demo_flat = demo_agent_emb.reshape(N * L * A, D)
    data['demo'].x = demo_flat               # [N*L*A, D]
    demo_pos_flat = demo_agent_pos.view(N*L*A,2)
    data['demo'].pos = demo_pos_flat.float()               # [N*L*A, 2]
    

    # helper to index (n,l,a) in flattened 'demo'
    def demo_idx(n, l, a):  # 0<=n<N, 0<=l<L, 0<=a<A
        return (n * L + l) * A + a

    # temporal edges in demo: (n, l, a) -> (n, l+1, a)
    if L > 1:
        src_t, dst_t = [], []
        for n in range(N):
            for a in range(A):
                for l in range(L - 1):
                    src_t.append(demo_idx(n, l, a))
                    dst_t.append(demo_idx(n, l + 1, a))
        ei_demo_temporal = torch.tensor([src_t, dst_t], dtype=torch.long)
        data[('demo', 'temporal', 'demo')].edge_index = ei_demo_temporal
        dp = data['demo'].pos
        rel = dp[ei_demo_temporal[1] - ei_demo_temporal[0]]
        data[('demo', 'temporal', 'demo')].edge_attr = fourier_embed_2d(rel, num_freqs=num_freqs)

    # demo <-> curr: full bipartite
    Nd = N * L * A
    src_demo = torch.arange(Nd).repeat_interleave(A)
    dst_curr = torch.arange(A).repeat(Nd)
    ei_demo_to_curr = torch.stack([src_demo, dst_curr], dim=0)
    data[('demo', 'to', 'curr')].edge_index = ei_demo_to_curr
    # rel = data['curr'].pos[ei_demo_to_curr[1]] - data['demo'].pos[ei_demo_to_curr[0]]
    # data[('demo','to','curr')].edge_attr = fourier_embed_2d(rel, num_freqs=num_freqs)

    src_curr = torch.arange(A).repeat_interleave(Nd)
    dst_demo = torch.arange(Nd).repeat(A)
    ei_curr_to_demo = torch.stack([src_curr, dst_demo], dim=0)
    data[('curr', 'to', 'demo')].edge_index = ei_curr_to_demo
    # rel = data['demo'].pos[ei_curr_to_demo[1]] - data['curr'].pos[ei_curr_to_demo[0]]
    # data[('curr','to','demo')].edge_attr = fourier_embed_2d(rel, num_freqs=num_freqs)

    return data

def build_context_graph_batch(
    curr_agent_emb: Tensor,       # [A, D]
    curr_agent_info: Tensor,       # [A,6]
    demo_agent_emb: Tensor,       # [N, L, A, D]
    demo_agent_info: Tensor,       # [N,L,A,6]
    num_freqs: int = 10,
) -> HeteroBatch:
    B, A, D = curr_agent_emb.shape
    Bb, N, L, Aa, Dd = demo_agent_emb.shape
    assert B == Bb and A == Aa and D == Dd

    data_list: List[HeteroData] = []
    for b in range(B):
        d = build_context_heterodata_single(
            curr_agent_emb[b],       # [A, D]
            curr_agent_info[b],
            demo_agent_emb[b],       # [N, L, A, D]
            demo_agent_info[b],
            num_freqs
        )
        data_list.append(d)
    return HeteroBatch.from_data_list(data_list)

def build_action_heterodata_single(
    curr_agent_ctx: Tensor,   # [A, D] (ϕ-aligned current)
    curr_agent_info: Tensor,       # [A,6]
    pred_agent_emb: Tensor,   # [T, A, D] (ρ over predicted obs/agents per future step)
    pred_agent_info: Tensor,       # [T,A,6]
    num_freqs: int = 10,
) -> HeteroData:
    """
    Nodes:
      - 'curr': A nodes with context-aligned features
      - 'act':  T*A nodes (concatenate over T) with per-time gripper-node embeddings
    Edges:
      - ('act','temporal','act'): for each keypoint a, connect t -> t+1
      - ('curr','to','act') and ('act','to','curr'): full bipartite
    """
    A, D = curr_agent_ctx.shape
    T, A2, D2 = pred_agent_emb.shape
    assert A == A2 and D == D2

    curr_agent_pos = curr_agent_info[...,:2]
    pred_agent_pos = pred_agent_info[...,:2]

    data = HeteroData()
    data['curr'].x = curr_agent_ctx                        # [A, D]
    data['curr'].pos = curr_agent_pos.float()                        # [A, D]

    act_flat = pred_agent_emb.reshape(T * A, D)
    data['act'].x = act_flat                               # [T*A, D]
    pred_agent_pos_flat = pred_agent_pos.reshape(T*A, 2)
    data['act'].x = pred_agent_pos_flat                               # [T*A, 2]


    def act_idx(t, a):  # 0<=t<T, 0<=a<A
        return t * A + a

    if T > 1:
        src_t, dst_t = [], []
        for a in range(A):
            for t in range(T - 1):
                src_t.append(act_idx(t, a))
                dst_t.append(act_idx(t + 1, a))
        ei_act_temporal = torch.tensor([src_t, dst_t], dtype=torch.long)
        data[('act', 'temporal', 'act')].edge_index = ei_act_temporal
        rel = data['act'].pos[ei_act_temporal[1]] - data['act'].pos[ei_act_temporal[0]]
        data[('act','temporal','act')].edge_attr = fourier_embed_2d(rel, num_freqs=num_freqs)


    Na = T * A
    # curr <-> act full bipartite
    src_curr = torch.arange(A).repeat_interleave(Na)
    dst_act  = torch.arange(Na).repeat(A)
    ei = torch.stack([src_curr, dst_act], dim=0)
    data[('curr', 'to', 'act')].edge_index = ei
    rel = data['act'].pos[ei[1]] - data['act'].pos[ei[0]]
    data[('act','temporal','act')].edge_attr = fourier_embed_2d(rel, num_freqs=num_freqs)

    src_act = torch.arange(Na).repeat_interleave(A)
    dst_curr = torch.arange(A).repeat(Na)
    ei = torch.stack([src_act, dst_curr], dim=0)
    data[('act', 'to', 'curr')].edge_index = ei
    rel = data['act'].pos[ei[1]] - data['act'].pos[ei[0]]
    data[('act','temporal','act')].edge_attr = fourier_embed_2d(rel, num_freqs=num_freqs)
    return data

def build_action_graph_batch(
    curr_agent_ctx: Tensor,   # [B, A, D]
    curr_agent_info: Tensor,    # [B,A,6]
    pred_agent_emb: Tensor,   # [B, T, A, D]
    pred_agent_info: Tensor,   # [B, T, A, 2]
    num_freqs: int = 10,
) -> HeteroBatch:
    B, A, D = curr_agent_ctx.shape
    Bb, T, Aa, Dd = pred_agent_emb.shape
    assert B == Bb and A == Aa and D == Dd

    data_list: List[HeteroData] = []
    for b in range(B):
        d = build_action_heterodata_single(
                            curr_agent_ctx[b], 
                            curr_agent_info[b],
                            pred_agent_emb[b],
                            pred_agent_info[b],
                            num_freqs)
        data_list.append(d)
    return HeteroBatch.from_data_list(data_list)