import math
from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Hyperbolic utilities (Poincaré ball, curvature c > 0) ----------
def artanh(x, eps=1e-15):
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def poincare_exp0(v, c):
    # exp^c_0(v) = tanh( sqrt(c)*||v|| ) / ( sqrt(c)*||v|| ) * v
    norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(1e-9)
    factor = torch.tanh(torch.sqrt(c) * norm) / (torch.sqrt(c) * norm)
    return factor * v

def poincare_log0(x, c):
    # log^c_0(x) = (1/sqrt(c)) * arctanh( sqrt(c)*||x|| ) * x / ||x||
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-9)
    factor = artanh(torch.sqrt(c) * norm) / (torch.sqrt(c) * norm)
    return factor * x

def mobius_add(x, y, c):
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c**2) * x2 * y2
    return num / den.clamp_min(1e-9)

def poincare_dist(x, y, c, eps=1e-9):
    # d_B^c(x,y) = (2/sqrt(c)) * artanh( sqrt(c) * ||(-x) ⊕_c y|| )
    diff = mobius_add(-x, y, c)
    norm = torch.norm(diff, dim=-1).clamp_min(eps)
    return (2.0 / torch.sqrt(c)) * artanh(torch.sqrt(c) * norm)

# ---------- Tree builder via temporal clustering ----------
def _ang_diff(a: float, b: float) -> float:
    # shortest signed difference on the circle
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

import torch
from typing import List, Tuple

# ---------- angle utilities ----------
def _cluster(
        cluster_idxs,
        theta,
        state,
        gran
):
    earliest_index = cluster_idxs.pop(0)
    clusters = []
    children = []
    for idx in cluster_idxs:
        # if state[earliest_index] != state[idx] or \
        if abs(math.radians(theta[earliest_index] - theta[idx])) > gran:
            clusters.append((earliest_index, children))
            earliest_index = idx
            children = []
            continue
        children.append(idx)
    
    clusters.append((earliest_index, children))
    return clusters

def build_temporal_tree_multigran_K(
    theta: torch.Tensor,            # [T] angles (deg or rad)
    state: torch.Tensor,            # [T] discrete (e.g., 0/1)
    grans: List[float],             # e.g., [g1, g2, g3, g4]  (K levels)
    use_degrees: bool = True,       # True => theta in degrees; False => radians
) -> Tuple[List[int], List[List[int]]]:
    """
    Returns:
      parent:  length (T+1), indices in [0..T], where 0 is the root. root's parent = -1
      children: length (T+1) list of lists
    Desired structure from your example:
        root(0) -> {1, 7}
        1 -> {2, 4}, 2 -> {3}, 4 -> {5, 6}
        7 -> {8, 10}, 8 -> {9}, 10 -> {11, 12}
    """
    assert theta.ndim == 1 and state.ndim == 1 and theta.shape[0] == state.shape[0]
    T = theta.shape[0]
    N = T + 1  # include root at index 0; items are 1..T


    children_of = [-1 for _ in range(T)]
    queue_clusters = [(-1 , [i for i in range(T)])]

    for gran in grans:
        new_cluster = []
        while len(queue_clusters) > 0:
            parent, cluster = queue_clusters.pop(0)
            if len(cluster) <= 0:
                continue 
            _new_cluster = _cluster(cluster, # indexes
                                   theta,
                                   state,
                                   gran,
                                   )
            new_cluster = new_cluster + _new_cluster
        
        for cluster in new_cluster:
            parent, children = cluster
            for child in children:
                children_of[child] = parent
        queue_clusters = new_cluster


    # help me use children_of to remake parent and children so that i can link back to the api
    parent = [-1] * T
    children = [[] for _ in range(T)]
    ROOT = 0
    parent[ROOT] = -1

    for c, p in enumerate(children_of):
        if p != -1:
            children[p].append(c)
        parent[c] = p

    if T == 0:
        return parent, children
    return parent, children
        



# ---------- SK-style hyperbolic constructor in 2D ----------
class SKConstructor2D(nn.Module):
    """
    Minimal SK-like constructor:
      - Assign disjoint cones around each parent
      - Place children radially using target hyperbolic distances
      - Output: 2D Poincaré embeddings for all nodes (frames)
    """
    def __init__(self, curvature: float = 1.0, base_cone_deg: float = 30.0, min_edge_dist: float = 0.5):
        super().__init__()
        self.register_buffer("c", torch.tensor(float(curvature)))
        self.base_cone = math.radians(base_cone_deg)
        self.min_edge_dist = min_edge_dist  # hyperbolic distance floor to keep Voronoi edges stable

    def forward(self, children: List[List[int]], parent: List[int], L: int) -> torch.Tensor:
        device = self.c.device
        emb = torch.zeros(L, 2, device=device)  # start at origin for frame 0
        # BFS over the tree(s); each parent allocates disjoint angular cones to its children
        # Root angle defaults to 0; keep an angle registry for nodes.
        node_angle = torch.zeros(L, device=device)
        node_radius = torch.zeros(L, device=device)  # hyperbolic radius from origin (temporal depth proxy)

        # Precompute per-parent cone splits
        for p in range(L):
            ch = children[p]
            if len(ch) == 0:
                continue
            # distribute within a fan around parent direction:
            # center at node_angle[p]; spread = len(ch) * base_cone
            total_span = max(self.base_cone * len(ch), self.base_cone)
            start = node_angle[p] - total_span / 2.0
            for i, kid in enumerate(ch):
                node_angle[kid] = start + (i + 0.5) * (total_span / len(ch))
                # Hyperbolic radial increment for edge length; ensure >= min_edge_dist
                node_radius[kid] = node_radius[p] + self.min_edge_dist

        # Convert polar (node_radius, node_angle) to a 2D vector via tangent->ball exp map:
        # Build tangent vector with Euclidean norm = node_radius (so exp moves that far hyperbolically).
        e_x = torch.stack([torch.cos(node_angle), torch.sin(node_angle)], dim=-1)  # unit direction
        v = e_x * node_radius.unsqueeze(-1)  # in tangent space
        emb = poincare_exp0(v, self.c)
        return emb  # [L,2]

# ---------- Inter-/Intra-level hyperbolic message parsing (Paper 2 style) ----------
class HyperbolicTemporalMixer(nn.Module):
    """
    Inter-level (radial scaling) + Intra-level (rotation) in 2D hyperbolic space.
    Depth is defined by shortest-parent-chain length.
    """
    def __init__(self, curvature: float = 1.0):
        super().__init__()
        self.register_buffer("c", torch.tensor(float(curvature)))
        # Inter-level: scale per depth (learned)
        self.max_depth = 512  # cap; dynamic masked
        self.depth_scale = nn.Embedding(self.max_depth, 1)
        nn.init.ones_(self.depth_scale.weight)
        # Intra-level: rotation per (depth) (learned)
        self.depth_theta = nn.Embedding(self.max_depth, 1)
        nn.init.zeros_(self.depth_theta.weight)

    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        x: [L,2] Poincaré embeddings
        depth: [L] integer depth per node
        """
        d = depth.clamp(max=self.max_depth - 1)
        # Move to tangent @0
        v = poincare_log0(x, self.c)  # [L,2]

        # Inter-level radial scaling (level-aware): v' = k_d * v
        k = self.depth_scale(d).view(-1, 1)  # [L,1]
        v = k * v

        # Intra-level rotation (Givens in 2D)
        ang = self.depth_theta(d).view(-1)
        cos_a, sin_a = torch.cos(ang), torch.sin(ang)
        R = torch.stack([torch.stack([cos_a, -sin_a], dim=-1),
                         torch.stack([sin_a,  cos_a], dim=-1)], dim=-2)  # [L,2,2]
        v = torch.einsum('lij,lj->li', R, v)

        # Back to hyperbolic
        return poincare_exp0(v, self.c)

# ---------- Full demo handler ----------
class DemoHandler(nn.Module):
    """
    Input: demo_agent_info [B, N, L, A, 6]  (x,y,theta,state,time,done)
    Output: embeddings [B, N, L, 2]
    Strategy:
      1) For each (b,n), read agent 0's theta/state over time.
      2) Build temporal tree (clustering rules).
      3) SK-style embed the tree in Poincaré(2D).
      4) Inter/Intra temporal mixing for neighborhood awareness.
    """
    def __init__(self,
                 curvature: float = 1.0,
                 angular_granularities_deg: float = [60.0,45.0,30.0,15.0],
                 base_cone_deg: float = 30.0,
                 min_edge_dist: float = 0.5):
        super().__init__()
        self.c = curvature
        self.ang_grans = [math.radians(g) for g in angular_granularities_deg]
        self.sk = SKConstructor2D(curvature=curvature,
                                  base_cone_deg=base_cone_deg,
                                  min_edge_dist=min_edge_dist)
        self.mixer = HyperbolicTemporalMixer(curvature=curvature)

    @staticmethod
    def _compute_depth(parent: List[int]) -> torch.Tensor:
        L = len(parent)
        depth = torch.zeros(L, dtype=torch.long)
        for i in range(L):
            d, p = 0, parent[i]
            while p != -1 and p is not None:
                d += 1
                p = parent[p]
            depth[i] = d
        return depth

    def forward(self, demo_agent_info: torch.Tensor) -> torch.Tensor:
        """
        demo_agent_info: [B, N, L, A, 6]  -> returns [B, N, L, 2]
        We use agent index 0 as the "proceeding" agent.
        """
        B, N, L, A, D = demo_agent_info.shape
        device = demo_agent_info.device
        assert D >= 6, "last dim must contain at least (x,y,theta,state,time,done)"

        # Slice agent 0's theta & state per (B,N)
        theta = demo_agent_info[..., 0, 2]          # [B,N,L]
        state = demo_agent_info[..., 0, 3]          # [B,N,L]

        out = torch.zeros(B, N, L, 2, device=device)

        for b in range(B):
            for n in range(N):
                th_seq = theta[b, n]  # [L]
                st_seq = state[b, n]  # [L]

                # Build temporal tree by clustering
                parent, children = build_temporal_tree_multigran_K(th_seq, st_seq, self.ang_grans)
                depth = self._compute_depth(parent).to(device)  # [L]

                # SK constructor -> initial 2D embeddings
                emb = self.sk(children, parent, L)  # [L,2]

                # Inter-/Intra-level mixing (temporal message parsing)
                emb = self.mixer(emb, depth)       # [L,2]

                out[b, n] = emb
        return out  # [B,N,L,2]
