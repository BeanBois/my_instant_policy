import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, TransformerConv

class Phi(nn.Module):
    """
    ϕ: Context aligner.
    Hetero types: 'demo', 'curr'
    Rels: ('demo','temporal','demo'), ('demo','to','curr'), ('curr','to','demo')
    Output: updated 'curr' features [B, A, D]
    """
    def __init__(self, dim: int, num_layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        hid = dim
        self.in_proj = nn.ModuleDict({
            'demo': nn.Linear(dim, hid),
            'curr': nn.Linear(dim, hid),
        })
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            convs = {
                ('demo','to','curr'): TransformerConv((hid, hid), hid // heads, heads=heads, edge_dim=0, dropout=dropout),
                ('curr','to','demo'): TransformerConv((hid, hid), hid // heads, heads=heads, edge_dim=0, dropout=dropout),
            }
            # temporal demo edges (optional if L>1)
            convs[('demo','temporal','demo')] = TransformerConv((hid, hid), hid // heads, heads=heads, edge_dim=0, dropout=dropout)
            self.layers.append(HeteroConv(convs, aggr='sum'))

        self.norm = nn.ModuleDict({'demo': nn.LayerNorm(hid), 'curr': nn.LayerNorm(hid)})
        self.act = nn.SiLU()
        self.out_proj = nn.ModuleDict({'demo': nn.Linear(hid, dim), 'curr': nn.Linear(hid, dim)})

    def forward(self, data):
        x = {
            'demo': self.act(self.in_proj['demo'](data['demo'].x)),
            'curr': self.act(self.in_proj['curr'](data['curr'].x)),
        }
        for layer in self.layers:
            x = layer(x, data.edge_index_dict)
            for t in x:
                x[t] = self.act(self.norm[t](x[t]))
        # project out
        x['curr'] = self.out_proj['curr'](x['curr'])
        return x['curr']  # [sum_B A, D] but PyG batch keeps .batch to recover per-graph

