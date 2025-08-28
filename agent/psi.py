import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, TransformerConv

class Psi(nn.Module):
    """
    ψ: Foresight to action nodes.
    Hetero types: 'curr', 'act'
    Rels: ('act','temporal','act'), ('curr','to','act'), ('act','to','curr')
    Output: updated 'act' features [B, T*A, D] (you'll reshape to [B,T,A,D])
    """
    def __init__(self, dim: int, num_layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        hid = dim
        self.in_proj = nn.ModuleDict({
            'curr': nn.Linear(dim, hid),
            'act':  nn.Linear(dim, hid),
        })
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            convs = {
                ('curr','to','act'): TransformerConv((hid, hid), hid // heads, heads=heads, edge_dim=0, dropout=dropout),
                ('act','to','curr'): TransformerConv((hid, hid), hid // heads, heads=heads, edge_dim=0, dropout=dropout),
                ('act','temporal','act'): TransformerConv((hid, hid), hid // heads, heads=heads, edge_dim=0, dropout=dropout),
            }
            self.layers.append(HeteroConv(convs, aggr='sum'))

        self.norm = nn.ModuleDict({'curr': nn.LayerNorm(hid), 'act': nn.LayerNorm(hid)})
        self.act = nn.SiLU()
        self.out_proj = nn.ModuleDict({'curr': nn.Linear(hid, dim), 'act': nn.Linear(hid, dim)})

    def forward(self, data):
        x = {
            'curr': self.act(self.in_proj['curr'](data['curr'].x)),
            'act':  self.act(self.in_proj['act'](data['act'].x)),
        }
        for layer in self.layers:
            x = layer(x, data.edge_index_dict)
            for t in x:
                x[t] = self.act(self.norm[t](x[t]))
        x['act'] = self.out_proj['act'](x['act'])
        return x['act']
