"""Small dense CPU encoders; one shared symmetric pair-decoder architecture."""
import torch
from torch import nn


def operators(features, edges):
    n = features.shape[0]
    adjacency = features.new_zeros((n, n))
    if edges:
        index = torch.tensor(edges, dtype=torch.long)
        adjacency[index[:, 0], index[:, 1]] = 1
    mean = adjacency / adjacency.sum(1, keepdim=True).clamp(min=1)
    with_self = adjacency + torch.eye(n, dtype=features.dtype)
    inv = with_self.sum(1).rsqrt()
    normalized = inv[:, None] * with_self * inv[None, :]
    return normalized, mean


class PairModel(nn.Module):
    def __init__(self, kind, input_dim, hidden):
        super().__init__()
        if kind not in {"mlp", "gcn", "sage", "ego_gcn"}:
            raise ValueError(f"Unknown encoder: {kind}")
        self.kind = kind
        self.encoder = nn.Linear(input_dim * (2 if kind in {"sage", "ego_gcn"} else 1), hidden)
        self.decoder = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, features, normalized, mean, pairs):
        if self.kind == "mlp":
            inputs = features
        elif self.kind == "gcn":
            inputs = normalized @ features
        elif self.kind == "sage":
            inputs = torch.cat((features, mean @ features), dim=1)
        else:
            # GCN aggregation is kept fixed; an explicit raw ego path is added.
            inputs = torch.cat((features, normalized @ features), dim=1)
        h = torch.relu(self.encoder(inputs))
        a, b = h[pairs[:, 0]], h[pairs[:, 1]]
        symmetric = torch.cat((a + b, (a - b).abs(), a * b), dim=1)
        return self.decoder(symmetric).squeeze(-1)
