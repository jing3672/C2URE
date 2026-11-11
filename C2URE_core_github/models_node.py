"""Two-layer GCN for node classification on a single (dense) graph."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(adj):
    a = adj + torch.eye(adj.shape[0], device=adj.device)
    deg = a.sum(1).clamp(min=1.0)
    dinv = deg.pow(-0.5)
    return a * dinv[:, None] * dinv[None, :]


class NodeGCN(nn.Module):
    def __init__(self, in_dim, hidden=32, num_classes=7, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Linear(in_dim, hidden)
        self.conv2 = nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        a = normalize_adj(adj)
        h = F.relu(self.conv1(x))
        recovery_mask = getattr(self, "_recovery_channel_mask", None)
        if recovery_mask is not None:
            h = h * recovery_mask.to(
                device=h.device, dtype=h.dtype).reshape(1, -1)
        h1 = torch.mm(a, h)
        h = F.dropout(h1, self.dropout, training=self.training)
        h = self.conv2(h)
        h2 = torch.mm(a, h)
        self._feats = [h1, h2]
        return h2


class NodeGraphSAGE(nn.Module):
    """Two-layer mean-aggregator GraphSAGE for node classification.

    The constructor and ``conv1``/``conv2`` attribute names deliberately
    match :class:`NodeGCN` so the existing federated training, aggregation,
    repair, and classifier-head utilities can use either backbone without a
    separate code path.
    """

    def __init__(self, in_dim, hidden=32, num_classes=7, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Linear(2 * in_dim, hidden)
        self.conv2 = nn.Linear(2 * hidden, num_classes)
        self.dropout = dropout

    @staticmethod
    def _mean_aggregate(x, adj):
        a = adj + torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
        return torch.mm(a, x) / a.sum(1, keepdim=True).clamp(min=1.0)

    def forward(self, x, adj):
        neigh = self._mean_aggregate(x, adj)
        h = F.relu(self.conv1(torch.cat([x, neigh], dim=1)))
        recovery_mask = getattr(self, "_recovery_channel_mask", None)
        if recovery_mask is not None:
            h = h * recovery_mask.to(
                device=h.device, dtype=h.dtype).reshape(1, -1)
        h1 = h
        h = F.dropout(h, self.dropout, training=self.training)
        neigh_h = self._mean_aggregate(h, adj)
        h2 = self.conv2(torch.cat([h, neigh_h], dim=1))
        self._feats = [h1, h2]
        return h2


class AuthorEnergyNodeGCN(nn.Module):
    """Dense equivalent of the public FedTD release's ``models/GCN.py``.

    This is deliberately a separate model from :class:`NodeGCN`.  The public
    FedTD code applies LayerNorm before its first GCN layer and after its
    hidden activation, and defines its client energy from the *pre-softmax*
    logits.  Reusing NodeGCN silently changes both of those mechanisms.
    """

    def __init__(self, in_dim, hidden=32, num_classes=7, dropout=0.5,
                 layer_norm_first=True, use_ln=True):
        super().__init__()
        self.layer_norm_first = bool(layer_norm_first)
        self.use_ln = bool(use_ln)
        self.input_ln = nn.LayerNorm(in_dim)
        self.conv1 = nn.Linear(in_dim, hidden)
        self.hidden_ln = nn.LayerNorm(hidden)
        self.conv2 = nn.Linear(hidden, num_classes)
        self.dropout = float(dropout)
        # Match the release's explicit Kaiming initialization of GCN weights.
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def _logits(self, x, adj):
        a = normalize_adj(adj)
        if self.layer_norm_first:
            x = self.input_ln(x)
        # PyG GCNConv applies the learnable linear map after normalized
        # propagation and then adds its bias.  Keeping the bias outside the
        # adjacency multiplication matters because normalized row sums are
        # not generally one.
        h = F.relu(self.conv1(torch.mm(a, x)))
        if self.use_ln:
            h = self.hidden_ln(h)
        h = F.dropout(h, self.dropout, training=self.training)
        logits = self.conv2(torch.mm(a, h))
        self._feats = [h, logits]
        return logits

    def forward(self, x, adj):
        # The release returns log_softmax here, while its local objective is
        # NLL.  The rest of this project uses cross entropy; CE(log-probs, y)
        # is exactly NLL(log-probs, y), so keeping log-probs preserves both
        # interfaces without changing the author model.
        return F.log_softmax(self._logits(x, adj), dim=1)

    def forward_energy(self, x, adj):
        """Author-release energy: logsumexp of pre-softmax logits."""
        return self._logits(x, adj).logsumexp(dim=1)


class CausalGateNodeGCN(NodeGCN):
    """Node GCN with a learned, target-free causal message gate.

    The gate is a property of the predictive mechanism rather than an
    evaluation-time trigger detector.  It scores every source node from its
    attributes and local structural context, then blocks messages through
    edges incident to nodes outside the learned causal subgraph.  Self
    information is always retained by ``normalize_adj``.
    """

    def __init__(self, in_dim, hidden=32, num_classes=7, dropout=0.5):
        super().__init__(in_dim, hidden=hidden, num_classes=num_classes,
                         dropout=dropout)
        self.feature_gate = nn.Linear(in_dim, 1)
        self.structure_gate = nn.Sequential(
            nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
        # Begin close to the ordinary GCN so the causal objective, rather than
        # random initialization, decides which messages should be blocked.
        nn.init.zeros_(self.feature_gate.weight)
        nn.init.constant_(self.feature_gate.bias, 2.944439)
        nn.init.zeros_(self.structure_gate[0].weight)
        nn.init.zeros_(self.structure_gate[0].bias)
        nn.init.zeros_(self.structure_gate[2].weight)
        nn.init.zeros_(self.structure_gate[2].bias)
        self._causal_gate = None

    def causal_gate(self, x, adj):
        degree = adj.sum(1)
        safe_degree = degree.clamp(min=1.0)
        degree_scale = degree / degree.mean().clamp(min=1.0)
        neighbor = torch.mm(adj, x) / safe_degree[:, None]
        residual = (x - neighbor).norm(dim=1)
        residual = residual / (
            x.norm(dim=1) + neighbor.norm(dim=1)).clamp(min=1e-6)
        feature_mass = x.norm(dim=1)
        feature_mass = feature_mass / feature_mass.mean().clamp(min=1e-6)
        structural = torch.stack(
            [degree_scale.clamp(max=10.0), residual.clamp(max=10.0),
             feature_mass.clamp(max=10.0)], dim=1)
        logits = self.feature_gate(x).squeeze(1)
        logits = logits + self.structure_gate(structural).squeeze(1)
        return torch.sigmoid(logits)

    def forward(self, x, adj):
        gate = self.causal_gate(x, adj)
        edge_gate = torch.sqrt(
            gate[:, None].clamp(min=1e-6)
            * gate[None, :].clamp(min=1e-6))
        gated_adj = adj * edge_gate
        a = normalize_adj(gated_adj)
        h = F.relu(self.conv1(x))
        h1 = torch.mm(a, h)
        h = F.dropout(h1, self.dropout, training=self.training)
        h = self.conv2(h)
        h2 = torch.mm(a, h)
        self._causal_gate = gate
        self._feats = [h1, h2]
        return h2
