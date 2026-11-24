import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class SERI_MultiModal_Engine(nn.Module):
    def __init__(self, in_channels: int = 6, gat_heads: int = 4, gat_dropout: float = 0.2):
        super().__init__()

        # Branch 1: 2D-CNN for 64x64 patches -> [Batch, 128]
        self.vision_branch = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
        )

        # Branch 2: GATv2 graph encoder -> [N, 64]
        self.gat1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=32,
            heads=gat_heads,
            concat=False,
            dropout=gat_dropout,
            add_self_loops=True,
        )
        self.gat2 = GATv2Conv(
            in_channels=32,
            out_channels=64,
            heads=gat_heads,
            concat=False,
            dropout=gat_dropout,
            add_self_loops=True,
        )

        # Late fusion -> 3 classes: 0 (Prime), 1 (Marginal), 2 (Borderline)
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),
        )

    def forward(self, image, x_graph, edge_index, target_node_idx=None):
        vision_feat = self.vision_branch(image)

        g_feat = F.elu(self.gat1(x_graph, edge_index))
        g_feat = F.elu(self.gat2(g_feat, edge_index))

        if target_node_idx is not None:
            if isinstance(target_node_idx, int):
                target_node_idx = torch.tensor(
                    [target_node_idx], dtype=torch.long, device=x_graph.device
                )
            else:
                target_node_idx = target_node_idx.to(dtype=torch.long, device=x_graph.device)
            graph_feat = g_feat[target_node_idx]
        else:
            graph_feat = g_feat[: vision_feat.size(0)]

        fused = torch.cat((vision_feat, graph_feat), dim=1)
        return self.fusion_layer(fused)