import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class SERI_MultiModal_Engine(nn.Module):
    def __init__(self):
        super(SERI_MultiModal_Engine, self).__init__()
        
        # BRANCH 1: The Vision Engine (2D-CNN for 64x64 Satellite Patches)
        self.vision_branch = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128) 
        )
        
        # BRANCH 2: The Graph Engine (GCN replacing static MLP vector branch)
        # Layer 1: Ingests 5-D Node Features -> 32-D spatial embedding
        self.gcn1 = GCNConv(in_channels=5, out_channels=32)
        self.relu_gcn = nn.ReLU()
        # Layer 2: Aggregates neighbor graph embeddings into a 64-D vector representation
        self.gcn2 = GCNConv(in_channels=32, out_channels=64)
        
        # THE LATE-FUSION LAYER (Fusing 128-D Vision + 64-D Graph Node Embedding)
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(64, 3) # Output Classes: Prime, Marginal, Borderline
        )

    def forward(self, image, x_graph, edge_index, target_node_idx=None):
        """
        image: [Batch, 1, 64, 64] Satellite Raster Patch
        x_graph: [N, 5] Node Feature Matrix across the Karnataka graph
        edge_index: [2, E] Adjacency Matrix Edge Index Tensor
        target_node_idx: target node index to extract embedded features for
        """
        # 1. Forward pass vision branch
        vision_feat = self.vision_branch(image) # [Batch, 128]
        
        # 2. Perform Message Passing over Graph Topology
        g_feat = self.gcn1(x_graph, edge_index)
        g_feat = self.relu_gcn(g_feat)
        g_feat = self.gcn2(g_feat, edge_index) # [N, 64]
        
        # 3. Extract embedding for target node(s)
        if target_node_idx is not None:
            if isinstance(target_node_idx, int):
                target_node_idx = torch.tensor([target_node_idx], device=x_graph.device)
            graph_feat = g_feat[target_node_idx] # [Batch, 64]
        else:
            graph_feat = g_feat[:vision_feat.size(0)]
            
        # 4. Concatenate multi-modal features via Late Fusion
        fused = torch.cat((vision_feat, graph_feat), dim=1)
        
        return self.fusion_layer(fused)