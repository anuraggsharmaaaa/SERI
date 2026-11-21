# utils/graph_builder.py
import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import os

def build_karnataka_spatial_graph(csv_path='SERI_Core/data/karnataka_master_data.csv', 
                                  output_path='SERI_Core/data/karnataka_graph.pt', 
                                  max_distance_km=0.15, 
                                  k_neighbors=6):
    """
    Constructs a PyTorch Geometric Graph Data object from tabular coordinate data.
    - Nodes (V): Each coordinate in Karnataka.
    - Features (X): [Logistics_Risk, Cult_Risk, Nature_Risk, Econ_Risk, Engine1_Score]
    - Edges (E): Connects spatial neighbors to build the adjacency matrix for Message Passing.
    """
    if not os.path.exists(csv_path):
        # Fallback check for alternate data paths
        if os.path.exists('SERI_Bank/karnataka_master_data.csv'):
            csv_path = 'SERI_Bank/karnataka_master_data.csv'
        elif os.path.exists('data/karnataka_master_data.csv'):
            csv_path = 'data/karnataka_master_data.csv'
        else:
            raise FileNotFoundError(f"❌ Could not locate karnataka_master_data.csv in SERI_Core/data/ or SERI_Bank/")
        
    print(f"📖 Loading master grid coordinates from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Ensure 5th Vector (Engine 1 Mineral Score) exists or initialize baseline
    if 'Engine1_Score' not in df.columns:
        print("⚠️ Engine1_Score column not found. Injecting baseline spatial weights...")
        np.random.seed(42)
        df['Engine1_Score'] = np.random.uniform(0.1, 0.9, len(df))
    
    # 2. Extract 5 Node Features matching your exact CSV column names
    feature_cols = ['Logistics_Risk', 'Cult_Risk', 'Nature_Risk', 'Econ_Risk', 'Engine1_Score']
    
    missing_cols = [c for c in feature_cols if c not in df.columns and c != 'Engine1_Score']
    if missing_cols:
        raise ValueError(f"❌ Missing expected columns in CSV: {missing_cols}. Found columns: {list(df.columns)}")
        
    x_features = df[feature_cols].values
    x_tensor = torch.tensor(x_features, dtype=torch.float32)

    # 3. Build Spatial Adjacency Matrix using cKDTree
    print("🕸️ Building spatial adjacency matrix (Euclidean graph connectivity)...")
    coords = df[['Latitude', 'Longitude']].values
    tree = cKDTree(coords)
    
    edge_list = []
    # Query k-nearest spatial neighbors for every node
    distances, indices = tree.query(coords, k=k_neighbors+1) # +1 includes self
    
    for i in range(len(coords)):
        for j_idx, dist in zip(indices[i][1:], distances[i][1:]): # skip self-loop
            if dist <= max_distance_km:
                edge_list.append([i, j_idx]) # Direction i -> j
                edge_list.append([j_idx, i]) # Undirected bidirectional connection

    # Remove duplicates and convert to PyG Edge Index format [2, E]
    edge_index = np.unique(np.array(edge_list), axis=0).T
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

    # 4. Encapsulate into PyTorch Geometric Data Structure
    graph_data = Data(x=x_tensor, edge_index=edge_index_tensor)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(graph_data, output_path)
    
    print(f"✅ Graph Construction Complete!")
    print(f"   • Nodes (V): {graph_data.num_nodes}")
    print(f"   • Edges (E): {graph_data.num_edges}")
    print(f"   • Feature Vector Dim: {graph_data.num_node_features}")
    print(f"💾 Serialized to {output_path}")

if __name__ == "__main__":
    build_karnataka_spatial_graph()