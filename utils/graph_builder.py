import os
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data


def resolve_csv_path(explicit_path=None):
    candidates = [
        explicit_path,
        "data/karnataka_master_data.csv",
        "SERI_Core/data/karnataka_master_data.csv",
        "karnataka_master_data.csv",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError("Could not locate karnataka_master_data.csv.")


def build_karnataka_spatial_graph(
    csv_path=None,
    output_path=None,
    max_coord_dist=0.15,
    k_neighbors=6,
):
    resolved_csv = resolve_csv_path(csv_path)
    df = pd.read_csv(resolved_csv)

    risk_cols = ["Logistics_Risk", "Cult_Risk", "Nature_Risk", "Econ_Risk"]
    missing = [c for c in ["Latitude", "Longitude"] + risk_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    lat = df["Latitude"].to_numpy(dtype=np.float32)
    lon = df["Longitude"].to_numpy(dtype=np.float32)

    lat_norm = (lat - lat.min()) / (lat.max() - lat.min() + 1e-8)
    lon_norm = (lon - lon.min()) / (lon.max() - lon.min() + 1e-8)

    # 6 genuine features per node (zero synthetic random noise)
    x_np = np.column_stack([df[risk_cols].to_numpy(dtype=np.float32), lat_norm, lon_norm])
    x_tensor = torch.from_numpy(x_np.astype(np.float32))

    coords = df[["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
    tree = cKDTree(coords)
    dists, indices = tree.query(coords, k=k_neighbors + 1)

    edge_list = []
    for i in range(len(coords)):
        for j_idx, dist in zip(indices[i][1:], dists[i][1:]):
            if dist <= max_coord_dist:
                j = int(j_idx)
                edge_list.append([i, j])
                edge_list.append([j, i])

    edge_index = np.unique(np.array(edge_list, dtype=np.int64), axis=0).T
    edge_index_tensor = torch.from_numpy(edge_index)

    graph_data = Data(x=x_tensor, edge_index=edge_index_tensor)

    if output_path is None:
        base_dir = "data" if os.path.isdir("data") else "SERI_Core/data"
        output_path = os.path.join(base_dir, "karnataka_graph.pt")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(graph_data, output_path)

    print(
        f"Saved graph to {output_path} | Nodes: {graph_data.num_nodes} | "
        f"Edges: {graph_data.num_edges} | Features: {graph_data.num_node_features}"
    )
    return graph_data


if __name__ == "__main__":
    build_karnataka_spatial_graph()