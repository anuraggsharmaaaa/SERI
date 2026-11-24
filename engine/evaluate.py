import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from engine.architecture import SERI_MultiModal_Engine


def get_patch_tensor(idx, risk_vals, patch_dir="seri_patches"):
    npy_path = os.path.join(patch_dir, f"patch_{idx}.npy")
    if os.path.exists(npy_path):
        arr = np.load(npy_path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        t = torch.from_numpy(arr).unsqueeze(0)
        if t.shape[-2:] != (64, 64):
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False
            ).squeeze(0)
        return t

    # Deterministic non-leaky patch matching train.py
    grid = np.full((1, 64, 64), float(np.mean(risk_vals)), dtype=np.float32)
    for k, val in enumerate(risk_vals[:4]):
        r0, r1 = (k // 2) * 32, ((k // 2) + 1) * 32
        c0, c1 = (k % 2) * 32, ((k % 2) + 1) * 32
        grid[0, r0:r1, c0:c1] = float(val)
    return torch.from_numpy(grid)


def evaluate_seri_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Graph Data first to verify feature dimension
    graph_path = "data/karnataka_graph.pt"
    if not os.path.exists(graph_path) and os.path.exists("SERI_Core/" + graph_path):
        graph_path = "SERI_Core/" + graph_path

    graph_data = torch.load(graph_path, map_location=device, weights_only=False)
    x_graph = graph_data.x.to(device)
    edge_index = graph_data.edge_index.to(device)

    # Load Architecture & Weights
    model = SERI_MultiModal_Engine(
        in_channels=x_graph.shape[1], gat_heads=4, gat_dropout=0.2
    ).to(device)

    model_path = "engine/seri_gat_v1.pth"
    if not os.path.exists(model_path) and os.path.exists("SERI_Core/" + model_path):
        model_path = "SERI_Core/" + model_path
    if not os.path.exists(model_path):
        fallback = "engine/seri_multimodal_v1.pth"
        model_path = "SERI_Core/" + fallback if os.path.exists("SERI_Core/" + fallback) else fallback

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✅ Loaded weights from {model_path}")
    else:
        print(f"⚠️ Weights not found at {model_path}. Run training first.")
        return

    model.eval()

    # Load CSV and use Quantiles for balanced tier distribution
    csv_path = "data/karnataka_master_data.csv"
    if not os.path.exists(csv_path) and os.path.exists("SERI_Core/" + csv_path):
        csv_path = "SERI_Core/" + csv_path

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        risk_cols = ["Logistics_Risk", "Cult_Risk", "Nature_Risk", "Econ_Risk"]
        risk_matrix = df[risk_cols].to_numpy(dtype=np.float32)

        target_col = "SERI_Target" if "SERI_Target" in df.columns else df.columns[-1]
        scores = df[target_col].values

        q1, q2 = np.quantile(scores, [0.33, 0.66])
        y_true = np.zeros(len(scores), dtype=int)
        y_true[scores > q1] = 1  # Marginal
        y_true[scores > q2] = 2  # Borderline
    else:
        print("⚠️ Could not locate master CSV for ground-truth derivation.")
        return

    # Sample 1,000 nodes for a fast, clean audit test
    total_nodes = x_graph.size(0)
    sample_size = min(1000, total_nodes)
    rng = np.random.default_rng(42)
    sampled_indices = rng.choice(total_nodes, sample_size, replace=False)

    y_preds = []
    y_true_sampled = y_true[sampled_indices]

    print(f"Evaluating GAT multi-modal fusion across {sample_size} nodes...")

    with torch.no_grad():
        for idx in sampled_indices:
            node_idx = int(idx)
            img_tensor = get_patch_tensor(node_idx, risk_matrix[node_idx]).unsqueeze(0).to(device)
            output = model(img_tensor, x_graph, edge_index, target_node_idx=node_idx)
            pred_idx = torch.argmax(output, dim=1).item()
            y_preds.append(pred_idx)

    acc = accuracy_score(y_true_sampled, y_preds)
    print(f"\n📊 Sampled GAT Evaluation Results (N = {sample_size}):")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print("-" * 40)
    print(classification_report(y_true_sampled, y_preds, target_names=["Prime", "Marginal", "Borderline"], zero_division=0))


if __name__ == "__main__":
    evaluate_seri_model()