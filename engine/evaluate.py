import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
import scipy.ndimage
import os
from engine.architecture import SERI_MultiModal_Engine

def evaluate_seri_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Architecture & Weights
    model = SERI_MultiModal_Engine().to(device)
    model_path = 'engine/seri_multimodal_v1.pth'
    if not os.path.exists(model_path) and os.path.exists('SERI_Core/' + model_path):
        model_path = 'SERI_Core/' + model_path
        
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✅ Loaded weights from {model_path}")
    else:
        print(f"⚠️ Weights not found at {model_path}. Run training first.")
        return
        
    model.eval()

    # Load Graph Data
    graph_path = 'data/karnataka_graph.pt'
    if not os.path.exists(graph_path) and os.path.exists('SERI_Core/' + graph_path):
        graph_path = 'SERI_Core/' + graph_path
        
    graph_data = torch.load(graph_path, map_location=device, weights_only=False)
    x_graph = graph_data.x.to(device)
    edge_index = graph_data.edge_index.to(device)
    
    # Load CSV and use Quantiles for balanced tier distribution
    csv_path = 'data/karnataka_master_data.csv'
    if not os.path.exists(csv_path) and os.path.exists('SERI_Core/' + csv_path):
        csv_path = 'SERI_Core/' + csv_path
        
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        target_col = 'SERI_Target' if 'SERI_Target' in df.columns else df.columns[-1]
        scores = df[target_col].values
        
        scores_series = pd.Series(scores)
        q1 = scores_series.quantile(0.33)
        q2 = scores_series.quantile(0.66)
        
        y_true = np.zeros(len(scores), dtype=int)
        for i, score in enumerate(scores):
            if score <= q1:
                y_true[i] = 0 # Prime
            elif score <= q2:
                y_true[i] = 1 # Marginal
            else:
                y_true[i] = 2 # Borderline
    else:
        print("⚠️ Could not locate master CSV for ground-truth derivation.")
        return

    # Sample 1,000 nodes for a fast, clean audit test
    total_nodes = x_graph.size(0)
    sample_size = min(1000, total_nodes)
    np.random.seed(42)
    sampled_indices = np.random.choice(total_nodes, sample_size, replace=False)
    
    y_preds = []
    y_true_sampled = y_true[sampled_indices]
    
    print(f"Evaluating multi-modal fusion across {sample_size} nodes with live terrain patches...")
    
    with torch.no_grad():
        for idx in sampled_indices:
            node_idx = int(idx)
            
            # Generate deterministic terrain patch matching dashboard logic
            np.random.seed(node_idx)
            base_noise = np.random.normal(0, 1, (64, 64))
            terrain = scipy.ndimage.gaussian_filter(base_noise, sigma=1.2)
            normalized_patch = (terrain - terrain.min()) / (terrain.max() - terrain.min())
            
            # Format as tensor [1, 1, 64, 64]
            img_tensor = torch.tensor(normalized_patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            output = model(img_tensor, x_graph, edge_index, target_node_idx=node_idx)
            pred_idx = torch.argmax(output, dim=1).item()
            y_preds.append(pred_idx)

    acc = accuracy_score(y_true_sampled, y_preds)
    print(f"\n📊 Corrected Sampled Evaluation Results (N = {sample_size}):")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print("-" * 40)
    print(classification_report(y_true_sampled, y_preds, target_names=["Prime", "Marginal", "Borderline"], zero_division=0))

if __name__ == "__main__":
    evaluate_seri_model()