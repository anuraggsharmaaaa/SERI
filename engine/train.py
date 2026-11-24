import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from engine.architecture import SERI_MultiModal_Engine


def resolve_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {candidates}")


def load_all_patches(num_nodes, risk_matrix, patch_dir="seri_patches"):
    """Loads all 64x64 patches into a single [N, 1, 64, 64] float32 tensor."""
    resolved_dir = patch_dir if os.path.isdir(patch_dir) else os.path.join("SERI_Core", patch_dir)
    patches = np.zeros((num_nodes, 1, 64, 64), dtype=np.float32)

    if os.path.isdir(resolved_dir):
        for idx in range(num_nodes):
            npy_path = os.path.join(resolved_dir, f"patch_{idx}.npy")
            if os.path.exists(npy_path):
                arr = np.load(npy_path).astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[0]
                if arr.shape != (64, 64):
                    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                    arr = (
                        torch.nn.functional.interpolate(
                            t, size=(64, 64), mode="bilinear", align_corners=False
                        )
                        .squeeze(0)
                        .squeeze(0)
                        .numpy()
                    )
                patches[idx, 0] = arr
                continue

            vals = risk_matrix[idx]
            grid = np.full((64, 64), float(np.mean(vals)), dtype=np.float32)
            for k, val in enumerate(vals[:4]):
                r0, r1 = (k // 2) * 32, ((k // 2) + 1) * 32
                c0, c1 = (k % 2) * 32, ((k % 2) + 1) * 32
                grid[r0:r1, c0:c1] = float(val)
            patches[idx, 0] = grid
    else:
        for idx in range(num_nodes):
            vals = risk_matrix[idx]
            grid = np.full((64, 64), float(np.mean(vals)), dtype=np.float32)
            for k, val in enumerate(vals[:4]):
                r0, r1 = (k // 2) * 32, ((k // 2) + 1) * 32
                c0, c1 = (k % 2) * 32, ((k % 2) + 1) * 32
                grid[r0:r1, c0:c1] = float(val)
            patches[idx, 0] = grid

    return torch.from_numpy(patches)


def make_spatial_splits(lats, lons, grid_bins=8, seed=42):
    """Splits nodes by spatial grid blocks to prevent nearby-point leakage."""
    lat_bins = np.clip(
        np.floor((lats - lats.min()) / (lats.max() - lats.min() + 1e-8) * grid_bins).astype(int),
        0,
        grid_bins - 1,
    )
    lon_bins = np.clip(
        np.floor((lons - lons.min()) / (lons.max() - lons.min() + 1e-8) * grid_bins).astype(int),
        0,
        grid_bins - 1,
    )
    block_ids = lat_bins * grid_bins + lon_bins
    unique_blocks = np.unique(block_ids)

    rng = np.random.default_rng(seed)
    shuffled_blocks = rng.permutation(unique_blocks)

    n_blocks = len(shuffled_blocks)
    n_train = max(1, int(0.7 * n_blocks))
    n_val = max(1, int(0.15 * n_blocks))

    train_blocks = set(shuffled_blocks[:n_train])
    val_blocks = set(shuffled_blocks[n_train : n_train + n_val])

    train_idx = np.array([i for i, b in enumerate(block_ids) if b in train_blocks], dtype=np.int64)
    val_idx = np.array([i for i, b in enumerate(block_ids) if b in val_blocks], dtype=np.int64)
    test_idx = np.array(
        [i for i, b in enumerate(block_ids) if b not in train_blocks and b not in val_blocks],
        dtype=np.int64,
    )
    if len(test_idx) == 0:
        test_idx = val_idx.copy()
    return train_idx, val_idx, test_idx


def eval_subset(model, indices, all_patches, x_graph, edge_index, y_targets, criterion, device, chunk_size=512):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for start in range(0, len(indices), chunk_size):
            sub = indices[start : start + chunk_size]
            sub_t = torch.tensor(sub, dtype=torch.long, device=device)
            imgs = all_patches[sub_t].to(device)
            logits = model(imgs, x_graph, edge_index, target_node_idx=sub_t)
            targets = y_targets[sub_t]
            loss = criterion(logits, targets)

            total_loss += loss.item() * len(sub)
            correct += (torch.argmax(logits, dim=1) == targets).sum().item()
            total += len(sub)

    return total_loss / max(total, 1), (correct / max(total, 1)) * 100.0


def train_seri(epochs=25, batch_size=128, steps_per_epoch=40, lr=5e-4, patch_dir="seri_patches"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training SERI GAT Engine on {device} (spatial block split + nightlight patches)...")

    graph_path = resolve_path("data/karnataka_graph.pt", "SERI_Core/data/karnataka_graph.pt")
    csv_path = resolve_path(
        "data/karnataka_master_data.csv",
        "SERI_Core/data/karnataka_master_data.csv",
        "karnataka_master_data.csv",
    )

    graph_data = torch.load(graph_path, map_location=device, weights_only=False)
    x_graph = graph_data.x.to(device)
    edge_index = graph_data.edge_index.to(device)
    num_nodes = graph_data.num_nodes

    df = pd.read_csv(csv_path)
    if len(df) != num_nodes:
        raise ValueError(f"Row/node mismatch: CSV has {len(df)} rows, graph has {num_nodes} nodes.")

    risk_cols = ["Logistics_Risk", "Cult_Risk", "Nature_Risk", "Econ_Risk"]
    risk_matrix = df[risk_cols].to_numpy(dtype=np.float32)
    lats = df["Latitude"].to_numpy(dtype=np.float64)
    lons = df["Longitude"].to_numpy(dtype=np.float64)

    target_scores = df["SERI_Target"].to_numpy(dtype=np.float32)
    q33, q66 = np.quantile(target_scores, [0.33, 0.66])
    labels = np.zeros(num_nodes, dtype=np.int64)
    labels[target_scores > q33] = 1
    labels[target_scores > q66] = 2
    y_targets = torch.tensor(labels, dtype=torch.long, device=device)

    all_patches = load_all_patches(num_nodes, risk_matrix, patch_dir=patch_dir)
    train_pool, val_pool, test_pool = make_spatial_splits(lats, lons, grid_bins=8, seed=42)
    print(f"Spatial split -> Train: {len(train_pool)} | Val: {len(val_pool)} | Test: {len(test_pool)}")

    model = SERI_MultiModal_Engine(
        in_channels=x_graph.shape[1], gat_heads=4, gat_dropout=0.2
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    save_dir = "engine" if os.path.isdir("engine") else "SERI_Core/engine"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "seri_gat_v1.pth")

    best_val_acc = -1.0
    rng = np.random.default_rng(42)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            b_idx = rng.choice(train_pool, size=min(batch_size, len(train_pool)), replace=False)
            b_idx_t = torch.tensor(b_idx, dtype=torch.long, device=device)
            batch_imgs = all_patches[b_idx_t].to(device)

            logits = model(batch_imgs, x_graph, edge_index, target_node_idx=b_idx_t)
            targets = y_targets[b_idx_t]
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(b_idx)
            epoch_correct += (torch.argmax(logits, dim=1) == targets).sum().item()
            epoch_total += len(b_idx)

        train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = (epoch_correct / max(epoch_total, 1)) * 100.0
        val_loss, val_acc = eval_subset(
            model, val_pool, all_patches, x_graph, edge_index, y_targets, criterion, device
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Full Val Loss: {val_loss:.4f} | Full Val Acc: {val_acc:.2f}%"
        )

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    test_loss, test_acc = eval_subset(
        model, test_pool, all_patches, x_graph, edge_index, y_targets, criterion, device
    )
    print(f"Saved best GAT weights to {save_path}")
    print(f"Final Full Val Acc: {best_val_acc:.2f}% | Held-Out Spatial Test Acc: {test_acc:.2f}% (Test Loss: {test_loss:.4f})")


if __name__ == "__main__":
    train_seri()