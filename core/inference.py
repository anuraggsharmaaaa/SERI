import os
import numpy as np
import torch
import torch.nn.functional as F
from engine.architecture import SERI_MultiModal_Engine


class SERI_Predictor:
    def __init__(
        self,
        model_path="engine/seri_gat_v1.pth",
        graph_path="data/karnataka_graph.pt",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Spatial Graph first so in_channels matches graph node features
        if not os.path.exists(graph_path) and os.path.exists("SERI_Core/" + graph_path):
            graph_path = "SERI_Core/" + graph_path

        self.graph_data = torch.load(graph_path, map_location=self.device, weights_only=False)
        in_channels = int(self.graph_data.x.shape[1])

        self.model = SERI_MultiModal_Engine(
            in_channels=in_channels, gat_heads=4, gat_dropout=0.2
        ).to(self.device)

        # Load Model Weights
        if not os.path.exists(model_path) and os.path.exists("SERI_Core/" + model_path):
            model_path = "SERI_Core/" + model_path
        if not os.path.exists(model_path):
            fallback = "engine/seri_multimodal_v1.pth"
            if os.path.exists(fallback):
                model_path = fallback
            elif os.path.exists("SERI_Core/" + fallback):
                model_path = "SERI_Core/" + fallback

        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device, weights_only=True)
                )
                print(f"✅ SERI GAT Weights loaded successfully from {model_path}")
            except Exception as e:
                print(f"⚠️ Weight loading error: {e}")
        else:
            print(f"⚠️ Warning: Could not find weights at {model_path}. Run 'python3 -m engine.train' first.")

        self.model.eval()

        # Grad-CAM Hooks State Setup
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.vision_branch[3]

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def predict(self, image_patch, node_idx=0, override_vectors=None):
        self.model.zero_grad()

        # Format Patch Tensor: [1, 1, 64, 64]
        img_tensor = (
            torch.tensor(image_patch, dtype=torch.float32, requires_grad=True)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        x_graph = self.graph_data.x.clone().to(self.device)
        edge_index = self.graph_data.edge_index.to(self.device)

        if override_vectors is not None:
            expected_dim = x_graph.shape[1]
            if len(override_vectors) == expected_dim:
                x_graph[node_idx] = torch.tensor(override_vectors, dtype=torch.float32, device=self.device)
            elif len(override_vectors) == 5 and expected_dim == 6:
                # Preserve node's spatial coordinates while updating the 4 risk features
                x_graph[node_idx, :4] = torch.tensor(override_vectors[:4], dtype=torch.float32, device=self.device)

        # Forward Pass
        output = self.model(img_tensor, x_graph, edge_index, target_node_idx=node_idx)

        # Real Softmax Probabilities & Model Confidence
        raw_probs = F.softmax(output, dim=1).detach().cpu().numpy()[0]
        prediction_idx = int(torch.argmax(output, dim=1).item())
        confidence = float(raw_probs[prediction_idx])
        probs_list = [float(p) for p in raw_probs]

        # Execute Backward Pass for Authentic Grad-CAM Calculation
        target_score = output[0, prediction_idx]
        target_score.backward()

        # Compute Grad-CAM Heatmap
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
        cam_resized = F.interpolate(cam_tensor, size=(64, 64), mode="bilinear", align_corners=False)
        real_gradcam_map = cam_resized.squeeze().numpy()

        labels = ["Prime", "Marginal", "Borderline"]

        return {
            "tier": labels[prediction_idx],
            "confidence": confidence,
            "raw_probs": probs_list,
            "target_node_idx": node_idx,
            "connected_edges": int((edge_index[0] == node_idx).sum().item()),
            "gradcam_map": real_gradcam_map.tolist(),
        }


if __name__ == "__main__":
    predictor = SERI_Predictor()
    test_img = np.random.rand(64, 64)
    res = predictor.predict(test_img, node_idx=42)
    print(f"Prediction: {res['tier']} ({res['confidence']*100:.2f}%)")