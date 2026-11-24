import cv2
import numpy as np
import torch


class SERI_Explainer:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.handlers = []

    def save_gradient(self, grad):
        self.gradients = grad

    def save_activation(self, module, input, output):
        self.activations = output

    def get_heatmap(self, image_tensor, x_graph, edge_index, target_node_idx, target_class):
        self.handlers = []
        # Hook into the second Conv2d layer (vision_branch[3])
        target_layer = self.model.vision_branch[3]

        self.handlers.append(target_layer.register_forward_hook(self.save_activation))
        self.handlers.append(
            target_layer.register_full_backward_hook(lambda m, gi, go: self.save_gradient(go[0]))
        )

        # Forward Pass matching SERI_MultiModal_Engine.forward
        output = self.model(image_tensor, x_graph, edge_index, target_node_idx=target_node_idx)
        self.model.zero_grad()

        # Backward Pass
        one_hot = torch.zeros((1, output.size(-1)), dtype=torch.float32, device=output.device)
        one_hot[0][target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Generate Heatmap
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()

        # Cleanup hooks
        for h in self.handlers:
            h.remove()
        self.handlers = []

        # Process Heatmap for Visuals
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (64, 64))
        cam = cam - np.min(cam)
        max_val = np.max(cam)
        if max_val > 0:
            cam = cam / max_val

        return cam