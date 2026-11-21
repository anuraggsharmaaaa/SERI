import torch
import cv2
import numpy as np

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

    def get_heatmap(self, image_tensor, vector_tensor, target_class):
        # 1. Hook into the last Convolutional Layer
        # In our architecture, this is vision_branch[3]
        target_layer = self.model.vision_branch[3]
        
        self.handlers.append(target_layer.register_forward_hook(self.save_activation))
        self.handlers.append(target_layer.register_full_backward_hook(lambda m, gi, go: self.save_gradient(go[0])))

        # 2. Forward Pass
        output = self.model(image_tensor, vector_tensor)
        self.model.zero_grad()

        # 3. Backward Pass
        one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 4. Generate Heatmap
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().numpy()
        
        # Cleanup hooks
        for h in self.handlers: h.remove()

        # 5. Process Heatmap for Visuals
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (64, 64))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam