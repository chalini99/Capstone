from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: str | None = 'features.8'):
        self.model = model
        self.target_layer_name = target_layer or 'features.8'
        self._target_module = self._resolve_target_layer(self.target_layer_name)
        self._feature_map = None
        self._gradients = None
        self._forward_handle = None
        self._backward_handle = None
        self._register_hooks()

    def _resolve_target_layer(self, layer_path: str) -> torch.nn.Module:
        """Resolve target layer by path or automatically fall back to the last Conv2d layer."""
        if layer_path:
            try:
                curr = self.model
                for part in layer_path.split('.'):
                    if hasattr(curr, part):
                        curr = getattr(curr, part)
                    elif hasattr(curr, '__getitem__') and part.isdigit():
                        curr = curr[int(part)]
                    else:
                        raise AttributeError(f"Module has no attribute '{part}'")
                return curr
            except Exception:
                pass

        # Robust fallback: automatically locate the last Conv2d in the feature extractor
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module

        if last_conv is not None:
            return last_conv

        raise AttributeError(f"Unable to resolve target layer '{layer_path}' in model architecture.")

    def _register_hooks(self) -> None:
        if self._forward_handle is not None:
            self._forward_handle.remove()
        if self._backward_handle is not None:
            self._backward_handle.remove()

        self._forward_handle = self._target_module.register_forward_hook(self._forward_hook)
        self._backward_handle = self._target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: Any, inputs: Any, output: Any) -> None:
        self._feature_map = output

    def _backward_hook(self, module: Any, grad_input: Any, grad_output: Any) -> None:
        self._gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        if input_tensor is None:
            raise ValueError('Grad-CAM requires a valid input tensor.')

        was_training = self.model.training
        self.model.eval()

        self.model.zero_grad()
        self._feature_map = None
        self._gradients = None

        with torch.enable_grad():
            tensor = input_tensor.clone().detach()
            tensor.requires_grad_(True)

            output = self.model(tensor)
            if target_class is None:
                target_class = int(output.argmax(dim=1).item())

            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1.0
            output.backward(gradient=one_hot, retain_graph=True)

        if was_training:
            self.model.train()

        gradients = self._gradients
        feature_map = self._feature_map
        if feature_map is None or gradients is None:
            raise RuntimeError('Grad-CAM gradients were not captured. Target layer was not reached during forward/backward pass.')

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = (weights * feature_map).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(input_tensor.shape[-2], input_tensor.shape[-1]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        denom = cam.max() - cam.min()
        if denom > 1e-8:
            cam = (cam - cam.min()) / denom
        else:
            cam = np.zeros_like(cam)
        return cam

    def __del__(self):
        if self._forward_handle is not None:
            try:
                self._forward_handle.remove()
            except Exception:
                pass
        if self._backward_handle is not None:
            try:
                self._backward_handle.remove()
            except Exception:
                pass


def get_heatmap_image(heatmap: np.ndarray, size: tuple[int, int] | None = None) -> Image.Image:
    """Generate a standalone colorized heatmap image using JET colormap."""
    if heatmap is None:
        raise ValueError("Heatmap array is required.")

    heatmap_2d = np.squeeze(heatmap)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_2d, 0, 1))

    try:
        import cv2
        if size is not None:
            heatmap_uint8 = cv2.resize(heatmap_uint8, size)
        colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(colored_rgb)
    except Exception:
        img = Image.fromarray(heatmap_uint8).convert("RGB")
        if size is not None:
            img = img.resize(size, Image.Resampling.BILINEAR)
        return img


def overlay_heatmap_on_image(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    if image is None:
        raise ValueError('Original image is required for heatmap overlay.')
    if heatmap is None:
        return image

    try:
        import cv2
        heatmap_2d = np.squeeze(heatmap)
        heatmap_resized = cv2.resize(heatmap_2d, (image.width, image.height))
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))

        # Apply JET colormap for standard Grad-CAM pseudo-coloring
        colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        orig_rgb = np.array(image.convert("RGB"))
        blended = np.uint8(alpha * colored_rgb + (1 - alpha) * orig_rgb)
        return Image.fromarray(blended)
    except Exception:
        # Fallback to PIL blending if cv2 encounters any issue
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize(image.size, Image.Resampling.BILINEAR)
        heatmap_colored = heatmap_img.convert('RGBA')
        original = image.convert('RGBA')
        overlay = Image.blend(original, heatmap_colored, alpha=0.35)
        return overlay.convert('RGB')
