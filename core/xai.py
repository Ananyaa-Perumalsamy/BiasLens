"""
core/xai.py
────────────
Explainability tools:
  • GradCAM       – gradient-weighted class activation maps
  • SmoothGradCAM – averaged over noisy copies (smoother maps)
  • SHAP wrapper  – DeepSHAP approximation via captum / fallback
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ══════════════════════════════════════════════════════════════
#  Grad-CAM
# ══════════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Args:
        model     : BiasAwareCNN (or any model with a 'backbone' attribute)
        target_layer_name (str): name of the conv layer to hook.
                                 For ResNet-18 backbone: 'layer4'
    """

    def __init__(self, model, target_layer_name: str = "layer4"):
        self.model  = model
        self.model.eval()

        self._gradients = None
        self._activations = None

        # Navigate into backbone to find the target layer
        target_layer = self._find_layer(target_layer_name)
        if target_layer is None:
            raise ValueError(f"Layer '{target_layer_name}' not found in model.")

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    # ----------------------------------------------------------
    def _find_layer(self, name: str):
        """Walk the backbone to find a named child."""
        # Handle ResNet wrapped in nn.Sequential by mapping original child names.
        resnet_name_map = {
            "conv1": 0,
            "bn1": 1,
            "relu": 2,
            "maxpool": 3,
            "layer1": 4,
            "layer2": 5,
            "layer3": 6,
            "layer4": 7,
            "avgpool": 8,
        }

        if hasattr(self.model, "backbone"):
            backbone = self.model.backbone
            if isinstance(backbone, torch.nn.Sequential):
                if name in resnet_name_map:
                    idx = resnet_name_map[name]
                    if idx < len(backbone):
                        return backbone[idx]
                if name.isdigit():
                    idx = int(name)
                    if 0 <= idx < len(backbone):
                        return backbone[idx]

        # Fallback: iterate named modules on the full model.
        for n, m in self.model.named_modules():
            if n == name or n.endswith(name) or m.__class__.__name__ == name:
                return m
        return None

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    # ----------------------------------------------------------
    def generate(self, img_tensor: torch.Tensor,
                 target_class: int = None) -> np.ndarray:
        """
        Args:
            img_tensor   : (1, C, H, W) float tensor
            target_class : class index. If None, uses argmax.

        Returns:
            cam (np.ndarray) : (H, W) float32 in [0, 1]
        """
        self.model.zero_grad()
        logits, _ = self.model(img_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward on target class score
        score = logits[0, target_class]
        score.backward()

        # Global average pooling of gradients → weights
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.astype(np.float32)

    def overlay(self, cam: np.ndarray, original_img: np.ndarray,
                alpha: float = 0.5, colormap=cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlays the CAM heatmap on the original image.

        Args:
            cam          : (H, W) float32 [0,1]
            original_img : (H, W, 3) uint8
            alpha        : blend factor
            colormap     : cv2 colormap

        Returns:
            blended (np.ndarray): (H, W, 3) uint8
        """
        h, w = original_img.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        blended = np.uint8(alpha * heatmap + (1 - alpha) * original_img)
        return blended


# ══════════════════════════════════════════════════════════════
#  Smooth Grad-CAM
# ══════════════════════════════════════════════════════════════

class SmoothGradCAM(GradCAM):
    """
    Averages Grad-CAM maps over n_samples noisy copies of the input.
    Produces smoother, less noisy explanations.
    """

    def __init__(self, model, target_layer_name: str = "layer4",
                 n_samples: int = 10, noise_std: float = 0.1):
        super().__init__(model, target_layer_name)
        self.n_samples = n_samples
        self.noise_std = noise_std

    def generate(self, img_tensor: torch.Tensor,
                 target_class: int = None) -> np.ndarray:
        cams = []
        for _ in range(self.n_samples):
            noisy = img_tensor + torch.randn_like(img_tensor) * self.noise_std
            cam   = super().generate(noisy, target_class)
            cams.append(cam)

        avg_cam = np.mean(cams, axis=0)
        avg_cam = (avg_cam - avg_cam.min()) / (avg_cam.max() + 1e-9)
        return avg_cam.astype(np.float32)


# ══════════════════════════════════════════════════════════════
#  Utility: preprocess PIL image for model
# ══════════════════════════════════════════════════════════════

PREPROCESS = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def pil_to_tensor(pil_img: Image.Image, device: str = "cpu") -> torch.Tensor:
    """Converts PIL image → (1, C, H, W) tensor ready for GradCAM."""
    tensor = PREPROCESS(pil_img).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    return tensor


def tensor_to_numpy_img(tensor: torch.Tensor) -> np.ndarray:
    """De-normalises a (1, C, H, W) or (C, H, W) tensor → (H, W, 3) uint8."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = tensor.squeeze().cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════
#  Batch GradCAM for gallery view
# ══════════════════════════════════════════════════════════════

def batch_gradcam(model, pil_images, device="cpu",
                  smooth=False, target_class=None):
    """
    Generates GradCAM overlays for a list of PIL images.

    Returns:
        List of (original_img_np, cam_np, overlay_np, predicted_class)
    """
    CAMClass = SmoothGradCAM if smooth else GradCAM
    cam_gen  = CAMClass(model)

    results = []
    for pil_img in pil_images:
        orig_np = np.array(pil_img.resize((128, 128)))
        tensor  = pil_to_tensor(pil_img, device)

        with torch.enable_grad():
            cam = cam_gen.generate(tensor, target_class)

        logits, _ = model(tensor)
        pred_class = logits.argmax(dim=1).item()

        overlay = cam_gen.overlay(cam, orig_np)
        results.append((orig_np, cam, overlay, pred_class))

    return results
