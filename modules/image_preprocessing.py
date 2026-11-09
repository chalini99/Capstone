from __future__ import annotations

from PIL import Image
import numpy as np
import torch
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocess_image_for_model(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    if image is None:
        raise ValueError('The input image is empty.')

    image = image.convert('RGB')
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    tensor = transform(image)
    return tensor.unsqueeze(0)


def uint8_image_from_tensor(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return arr


def resize_for_display(image: Image.Image, width: int = 512) -> Image.Image:
    if image is None:
        raise ValueError('No image provided for display.')
    current_width, current_height = image.size
    if current_width <= width:
        return image
    ratio = width / current_width
    new_height = max(1, int(current_height * ratio))
    return image.resize((width, new_height), Image.Resampling.LANCZOS)
