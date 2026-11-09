
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import efficientnet_b0

from config.config import (
    CLASS_MAPPING_PATH,
    EXPECTED_NUM_CLASSES,
    MODEL_PATH,
)
from modules.image_preprocessing import preprocess_image_for_model


@dataclass
class PredictionResult:
    crop: str
    disease: str
    confidence: float
    top_predictions: list
    prediction_label: str
    status: str
    message: str

    @property
    def confidence_pct(self):
        return self.confidence * 100


def build_efficientnet_b0(
    num_classes: int = EXPECTED_NUM_CLASSES
) -> torch.nn.Module:
    """Create the EfficientNet-B0 architecture used during training."""

    model = efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(
        in_features,
        num_classes
    )

    return model


def load_class_mapping(mapping_path=CLASS_MAPPING_PATH) -> dict:
    """Load the numeric class-to-label mapping."""

    mapping_path = Path(mapping_path)

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Class mapping file not found: {mapping_path}"
        )

    with open(mapping_path, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    if isinstance(mapping, list):
        mapping = {
            str(index): label
            for index, label in enumerate(mapping)
        }

    if not isinstance(mapping, dict):
        raise ValueError("Invalid class mapping format.")

    mapping = {
        str(key): str(value)
        for key, value in mapping.items()
    }

    if len(mapping) != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_NUM_CLASSES} classes, "
            f"but found {len(mapping)}."
        )

    return mapping


def parse_label(label: str) -> tuple:
    """
    Convert a PlantVillage label such as
    Apple___Apple_scab into crop and disease names.
    """

    label = str(label).strip()

    if "___" in label:
        crop, disease = label.split("___", 1)
    elif "__" in label:
        crop, disease = label.split("__", 1)
    else:
        parts = label.split("_", 1)
        crop = parts[0]
        disease = parts[1] if len(parts) > 1 else "Unknown"

    crop = crop.replace("_", " ").strip()
    disease = disease.replace("_", " ").strip()

    # Canonicalize known dataset label typo (PlantVillage Haunglongbing -> Huanglongbing)
    if "haunglongbing" in disease.lower():
        import re
        disease = re.sub(r'(?i)haunglongbing', 'Huanglongbing', disease)

    return crop, disease


def validate_checkpoint_compatibility(model, model_path):
    """Load the actual trained weights from the checkpoint."""

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    try:
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False
        )

        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid checkpoint format.")

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise ValueError("Model weights are missing.")

        # Strict loading ensures that the architecture
        # matches the trained EfficientNet-B0 checkpoint.
        model.load_state_dict(state_dict, strict=True)

        return True

    except Exception as error:
        raise RuntimeError(
            f"Checkpoint loading failed: {error}"
        ) from error


def load_trained_model(
    model_path=MODEL_PATH,
    mapping_path=CLASS_MAPPING_PATH
):
    """Load the trained EfficientNet-B0 model and class mapping."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    mapping = load_class_mapping(mapping_path)

    model = build_efficientnet_b0(
        num_classes=EXPECTED_NUM_CLASSES
    )

    validate_checkpoint_compatibility(model, model_path)

    model = model.to(device)
    model.eval()

    return model, mapping, device


def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
    class_mapping: dict,
    device: torch.device,
    top_k: int = 3
) -> PredictionResult:
    """Predict crop disease from a PIL image."""

    if image is None:
        raise ValueError("Please provide an image.")

    image = image.convert("RGB")

    # Use the existing project preprocessing module.
    input_tensor = preprocess_image_for_model(
        image,
        image_size=224
    )

    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(
            "Image preprocessing must return a PyTorch tensor."
        )

    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    model.eval()

    with torch.inference_mode():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)

        k = min(top_k, probabilities.shape[1])
        confidence_values, predicted_indices = torch.topk(
            probabilities,
            k=k,
            dim=1
        )

    confidence_values = confidence_values[0].cpu().tolist()
    predicted_indices = predicted_indices[0].cpu().tolist()

    top_predictions = []

    for index, confidence in zip(
        predicted_indices,
        confidence_values
    ):
        label = class_mapping.get(
            str(index),
            f"Unknown class {index}"
        )

        crop, disease = parse_label(label)

        top_predictions.append({
            "class_index": int(index),
            "crop": crop,
            "disease": disease,
            "label": label,
            "confidence": float(confidence),
            "confidence_pct": float(confidence) * 100
        })

    best_prediction = top_predictions[0]

    return PredictionResult(
        crop=best_prediction["crop"],
        disease=best_prediction["disease"],
        confidence=best_prediction["confidence"],
        top_predictions=top_predictions,
        prediction_label=best_prediction["label"],
        status="success",
        message="Disease prediction completed successfully."
    )


def predict_uploaded_image(
    uploaded_file,
    model,
    class_mapping,
    device,
    top_k=3
) -> PredictionResult:
    """Handle an uploaded image and return its disease prediction."""

    try:
        image = Image.open(uploaded_file).convert("RGB")

        return predict_image(
            image=image,
            model=model,
            class_mapping=class_mapping,
            device=device,
            top_k=top_k
        )

    except Exception as error:
        raise RuntimeError(
            f"Unable to process the uploaded image: {error}"
        ) from error
