from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from .config import CFG, CLASS_NAME_ZH
from .utils import get_device


def prediction_transform(image_size: int = CFG.image_size) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def predict_image(
    model: nn.Module,
    image_path: str | Path,
    class_names: list[str] | tuple[str, ...] = CFG.class_names,
    device: torch.device | None = None,
) -> dict:
    device = device or get_device()
    model.eval()
    model.to(device)

    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    tensor = prediction_transform()(image).unsqueeze(0).to(device)
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1).squeeze(0).cpu()
    confidence, pred_index = torch.max(probs, dim=0)
    label = class_names[int(pred_index)]

    return {
        "image_path": image_path,
        "pred_index": int(pred_index),
        "pred_label": label,
        "pred_label_zh": CLASS_NAME_ZH.get(label, label),
        "confidence": float(confidence),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }


def plot_prediction(result: dict, save_path: Path | None = None):
    image = Image.open(result["image_path"]).convert("RGB")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"{result['pred_label_zh']}  置信度: {result['confidence']:.2%}")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def predict_folder(
    model: nn.Module,
    folder: str | Path = CFG.predict_dir,
    class_names: list[str] | tuple[str, ...] = CFG.class_names,
    device: torch.device | None = None,
) -> list[dict]:
    folder = Path(folder)
    image_paths = sorted(
        [
            path
            for path in folder.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )
    return [predict_image(model, path, class_names=class_names, device=device) for path in image_paths]
