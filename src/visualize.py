from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

from .config import CFG, CLASS_NAME_ZH
from .data import build_transforms
from .utils import get_device


TITLE_KERNELS = "\u7b2c\u4e00\u5c42\u5377\u79ef\u6838\u53ef\u89c6\u5316"
TITLE_FEATURE_PREFIX = "\u5377\u79ef\u5757\u7279\u5f81\u56fe"
TITLE_ORIGINAL = "\u539f\u59cb\u56fe\u50cf"
TITLE_HEATMAP = "Grad-CAM \u70ed\u529b\u56fe"
TITLE_FOCUS = "\u5173\u6ce8\u533a\u57df"
TITLE_MISTAKES = "\u9519\u8bef\u5206\u7c7b\u6837\u672c\u5206\u6790"
TEXT_NO_MISTAKES = "\u6d4b\u8bd5\u96c6\u4e2d\u6ca1\u6709\u9519\u8bef\u5206\u7c7b\u6837\u672c"
TEXT_TRUE = "\u771f"
TEXT_PRED = "\u9884"


def denormalize_tensor(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image_tensor.detach().cpu() * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image


def save_first_layer_kernels(model: nn.Module, save_path: Path, max_kernels: int = 16):
    conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv = module
            break
    if conv is None:
        raise ValueError("No Conv2d layer found in model.")

    weights = conv.weight.detach().cpu()
    count = min(max_kernels, weights.size(0))
    cols = 4
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.array(axes).reshape(-1)

    for index in range(rows * cols):
        ax = axes[index]
        ax.axis("off")
        if index < count:
            kernel = weights[index]
            kernel = kernel - kernel.min()
            kernel = kernel / (kernel.max() + 1e-8)
            ax.imshow(kernel.permute(1, 2, 0).numpy())
            ax.set_title(f"Kernel {index + 1}")

    fig.suptitle(TITLE_KERNELS)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def save_feature_maps(
    model: nn.Module,
    image_path: str | Path,
    save_path: Path,
    layer_index: int = 0,
    max_maps: int = 16,
    device: torch.device | None = None,
):
    device = device or get_device()
    model.eval().to(device)

    conv_blocks = list(model.features.children())
    if layer_index >= len(conv_blocks):
        raise ValueError(f"layer_index must be < {len(conv_blocks)}")
    feature_extractor = nn.Sequential(*conv_blocks[: layer_index + 1]).to(device)

    image = Image.open(image_path).convert("RGB")
    _, eval_transform = build_transforms(CFG.image_size)
    tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(tensor).squeeze(0).detach().cpu()

    count = min(max_maps, features.size(0))
    cols = 4
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.3))
    axes = np.array(axes).reshape(-1)

    for index in range(rows * cols):
        ax = axes[index]
        ax.axis("off")
        if index < count:
            fmap = features[index].numpy()
            ax.imshow(fmap, cmap="viridis")
            ax.set_title(f"Map {index + 1}")

    fig.suptitle(f"{TITLE_FEATURE_PREFIX} {layer_index + 1}")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output.detach().clone()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach().clone()

    def remove_hooks(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, input_tensor: torch.Tensor, class_index: int | None = None) -> tuple[np.ndarray, int]:
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)
        if class_index is None:
            class_index = int(output.argmax(dim=1).item())
        score = output[:, class_index].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_index


def save_gradcam(
    model: nn.Module,
    image_path: str | Path,
    save_path: Path,
    class_names: list[str] | tuple[str, ...] = CFG.class_names,
    device: torch.device | None = None,
):
    device = device or get_device()
    model.eval().to(device)

    image_path = Path(image_path)
    raw_image = Image.open(image_path).convert("RGB")
    resized = raw_image.resize((CFG.image_size, CFG.image_size))
    _, eval_transform = build_transforms(CFG.image_size)
    tensor = eval_transform(raw_image).unsqueeze(0).to(device)

    target_layer = model.features[-1].block[3]
    gradcam = GradCAM(model, target_layer)
    try:
        cam, class_index = gradcam(tensor)
    finally:
        gradcam.remove_hooks()

    heatmap = cv2.resize(cam, (CFG.image_size, CFG.image_size))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    base = np.asarray(resized).astype(np.float32) / 255.0
    overlay = np.clip(0.55 * base + 0.45 * heatmap, 0, 1)

    label = class_names[class_index]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(base)
    axes[0].set_title(TITLE_ORIGINAL)
    axes[1].imshow(heatmap)
    axes[1].set_title(TITLE_HEATMAP)
    axes[2].imshow(overlay)
    axes[2].set_title(f"{TITLE_FOCUS}: {CLASS_NAME_ZH.get(label, label)}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def save_misclassified_grid(mistakes_df, save_path: Path, max_samples: int = 8):
    if mistakes_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, TEXT_NO_MISTAKES, ha="center", va="center", fontsize=14)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        return fig

    subset = mistakes_df.head(max_samples)
    cols = 4
    rows = int(np.ceil(len(subset) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.4))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(axes, subset.iterrows()):
        image = Image.open(row["image_path"]).convert("RGB")
        ax.imshow(image)
        ax.set_title(f"{TEXT_TRUE}: {row['true_label_zh']}\n{TEXT_PRED}: {row['pred_label_zh']}", fontsize=10)

    fig.suptitle(TITLE_MISTAKES)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig
