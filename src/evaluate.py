from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from .config import CFG, CLASS_NAME_ZH
from .utils import get_device


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = device or get_device()
    model.eval()
    model.to(device)

    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str] | tuple[str, ...] = CFG.class_names,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    y_true, y_pred, y_prob = collect_predictions(model, loader, device)
    target_names = [CLASS_NAME_ZH.get(name, name) for name in class_names]
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    return report_df, y_true, y_pred, y_prob


def plot_training_curves(history: pd.DataFrame, save_path: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="训练损失")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", label="验证损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("损失曲线")
    axes[0].grid(linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_acc"], marker="o", label="训练准确率")
    axes[1].plot(history["epoch"], history["val_acc"], marker="o", label="验证准确率")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("准确率曲线")
    axes[1].grid(linestyle="--", alpha=0.35)
    axes[1].legend()

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | tuple[str, ...] = CFG.class_names,
    save_path: Path | None = None,
):
    labels = [CLASS_NAME_ZH.get(name, name) for name in class_names]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("测试集混淆矩阵")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, cm


def find_misclassified_samples(dataset, y_true: np.ndarray, y_pred: np.ndarray, max_samples: int = 12):
    mistakes = np.where(y_true != y_pred)[0][:max_samples]
    rows = []
    for index in mistakes:
        path, _ = dataset.samples[index]
        rows.append(
            {
                "image_path": path,
                "true_label": dataset.classes[y_true[index]],
                "pred_label": dataset.classes[y_pred[index]],
                "true_label_zh": CLASS_NAME_ZH.get(dataset.classes[y_true[index]], dataset.classes[y_true[index]]),
                "pred_label_zh": CLASS_NAME_ZH.get(dataset.classes[y_pred[index]], dataset.classes[y_pred[index]]),
            }
        )
    return pd.DataFrame(rows)
