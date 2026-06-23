from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import CFG, CLASS_NAME_ZH


def build_transforms(image_size: int = CFG.image_size) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def create_datasets(data_dir: Path = CFG.data_dir):
    train_transform, eval_transform = build_transforms(CFG.image_size)
    datasets_map = {
        "train": datasets.ImageFolder(data_dir / "train", transform=train_transform),
        "val": datasets.ImageFolder(data_dir / "val", transform=eval_transform),
        "test": datasets.ImageFolder(data_dir / "test", transform=eval_transform),
    }
    return datasets_map


def create_dataloaders(
    data_dir: Path = CFG.data_dir,
    batch_size: int = CFG.batch_size,
    num_workers: int = CFG.num_workers,
):
    datasets_map = create_datasets(data_dir)
    loaders = {
        "train": DataLoader(
            datasets_map["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            datasets_map["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets_map["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    return datasets_map, loaders


def dataset_summary(data_dir: Path = CFG.data_dir) -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        for class_dir in sorted([path for path in split_dir.iterdir() if path.is_dir()]):
            count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
            rows.append(
                {
                    "split": split,
                    "class": class_dir.name,
                    "class_zh": CLASS_NAME_ZH.get(class_dir.name, class_dir.name),
                    "count": count,
                }
            )
    return pd.DataFrame(rows)


def plot_class_distribution(summary_df: pd.DataFrame, save_path: Path | None = None):
    pivot = summary_df.pivot(index="class_zh", columns="split", values="count").fillna(0)
    pivot = pivot[["train", "val", "test"]]
    ax = pivot.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_title("玉米叶片病害数据集类别分布")
    ax.set_xlabel("类别")
    ax.set_ylabel("图片数量")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    return ax


def show_dataset_samples(data_dir: Path = CFG.data_dir, split: str = "train", samples_per_class: int = 4):
    split_dir = data_dir / split
    class_dirs = sorted([path for path in split_dir.iterdir() if path.is_dir()])
    fig, axes = plt.subplots(
        len(class_dirs),
        samples_per_class,
        figsize=(samples_per_class * 3, len(class_dirs) * 2.8),
    )
    if len(class_dirs) == 1:
        axes = [axes]

    for row, class_dir in enumerate(class_dirs):
        images = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))[:samples_per_class]
        for col in range(samples_per_class):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(images):
                image = Image.open(images[col]).convert("RGB")
                ax.imshow(image)
                title = CLASS_NAME_ZH.get(class_dir.name, class_dir.name)
                if col == 0:
                    ax.set_title(title)
    plt.tight_layout()
    return fig


def get_class_counts(dataset: datasets.ImageFolder) -> Counter:
    return Counter(target for _, target in dataset.samples)
