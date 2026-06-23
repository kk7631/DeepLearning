from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CLASS_NAMES = [
    "common_rust",
    "gray_leaf_spot",
    "healthy",
    "northern_leaf_blight",
]

CLASS_NAME_ZH = {
    "common_rust": "\u666e\u901a\u9508\u75c5",
    "gray_leaf_spot": "\u7070\u6591\u75c5",
    "healthy": "\u5065\u5eb7\u53f6\u7247",
    "northern_leaf_blight": "\u5317\u65b9\u53f6\u67af\u75c5",
}


@dataclass(frozen=True)
class Config:
    data_dir: Path = PROJECT_ROOT / "corn_leaf_dataset"
    predict_dir: Path = PROJECT_ROOT / "corn_predict_images"
    model_dir: Path = PROJECT_ROOT / "models"
    output_dir: Path = PROJECT_ROOT / "outputs"
    best_model_path: Path = PROJECT_ROOT / "models" / "best_corn_leaf_cnn.pth"

    class_names: tuple[str, ...] = tuple(CLASS_NAMES)
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 2026


CFG = Config()
