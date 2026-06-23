from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import CFG
from src.data import create_dataloaders
from src.evaluate import evaluate_model, find_misclassified_samples
from src.model import create_model
from src.train import load_best_weights
from src.utils import ensure_dirs, setup_matplotlib_chinese
from src.visualize import (
    save_feature_maps,
    save_first_layer_kernels,
    save_gradcam,
    save_misclassified_grid,
)


def main() -> None:
    setup_matplotlib_chinese()
    ensure_dirs(CFG.output_dir)

    datasets_map, loaders = create_dataloaders(CFG.data_dir, CFG.batch_size, CFG.num_workers)
    model = create_model(num_classes=len(datasets_map["train"].classes))
    model, checkpoint = load_best_weights(model, CFG.best_model_path)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with val_acc={checkpoint['val_acc']:.4f}")

    save_first_layer_kernels(model, CFG.output_dir / "conv_kernels.png", max_kernels=16)

    demo_images = sorted(CFG.predict_dir.glob("*.jpg"))
    if not demo_images:
        raise RuntimeError(f"No demo images found in {CFG.predict_dir}")

    image_for_feature = demo_images[0]
    save_feature_maps(
        model,
        image_for_feature,
        CFG.output_dir / "feature_maps_block1.png",
        layer_index=0,
        max_maps=16,
    )
    save_feature_maps(
        model,
        image_for_feature,
        CFG.output_dir / "feature_maps_block4.png",
        layer_index=3,
        max_maps=16,
    )

    for index, image_path in enumerate(demo_images[:4], start=1):
        save_gradcam(
            model,
            image_path,
            CFG.output_dir / f"gradcam_{index:02d}.png",
            class_names=datasets_map["test"].classes,
        )

    report_df, y_true, y_pred, _ = evaluate_model(
        model,
        loaders["test"],
        class_names=datasets_map["test"].classes,
    )
    mistakes = find_misclassified_samples(datasets_map["test"], y_true, y_pred, max_samples=20)
    if mistakes.empty and (CFG.output_dir / "misclassified_samples.csv").exists():
        mistakes = pd.read_csv(CFG.output_dir / "misclassified_samples.csv")
    save_misclassified_grid(mistakes, CFG.output_dir / "misclassified_grid.png", max_samples=8)

    print("Generated visualization files:")
    for name in [
        "conv_kernels.png",
        "feature_maps_block1.png",
        "feature_maps_block4.png",
        "gradcam_01.png",
        "gradcam_02.png",
        "gradcam_03.png",
        "gradcam_04.png",
        "misclassified_grid.png",
    ]:
        path = CFG.output_dir / name
        print(f"  {name}: {path.exists()}")


if __name__ == "__main__":
    main()
