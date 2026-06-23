from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import CFG
from src.data import create_dataloaders, dataset_summary, plot_class_distribution
from src.evaluate import evaluate_model, find_misclassified_samples, plot_confusion_matrix, plot_training_curves
from src.model import create_model
from src.predict import predict_folder
from src.train import load_best_weights, train_model
from src.utils import count_parameters, ensure_dirs, get_device, set_seed, setup_matplotlib_chinese


def main() -> None:
    setup_matplotlib_chinese()
    set_seed(CFG.seed)
    ensure_dirs(CFG.model_dir, CFG.output_dir)

    print("Device:", get_device())
    print("Data dir:", CFG.data_dir)
    print("Classes:", CFG.class_names)

    summary = dataset_summary(CFG.data_dir)
    print("\nDataset summary:")
    print(summary)
    summary.to_csv(CFG.output_dir / "dataset_summary.csv", index=False, encoding="utf-8-sig")
    plot_class_distribution(summary, CFG.output_dir / "class_distribution.png")

    datasets_map, loaders = create_dataloaders(CFG.data_dir, CFG.batch_size, CFG.num_workers)
    model = create_model(num_classes=len(datasets_map["train"].classes))
    print(f"\nModel trainable parameters: {count_parameters(model):,}")

    history = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        epochs=CFG.epochs,
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        model_path=CFG.best_model_path,
        seed=CFG.seed,
    )
    history.to_csv(CFG.output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    plot_training_curves(history, CFG.output_dir / "training_curves.png")

    best_model = create_model(num_classes=len(datasets_map["train"].classes))
    best_model, checkpoint = load_best_weights(best_model, CFG.best_model_path)
    print("\nLoaded best checkpoint:", checkpoint["epoch"], checkpoint["val_acc"])

    report_df, y_true, y_pred, _ = evaluate_model(
        best_model,
        loaders["test"],
        class_names=datasets_map["test"].classes,
    )
    print("\nClassification report:")
    print(report_df)
    report_df.to_csv(CFG.output_dir / "classification_report.csv", encoding="utf-8-sig")
    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=datasets_map["test"].classes,
        save_path=CFG.output_dir / "confusion_matrix.png",
    )

    mistakes = find_misclassified_samples(datasets_map["test"], y_true, y_pred, max_samples=20)
    mistakes.to_csv(CFG.output_dir / "misclassified_samples.csv", index=False, encoding="utf-8-sig")

    predictions = predict_folder(best_model, CFG.predict_dir, class_names=datasets_map["test"].classes)
    prediction_rows = [
        {
            "image_path": str(item["image_path"]),
            "pred_label": item["pred_label"],
            "pred_label_zh": item["pred_label_zh"],
            "confidence": item["confidence"],
        }
        for item in predictions
    ]
    pd.DataFrame(prediction_rows).to_csv(CFG.output_dir / "prediction_results.csv", index=False, encoding="utf-8-sig")
    print("\nPrediction demo results:")
    print(pd.DataFrame(prediction_rows))


if __name__ == "__main__":
    main()
