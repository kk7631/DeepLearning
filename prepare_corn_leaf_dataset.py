from __future__ import annotations

import io
import random
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "corn_leaf_dataset"
PREDICT_DIR = ROOT / "corn_predict_images"

REPO_API = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"
USER_AGENT = "CornLeafCourseDesign/1.0"

CLASSES = {
    "healthy": "Corn_(maize)___healthy",
    "gray_leaf_spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "common_rust": "Corn_(maize)___Common_rust_",
    "northern_leaf_blight": "Corn_(maize)___Northern_Leaf_Blight",
}

SPLIT_COUNTS = {
    "train": 400,
    "val": 50,
    "test": 50,
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def ensure_dirs() -> None:
    for split in SPLIT_COUNTS:
        for label in CLASSES:
            (DATASET_DIR / split / label).mkdir(parents=True, exist_ok=True)
    PREDICT_DIR.mkdir(parents=True, exist_ok=True)


def clear_generated_images() -> None:
    for folder in [DATASET_DIR, PREDICT_DIR]:
        if not folder.exists():
            continue
        for path in folder.rglob("plantvillage_*.jpg"):
            path.unlink()


def list_class_images(remote_class_name: str) -> list[dict]:
    url = f"{REPO_API}/{remote_class_name}"
    response = SESSION.get(url, timeout=60)
    response.raise_for_status()
    items = response.json()
    return [
        item
        for item in items
        if item.get("type") == "file"
        and item.get("download_url")
        and item["name"].lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def load_image(url: str) -> Image.Image | None:
    try:
        response = SESSION.get(url, timeout=60)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image = ImageOps.exif_transpose(image).convert("RGB")
        if min(image.size) < 80:
            return None
        return image
    except Exception:
        return None


def save_image(image: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="JPEG", quality=92, optimize=True)


def target_path(label: str, split: str, index: int) -> Path:
    return DATASET_DIR / split / label / f"plantvillage_{label}_{split}_{index:04d}.jpg"


def download_label(label: str, remote_class_name: str) -> None:
    needed = sum(SPLIT_COUNTS.values())
    print(f"\n[{label}] {remote_class_name}")
    images = list_class_images(remote_class_name)
    print(f"  available: {len(images)}, needed: {needed}")
    if len(images) < needed:
        raise RuntimeError(f"{remote_class_name} has only {len(images)} images, need {needed}")

    rng = random.Random(2026)
    rng.shuffle(images)

    cursor = 0
    for split, count in SPLIT_COUNTS.items():
        saved = 0
        while saved < count and cursor < len(images):
            item = images[cursor]
            cursor += 1
            image = load_image(item["download_url"])
            if image is None:
                continue

            saved += 1
            save_image(image, target_path(label, split, saved))

            if saved == 1 or saved % 50 == 0 or saved == count:
                print(f"  {split:5s}: {saved:04d}/{count}")
            time.sleep(0.03)

        if saved < count:
            raise RuntimeError(f"{label}/{split} downloaded {saved}/{count}")


def prepare_predict_images() -> None:
    for old in PREDICT_DIR.glob("plantvillage_*.jpg"):
        old.unlink()

    copied = 0
    for label in CLASSES:
        for source in sorted((DATASET_DIR / "test" / label).glob("plantvillage_*.jpg"))[:3]:
            copied += 1
            target = PREDICT_DIR / f"plantvillage_{label}_{copied:02d}.jpg"
            target.write_bytes(source.read_bytes())
    print(f"\nPrepared prediction images: {copied}")


def summarize() -> None:
    print("\nDataset summary:")
    for split in SPLIT_COUNTS:
        for label in CLASSES:
            count = len(list((DATASET_DIR / split / label).glob("*.jpg")))
            print(f"  {split:5s} {label:22s}: {count}")


def main() -> None:
    ensure_dirs()
    clear_generated_images()
    for label, remote_class_name in CLASSES.items():
        download_label(label, remote_class_name)
    prepare_predict_images()
    summarize()


if __name__ == "__main__":
    main()
