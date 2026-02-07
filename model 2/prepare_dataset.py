import os
import shutil
import random

# ---------------- CONFIG ----------------
DATASET_DIR = r"C:\Users\RAAM\Downloads\DermaDetectAI-main\sd-198"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
TRAIN_RATIO = 0.8

SELECTED_CLASSES = [
    "Acne_Vulgaris",
    "Actinic_solar_Damage(Actinic_Keratosis)",
    "Basal_Cell_Carcinoma",
    "Benign_Keratosis",
    "Eczema",
    "Psoriasis",
    "Rosacea",
    "Vitiligo",
    "Impetigo",
    "Seborrheic_Dermatitis"
]

# ---------------- CLEAN OLD FOLDERS ----------------
for split in ["train", "val"]:
    split_path = os.path.join(DATASET_DIR, split)
    if os.path.exists(split_path):
        shutil.rmtree(split_path)

# ---------------- CREATE DIRS ----------------
for split in ["train", "val"]:
    for cls in SELECTED_CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)

# ---------------- SPLIT IMAGES ----------------
for cls in SELECTED_CLASSES:
    class_dir = os.path.join(IMAGES_DIR, cls)
    images = os.listdir(class_dir)
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for img in train_imgs:
        shutil.copy(
            os.path.join(class_dir, img),
            os.path.join(DATASET_DIR, "train", cls, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(class_dir, img),
            os.path.join(DATASET_DIR, "val", cls, img)
        )

print("✅ Dataset prepared successfully using folder-based classes")
