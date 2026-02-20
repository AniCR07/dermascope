import os
import shutil
import random

SOURCE_DIR = r"C:\Users\RAAM\Downloads\DermaDetectAI-main\sd-198\images"
TARGET_DIR = r"C:\Users\RAAM\Downloads\DermaDetectAI-main\model 3"

SELECTED_CLASSES = [
    "Acne_Vulgaris",
    "Actinic_solar_Damage(Actinic_Keratosis)",
    "Basal_Cell_Carcinoma",
    "Benign_Keratosis",
    "Bowen's_Disease",
    "Dermatofibroma",
    "Discoid_Lupus_Erythematosus",
    "Eczema",
    "Herpes_Simplex_Virus",
    "Herpes_Zoster",
    "Impetigo",
    "Lichen_Planus",
    "Malignant_Melanoma",
    "Molluscum_Contagiosum",
    "Psoriasis",
    "Rosacea",
    "Seborrheic_Dermatitis",
    "Seborrheic_Keratosis",
    "Skin_Tag",
    "Tinea_Corporis",
    "Tinea_Pedis",
    "Urticaria",
    "Vitiligo"
]
SPLIT = 0.8

for split in ["train", "val"]:
    for cls in SELECTED_CLASSES:
        os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

for cls in SELECTED_CLASSES:
    src_path = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(src_path)
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for img in train_imgs:
        shutil.copy(os.path.join(src_path, img),
                    os.path.join(TARGET_DIR, "train", cls, img))

    for img in val_imgs:
        shutil.copy(os.path.join(src_path, img),
                    os.path.join(TARGET_DIR, "val", cls, img))

print("Model 3 dataset prepared.")