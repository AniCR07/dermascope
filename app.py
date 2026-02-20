import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI DermaScope", layout="wide")

st.title("🩺 AI DermaScope")
st.write("Skin Disease Detection using Deep Learning")

# ===============================
# MODEL CONFIGURATION
# ===============================
MODEL_CONFIGS = {
    "Model 1 (5 classes)": {
        "num_classes": 5,
        "path": "model 1/skin_disease_model.pth",
        "classes": [
            "Acne_Vulgaris",
            "Actinic_solar_Damage(Actinic_Keratosis)",
            "Basal_Cell_Carcinoma",
            "Eczema",
            "Psoriasis"
        ]
    },
    "Model 2 (10 classes)": {
        "num_classes": 10,
        "path": "model 2/skin_disease_model.pth",
        "classes": [
            "Acne_Vulgaris",
            "Actinic_solar_Damage(Actinic_Keratosis)",
            "Basal_Cell_Carcinoma",
            "Benign_Keratosis",
            "Eczema",
            "Impetigo",
            "Psoriasis",
            "Rosacea",
            "Seborrheic_Dermatitis",
            "Vitiligo"
        ]
    },
    "Model 3 (23 classes)": {
        "num_classes": 23,
        "path": "model 3/skin_disease_model.pth",
        "classes": [
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
    }
}

# ===============================
# SELECT MODEL
# ===============================
st.header("Select Model")

model_key = st.radio(
    "Choose detection model:",
    list(MODEL_CONFIGS.keys()),
    horizontal=True
)

config = MODEL_CONFIGS[model_key]
st.success(f"Using {model_key}")

# ===============================
# LOAD MODEL
# ===============================
assert os.path.exists(config["path"]), f"Model file not found: {config['path']}"

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, config["num_classes"])

state_dict = torch.load(config["path"], map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

CLASS_NAMES = config["classes"]

# ===============================
# IMAGE TRANSFORM (MATCH TRAINING)
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# REMEDY DATABASE (ALL 23 CLASSES)
# ===============================
REMEDIES = {

    "Acne_Vulgaris": [
        "Wash face twice daily with mild cleanser",
        "Avoid oily skincare products",
        "Do not squeeze pimples",
        "Maintain balanced diet",
        "Stay hydrated"
    ],

    "Actinic_solar_Damage(Actinic_Keratosis)": [
        "Use sunscreen SPF 30+ daily",
        "Avoid prolonged sun exposure",
        "Wear protective clothing",
        "Regular skin check-ups"
    ],

    "Basal_Cell_Carcinoma": [
        "Avoid excessive sun exposure",
        "Use sunscreen regularly",
        "Do not ignore persistent skin lesions",
        "Consult dermatologist promptly"
    ],

    "Benign_Keratosis": [
        "Avoid scratching or picking lesions",
        "Maintain good skin hygiene",
        "Use sunscreen outdoors"
    ],

    "Bowen's_Disease": [
        "Avoid sun exposure",
        "Use protective clothing",
        "Seek medical evaluation"
    ],

    "Dermatofibroma": [
        "Avoid irritation to affected area",
        "Monitor for size changes",
        "Consult doctor if painful"
    ],

    "Discoid_Lupus_Erythematosus": [
        "Avoid direct sun exposure",
        "Use broad-spectrum sunscreen",
        "Wear protective clothing"
    ],

    "Eczema": [
        "Keep skin moisturized regularly",
        "Avoid harsh soaps and detergents",
        "Use lukewarm water for bathing",
        "Avoid scratching affected areas",
        "Wear soft cotton clothing"
    ],

    "Herpes_Simplex_Virus": [
        "Avoid touching affected area",
        "Maintain hygiene",
        "Reduce stress",
        "Avoid sharing personal items"
    ],

    "Herpes_Zoster": [
        "Keep rash area clean and dry",
        "Avoid scratching blisters",
        "Rest adequately",
        "Consult doctor if severe pain"
    ],

    "Impetigo": [
        "Maintain skin hygiene",
        "Avoid scratching",
        "Do not share towels or clothes",
        "Wash hands frequently"
    ],

    "Lichen_Planus": [
        "Avoid scratching affected areas",
        "Maintain good oral hygiene if involved",
        "Reduce stress levels"
    ],

    "Malignant_Melanoma": [
        "Avoid direct sun exposure",
        "Use high SPF sunscreen",
        "Monitor moles for changes",
        "Seek immediate medical evaluation"
    ],

    "Molluscum_Contagiosum": [
        "Avoid scratching bumps",
        "Do not share towels",
        "Maintain skin hygiene"
    ],

    "Psoriasis": [
        "Keep skin moisturized",
        "Manage stress",
        "Avoid smoking and alcohol",
        "Controlled sunlight exposure"
    ],

    "Rosacea": [
        "Avoid spicy food and alcohol",
        "Protect skin from sun",
        "Use gentle skincare products",
        "Avoid extreme temperatures"
    ],

    "Seborrheic_Dermatitis": [
        "Use mild medicated shampoos",
        "Avoid harsh hair products",
        "Maintain scalp hygiene"
    ],

    "Seborrheic_Keratosis": [
        "Avoid irritation of lesions",
        "Use sunscreen regularly",
        "Consult doctor if growth changes"
    ],

    "Skin_Tag": [
        "Avoid friction to affected area",
        "Maintain hygiene",
        "Consult doctor if irritated"
    ],

    "Tinea_Corporis": [
        "Keep affected area clean and dry",
        "Avoid sharing clothes or towels",
        "Wear breathable fabrics"
    ],

    "Tinea_Pedis": [
        "Keep feet dry",
        "Wear breathable footwear",
        "Avoid walking barefoot in public areas"
    ],

    "Urticaria": [
        "Identify and avoid triggers",
        "Avoid scratching",
        "Reduce stress"
    ],

    "Vitiligo": [
        "Protect skin from sunburn",
        "Use sunscreen daily",
        "Maintain balanced nutrition"
    ]
}

DEFAULT_REMEDY = [
    "Maintain good skin hygiene",
    "Avoid harsh chemicals and irritants",
    "Keep skin moisturized",
    "Consult a dermatologist if symptoms persist"
]

# ===============================
# IMAGE UPLOAD
# ===============================
st.header("Upload Skin Image")

uploaded = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, width=350)

    if st.button("Predict"):
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()

        predicted_disease = CLASS_NAMES[pred_idx]

        st.success(f"Predicted Disease: {predicted_disease}")

        remedies = REMEDIES.get(predicted_disease, DEFAULT_REMEDY)

        st.markdown("### Home Care & General Tips:")
        for tip in remedies:
            st.markdown(f"- {tip}")

        st.warning("⚠ This is an AI-based educational prediction and not a medical diagnosis.")