import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# -------------------------------------------------
# CONFIG FOR ALL MODELS
# -------------------------------------------------

MODEL_CONFIGS = {
    "Model 1 – 5 Diseases": {
        "num_classes": 5,
        "model_path": "model 1/models/skin_disease_model.pth",
        "classes": [
            "Acne_Vulgaris",
            "Actinic_solar_Damage(Actinic_Keratosis)",
            "Basal_Cell_Carcinoma",
            "Eczema",
            "Psoriasis"
        ]
    },

    "Model 2 – 10 Diseases": {
        "num_classes": 10,
        "model_path": "model 2/models/skin_disease_model.pth",
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

    "Model 3 – 23 Diseases": {
        "num_classes": 23,
        "model_path": "model 3/models/skin_disease_model.pth",
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

# -------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="AI DermaScope",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #ecfeff, #f0fdf4);
}
.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}
.center {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# MODEL LOADER (CACHED)
# -------------------------------------------------

@st.cache_resource
def load_model(model_path, num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# -------------------------------------------------
# IMAGE TRANSFORM
# -------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------------------------------
# UI – HOME PAGE
# -------------------------------------------------

st.markdown('<div class="card center">', unsafe_allow_html=True)
st.title("🩺 AI DermaScope")
st.subheader("Integrated Skin Disease Detection System")
st.markdown("""
Select a model based on the number of skin diseases you want to classify.
""")
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# MODEL SELECTION
# -------------------------------------------------

selected_model_name = st.selectbox(
    "Select Detection Model",
    list(MODEL_CONFIGS.keys())
)

config = MODEL_CONFIGS[selected_model_name]

st.info(
    f"**{selected_model_name} selected**  \n"
    f"Number of classes: {config['num_classes']}"
)

# -------------------------------------------------
# LOAD SELECTED MODEL
# -------------------------------------------------

model = load_model(
    config["model_path"],
    config["num_classes"]
)

CLASS_NAMES = config["classes"]

# -------------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("🔍 Predict Disease"):
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            pred_idx = torch.argmax(outputs, dim=1).item()

        predicted_disease = CLASS_NAMES[pred_idx]

        st.success(f"**Predicted Disease:** {predicted_disease}")
        st.warning("⚠ This system is for educational and research purposes only.")
