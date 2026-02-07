import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# ---------------- CONFIG ----------------
NUM_CLASSES = 23
MODEL_PATH = "models/skin_disease_model.pth"

CLASS_NAMES = [
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

DISEASE_INFO = {
    name: f"{name.replace('_',' ')} is a dermatological condition detected using deep learning analysis."
    for name in CLASS_NAMES
}

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="AI DermaScope - Model 3", page_icon="🩺")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg,#f0fdfa,#ecfeff); }
.card {
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0 8px 25px rgba(0,0,0,0.1);
}
.center { text-align:center; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- UI ----------------
st.markdown('<div class="card center">', unsafe_allow_html=True)
st.title("🩺 AI DermaScope")
st.subheader("Model 3 – 23 Skin Diseases")
st.markdown('</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("🔍 Predict"):
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            pred_idx = torch.argmax(outputs, dim=1).item()

        predicted_disease = CLASS_NAMES[pred_idx]

        st.success(f"**Predicted Disease:** {predicted_disease}")
        st.info(DISEASE_INFO[predicted_disease])
        st.warning("⚠ Educational use only. Not a medical diagnosis.")
