import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import telepot
from telepot.loop import MessageLoop
import io
import threading
import tempfile

# ===============================
# TELEGRAM BOT CONFIGURATION
# ===============================
TELEGRAM_TOKEN = "8639367055:AAG2HOZ8156Wc0vYh9D-RYvroYrduRgXdCY"  # Replace with your actual bot token

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI DermaScope",
    layout="wide"
)

# ===============================
# PROFESSIONAL UI STYLE
# ===============================
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(135deg,#e3f2fd,#bbdefb,#90caf9);
}

/* GLOBAL TEXT */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    color:#111;
}

/* MAIN TITLE */
.main-title{
    font-size:60px;
    font-weight:900;
    text-align:center;
    margin-top:20px;
}

/* SUBTITLE */
.subtitle{
    font-size:22px;
    text-align:center;
    margin-bottom:40px;
}

/* SECTION TITLES */
.section{
    font-size:26px;
    font-weight:700;
    margin-top:20px;
}

/* CARD STYLE */
.card{
    background:white;
    padding:35px;
    border-radius:18px;
    box-shadow:0px 10px 35px rgba(0,0,0,0.15);
}

/* BUTTON STYLE */
.stButton > button{
    background:#4a90e2;
    color:white;
    font-size:20px;
    font-weight:600;
    height:60px;
    width:230px;
    border-radius:12px;
    border:none;
}

.stButton > button:hover{
    background:#357ABD;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"]{
    background:white;
    padding:25px;
    border-radius:15px;
}

/* SUCCESS BOX */
.stSuccess{
    font-size:22px;
}

/* WARNING */
.stWarning{
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown('<div class="main-title">🩺 AI DermaScope</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered Skin Disease Detection Platform</div>', unsafe_allow_html=True)

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
            "Acne_Vulgaris","Actinic_solar_Damage(Actinic_Keratosis)","Basal_Cell_Carcinoma",
            "Benign_Keratosis","Bowen's_Disease","Dermatofibroma",
            "Discoid_Lupus_Erythematosus","Eczema","Herpes_Simplex_Virus",
            "Herpes_Zoster","Impetigo","Lichen_Planus",
            "Malignant_Melanoma","Molluscum_Contagiosum","Psoriasis",
            "Rosacea","Seborrheic_Dermatitis","Seborrheic_Keratosis",
            "Skin_Tag","Tinea_Corporis","Tinea_Pedis",
            "Urticaria","Vitiligo"
        ]
    }
}

# ===============================
# MODEL SELECT
# ===============================
st.markdown('<div class="section">Select Detection Model</div>', unsafe_allow_html=True)

model_key = st.radio(
    "",
    list(MODEL_CONFIGS.keys()),
    horizontal=True
)

config = MODEL_CONFIGS[model_key]
st.success(f"Active Model: {model_key}")

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
# IMAGE TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ===============================
# REMEDIES
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
# TELEGRAM BOT HANDLER
# ===============================
def handle_telegram_message(msg):
    """Handle incoming Telegram messages"""
    chat_id = msg['chat']['id']
    
    # Check if message contains a photo
    if 'photo' in msg:
        try:
            # Send processing message
            bot = telepot.Bot(TELEGRAM_TOKEN)
            bot.sendMessage(chat_id, "🔍 Analyzing your image... Please wait.")
            
            # Get the photo file ID (largest size is last)
            photo_id = msg['photo'][-1]['file_id']
            
            # Create a temporary file to save the image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_path = temp_file.name
            
            # Download the photo to the temporary file
            bot.download_file(photo_id, temp_path)
            
            # Open image with PIL
            image = Image.open(temp_path).convert("RGB")
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Perform prediction
            img_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
            
            predicted_disease = CLASS_NAMES[pred_idx]
            remedies = REMEDIES.get(predicted_disease, DEFAULT_REMEDY)
            
            # Format response
            response = f"🩺 *Prediction:* {predicted_disease}\n\n"
            response += "*💊 Care Suggestions:*\n"
            for remedy in remedies:
                response += f"• {remedy}\n"
            response += "\n⚠️ *This AI prediction is educational and not a medical diagnosis.*"
            
            # Send prediction back to same chat
            bot.sendMessage(chat_id, response, parse_mode='Markdown')
            
        except Exception as e:
            bot = telepot.Bot(TELEGRAM_TOKEN)
            bot.sendMessage(chat_id, f"❌ Error processing image: {str(e)}\nPlease try again with a clearer image.")
    
    # Handle text messages
    elif 'text' in msg:
        text = msg['text'].lower()
        bot = telepot.Bot(TELEGRAM_TOKEN)
        
        if text == '/start':
            welcome_msg = (
                "🤖 Welcome to AI DermaScope Bot!\n\n"
                "Send me a skin image and I'll analyze it using AI to detect potential skin conditions.\n\n"
                "Commands:\n"
                "/start - Show this message\n"
                "/help - Get help information\n"
                "/model - Show current active model"
            )
            bot.sendMessage(chat_id, welcome_msg)
        
        elif text == '/help':
            help_msg = (
                "📋 How to use:\n"
                "1. Send a clear image of the skin condition\n"
                "2. I'll analyze it using the active AI model\n"
                "3. You'll receive:\n"
                "   - Predicted condition\n"
                "   - Care suggestions\n\n"
                "⚠️ Note: This is for educational purposes only. Always consult a dermatologist for medical advice."
            )
            bot.sendMessage(chat_id, help_msg)
        
        elif text == '/model':
            bot.sendMessage(chat_id, f"🧠 Current Active Model: {model_key}\n📊 Number of classes: {len(CLASS_NAMES)}")
        
        else:
            bot.sendMessage(chat_id, "Please send an image of the skin condition for analysis.\nUse /help for more information.")

# ===============================
# START TELEGRAM BOT
# ===============================
def run_telegram_bot():
    """Run the Telegram bot"""
    if TELEGRAM_TOKEN != "YOUR_BOT_TOKEN_HERE":
        bot = telepot.Bot(TELEGRAM_TOKEN)
        MessageLoop(bot, handle_telegram_message).run_as_thread()
        return True
    return False

# Start Telegram bot in background thread
bot_running = run_telegram_bot()

if bot_running:
    st.sidebar.success("🤖 Telegram Bot is running!")
    st.sidebar.info(f"Active Model: {model_key}")
else:
    st.sidebar.warning("⚠️ Telegram bot token not configured. Add your bot token to enable Telegram functionality.")

# ===============================
# IMAGE UPLOAD
# ===============================
st.markdown('<div class="section">Upload Skin Image</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["jpg","jpeg","png"])

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image,use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:

        if st.button("🔬 Analyze Image"):

            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs,dim=1)
                pred_idx = torch.argmax(probs,dim=1).item()

            predicted_disease = CLASS_NAMES[pred_idx]

            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.success(f"Prediction: {predicted_disease}")

            remedies = REMEDIES.get(predicted_disease,DEFAULT_REMEDY)

            st.markdown("### Care Suggestions")

            for tip in remedies:
                st.write("•",tip)

            st.warning("This AI prediction is educational and not a medical diagnosis.")

            st.markdown('</div>', unsafe_allow_html=True)