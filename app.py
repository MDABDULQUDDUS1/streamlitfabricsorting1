"""
Fabric Classification Streamlit App
Classifies fabric images as Knit or Woven using a CustomizedDenseNet121 model
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import requests
import io
from model_definition import CustomizedDenseNet121

# Configuration
MODEL_PART1_URL = "https://github.com/MDABDULQUDDUS1/streamlitfabricsorting1/raw/main/densenet121_part1.pth"
MODEL_PART2_URL = "https://github.com/MDABDULQUDDUS1/streamlitfabricsorting1/raw/main/densenet121_part2.pth"
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['Knit', 'Woven']

# Page configuration
st.set_page_config(
    page_title="Fabric Classification",
    page_icon="🧵",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .prediction-box { padding: 2rem; border-radius: 10px; margin: 1rem 0; text-align: center; }
    .knit-box { background-color: #e3f2fd; border: 3px solid #2196f3; }
    .woven-box { background-color: #fff3e0; border: 3px solid #ff9800; }
    .prediction-label { font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem; }
    .confidence-label { font-size: 1.5rem; color: #555; }
    .info-box { background-color: #f5f5f5; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classification_model():
    """Load the trained model from two split files on GitHub"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Download part 1
        r1 = requests.get(MODEL_PART1_URL)
        part1 = torch.load(io.BytesIO(r1.content), map_location=device)

        # Download part 2
        r2 = requests.get(MODEL_PART2_URL)
        part2 = torch.load(io.BytesIO(r2.content), map_location=device)

        # Merge state_dicts
        full_state_dict = {**part1, **part2}

        # Load model
        model = CustomizedDenseNet121(num_classes=2)
        model.load_state_dict(full_state_dict)
        model.eval()
        model.to(device)

        return model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


def get_image_transforms():
    """Transforms used during training"""
    return transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_fabric(image, model, device, transform):
    """Predict fabric type from an image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_percent = confidence.item() * 100

    probs_dict = {CLASS_NAMES[i]: probabilities[0][i].item() * 100 for i in range(len(CLASS_NAMES))}

    return predicted_class, confidence_percent, probs_dict


def main():
    """Streamlit app main function"""
    st.markdown('<h1 class="main-header">🧵 Fabric Classification System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a fabric image to classify it as Knit or Woven</p>', unsafe_allow_html=True)

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This app classifies fabrics as **Knit** or **Woven** using CustomizedDenseNet121.")

        st.header("📊 Model Info")
        st.write("""
        - Architecture: DenseNet121
        - Classes: Knit, Woven
        - Input size: 224x224
        """)

        st.header("🔍 How to Use")
        st.write("""
        1. Upload a JPG/PNG image
        2. Model predicts fabric type
        3. View confidence and probabilities
        """)

    # Load model
    with st.spinner("Loading model..."):
        model, device = load_classification_model()
        transform = get_image_transforms()
    device_name = "GPU" if device.type == "cuda" else "CPU"
    st.success(f"✅ Model loaded successfully on {device_name}")

    # Upload image
    st.markdown("---")
    st.header("📤 Upload Fabric Image")
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg','jpeg','png'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image.convert('RGB'), use_container_width=True)
            st.caption(f"Size: {image.size[0]}x{image.size[1]} pixels")

        with st.spinner("🔍 Analyzing fabric..."):
            predicted_class, confidence, probabilities = predict_fabric(image, model, device, transform)

        with col2:
            st.subheader("🎯 Prediction Results")
            box_class = "knit-box" if predicted_class == "Knit" else "woven-box"
            emoji = "🧶" if predicted_class == "Knit" else "🪡"
            st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <div class="prediction-label">{emoji} {predicted_class}</div>
                    <div class="confidence-label">Confidence: {confidence:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("---")
        st.subheader("📊 Probability Breakdown")
        col_knit, col_woven = st.columns(2)
        with col_knit:
            st.metric(label="🧶 Knit", value=f"{probabilities['Knit']:.2f}%")
            st.progress(probabilities['Knit']/100)
        with col_woven:
            st.metric(label="🪡 Woven", value=f"{probabilities['Woven']:.2f}%")
            st.progress(probabilities['Woven']/100)

    else:
        st.info("👆 Please upload a fabric image to get started")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align:center; color:#888; font-size:0.9rem;">
        Fabric Classification System v1.0 | 🧵 Knit vs. Woven
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
