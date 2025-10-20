"""
Fabric Classification Streamlit App
Classifies fabric images as Knit or Woven using a CustomizedDenseNet121 model
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model_definition import load_model

# Configuration
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
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .knit-box {
        background-color: #e3f2fd;
        border: 3px solid #2196f3;
    }
    .woven-box {
        background-color: #fff3e0;
        border: 3px solid #ff9800;
    }
    .prediction-label {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .confidence-label {
        font-size: 1.5rem;
        color: #555;
    }
    .info-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classification_model():
    """Load the trained model (cached)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = load_model(model_dir=".", device=device)
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


def get_image_transforms():
    """Get the same transforms used during model training (validation mode)"""
    return transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def predict_fabric(image, model, device, transform):
    """
    Predict fabric type from image
    Returns: predicted_class, confidence_percent, probabilities
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_percent = confidence.item() * 100
    probs_dict = {
        CLASS_NAMES[i]: probabilities[0][i].item() * 100
        for i in range(len(CLASS_NAMES))
    }

    return predicted_class, confidence_percent, probs_dict


def main():
    """Main Streamlit app"""

    # Header
    st.markdown('<h1 class="main-header">🧵 Fabric Classification System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a fabric image to classify it as Knit or Woven</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This application uses a **CustomizedDenseNet121** deep learning model
        to classify fabric images as either **Knit** or **Woven**.
        """)

        st.header("📊 Model Information")
        st.write("""
        - **Architecture**: DenseNet121  
        - **Fine-tuning**: Progressive Unfreezing  
        - **Input Size**: 224x224 pixels  
        - **Classes**: Knit, Woven
        """)

        st.header("🔍 How to Use")
        st.write("""
        1. Upload a fabric image (JPG, JPEG, or PNG)  
        2. The model will analyze the image  
        3. View the prediction and confidence score  
        4. Check the probability breakdown
        """)

        st.header("💡 Tips")
        st.write("""
        - Use clear, well-lit images  
        - Ensure fabric texture is visible  
        - Crop to focus on the fabric  
        - Higher confidence = more reliable prediction
        """)

    # Load model
    with st.spinner("Loading model..."):
        model, device = load_classification_model()
        transform = get_image_transforms()

    device_name = "GPU" if device.type == "cuda" else "CPU"
    st.success(f"✅ Model loaded successfully on {device_name}")

    # File uploader
    st.markdown("---")
    st.header("📤 Upload Fabric Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the fabric"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, use_container_width=True)
            st.caption(f"Size: {image.size[0]}x{image.size[1]} pixels")

        # Prediction
        with st.spinner("🔍 Analyzing fabric..."):
            predicted_class, confidence, probabilities = predict_fabric(
                image, model, device, transform
            )

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
            st.metric("🧶 Knit", f"{probabilities['Knit']:.2f}%")
            st.progress(probabilities['Knit'] / 100)
        with col_woven:
            st.metric("🪡 Woven", f"{probabilities['Woven']:.2f}%")
            st.progress(probabilities['Woven'] / 100)

        # Technical details
        st.markdown("---")
        st.markdown(f"""
            <div class="info-box">
                <h4>🔬 Technical Details</h4>
                <ul>
                    <li><b>Prediction:</b> {predicted_class}</li>
                    <li><b>Confidence:</b> {confidence:.2f}%</li>
                    <li><b>Model:</b> CustomizedDenseNet121</li>
                    <li><b>Device:</b> {device_name}</li>
                    <li><b>Image Size:</b> {image.size[0]}x{image.size[1]} → Resized to 224x224</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        # Confidence messages
        if confidence >= 90:
            st.success("🎯 **Very High Confidence** - The model is very certain about this prediction.")
        elif confidence >= 75:
            st.info("👍 **High Confidence** - The model is quite confident about this prediction.")
        elif confidence >= 60:
            st.warning("⚠️ **Moderate Confidence** - The model is somewhat uncertain. Consider checking the image quality.")
        else:
            st.error("❓ **Low Confidence** - The model is uncertain. The image may be unclear or ambiguous.")
    else:
        st.info("👆 Please upload a fabric image to get started")
        st.markdown("---")
        st.subheader("📝 Example Classifications")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                **🧶 Knit Fabrics:**
                - T-shirts
                - Sweaters
                - Socks
                - Jersey fabric
                - Interlooped yarns
            """)
        with col2:
            st.markdown("""
                **🪡 Woven Fabrics:**
                - Denim
                - Canvas
                - Dress shirts
                - Bed sheets
                - Interlaced warp & weft
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #888; font-size: 0.9rem;">
            <p>Fabric Classification System v1.0 | Powered by CustomizedDenseNet121</p>
            <p>🧵 Knit vs. Woven Classification using Deep Learning</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
