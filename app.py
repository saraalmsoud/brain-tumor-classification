import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import base64
from io import BytesIO

# Function to convert images to base64 for display
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Set background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #EBF4FA; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the model
model = tf.keras.models.load_model("model1.keras")

# Define tumor classes (Ensure order matches model output)
classes = ["glioma", "meningioma", "notumor", "pituitary"]

# Preprocess image function
def preprocess_image(image):
    img = image.resize((224, 224))
    img = img.convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# UI Enhancements with CSS
st.markdown("""
    <style>
        .stFileUploader {
            border: 2px dashed #1f77b4 !important;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            background-color: #f9f9f9;
        }
        .stButton > button {
            background-color: #1f77b4 !important;
            color: white !important;
            border-radius: 10px;
            font-size: 18px;
            padding: 10px 20px;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #0056b3 !important;
        }
        .title-container {
            background-color: #1f77b4;
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin: 20px 0px 40px 0px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }
        .result-box {
            background-color: #f1f8ff;
            border-left: 5px solid #1f77b4;
            padding: 15px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            margin-top: 20px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)
# Add the main title
st.markdown('<div class="title-container">🧠 Brain Tumor Classification with AI</div>', unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("📤 Upload a Brain MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/jpeg;base64,{image_to_base64(image)}" alt="Uploaded Image" width="400">
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🔍 Analyze Image"):
        with st.spinner("🔄 Analyzing image..."):
            time.sleep(2)  # Simulating processing time
            
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)

            # Convert output to NumPy if needed
            if isinstance(prediction, tf.Tensor):
                prediction = prediction.numpy()

            # Extract predicted class and confidence
            predicted_class_index = np.argmax(prediction)
            predicted_class = classes[predicted_class_index]
            confidence = np.max(prediction) * 100  # Confidence percentage

            # 📌 Display results as elegant cards
            st.markdown("### 🔍 Analysis Results:")

            for i, cls in enumerate(classes):
                prob = prediction[0][i] * 100  # Convert to percentage
                
                # Highlight the predicted class
                card_color = "#1f77b4" if cls == predicted_class else "#f1f1f1"
                text_color = "white" if cls == predicted_class else "black"
            # Styled result cards
                st.markdown(
                    f"""
                    <div style="
                        background-color: {card_color};
                        color: {text_color};
                        padding: 15px;
                        border-radius: 10px;
                        margin: 10px 0;
                        text-align: center;
                        font-size: 18px;
                        font-weight: bold;
                        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                    ">
                        {cls.upper()} - {prob:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Assign colors and descriptions for each tumor type
            class_descriptions = {
                "glioma": "🔵 Glioma Tumor: A tumor that arises from glial cells in the brain, requiring medical follow-up.",
                "meningioma": "🟠 Meningioma Tumor: Grows in the membranes surrounding the brain and is usually benign.",
                "notumor": "✅ No Tumor Detected: The image shows no signs of a tumor.",
                "pituitary": "🟢 Pituitary Tumor: Affects the pituitary gland and may cause hormonal changes."
            }

            # Display result inside a styled box
            st.markdown(f'<div class="result-box">{class_descriptions[predicted_class]}</div>', unsafe_allow_html=True)

            # Display confidence score
            st.markdown(f"🎯 Confidence in Diagnosis: {confidence:.2f}%")
            st.progress(float(confidence) / 100)  # Convert to float64