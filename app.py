import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import pandas as pd

# Page setup
st.set_page_config(page_title="AI Scene Classifier", page_icon="🖼️", layout="wide")

# Custom CSS for a polished look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('scene_classifier_model.h5')
    with open('labels.json', 'r') as f:
        labels = json.load(f)
    return model, labels

# Sidebar
st.sidebar.title("Project Details")
st.sidebar.info("This DL model classifies images into Indoor/Outdoor sub-categories using MobileNetV2.")
st.sidebar.markdown("---")
st.sidebar.write("✅ **Dataset:** Custom Images")
st.sidebar.write("✅ **Framework:** TensorFlow / Keras")

# Main UI
st.title("🖼️ Smart Scene Recognition")
st.write("Upload an image below to identify the environment.")

try:
    model, labels = load_assets()
    
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Input Image", use_container_width=True)

    with col2:
        if uploaded_file:
            st.subheader("Classification Results")
            with st.spinner('AI is thinking...'):
                # Preprocessing
                img = image.resize((224, 224))
                img_array = np.array(img.convert('RGB')) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                preds = model.predict(img_array)[0]
                top_idx = np.argmax(preds)
                
                # Format label: "Outdoor/forest" -> "Outdoor - Forest"
                final_label = labels[str(top_idx)].replace("\\", "/").replace("/", " - ").title()
                
                # Big Result Display
                st.metric(label="Predicted Scene", value=final_label)
                st.progress(float(preds[top_idx]))
                st.write(f"Confidence: **{preds[top_idx]*100:.2f}%**")

                # Confidence Chart
                st.markdown("---")
                st.write("### Confidence Distribution")
                chart_data = pd.DataFrame({
                    'Category': [labels[str(i)].split('/')[-1].title() for i in range(len(labels))],
                    'Confidence': preds
                }).sort_values(by='Confidence', ascending=False)
                st.bar_chart(chart_data.set_index('Category'))

except Exception as e:
    st.warning("⚠️ Waiting for Model... Make sure you have run `train.py` and have `scene_classifier_model.h5` and `labels.json` in your folder.")