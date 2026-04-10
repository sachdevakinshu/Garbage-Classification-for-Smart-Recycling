
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

MODEL_PATH = "models/mobilenet_v2.h5"   # Change to custom_cnn.h5 if that won!
CLASSES    = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
IMG_SIZE   = (224, 224)

TIPS = {
    'Cardboard': '♻️ Flatten and place in dry recycling bin. Remove tape.',
    'Glass':     '🫙 Rinse and place in glass bin. Dont mix with ceramics.',
    'Metal':     '🥫 Rinse tin cans. Aluminium foil is recyclable when clean.',
    'Paper':     '📄 Keep dry. Greasy paper (like pizza boxes) goes to compost.',
    'Plastic':   '🧴 Check the number. Types 1 & 2 are usually accepted.',
    'Trash':     '🗑️ General waste — cannot be recycled. Reduce usage!'
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB').resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)   # shape (1, 224, 224, 3)

st.set_page_config(
    page_title="♻️ Smart Garbage Classifier",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Smart Garbage Classifier")
st.markdown("Upload an image of your waste and the model will identify the correct **recycling category**.")
st.markdown("---")

model = load_model()

uploaded_file = st.file_uploader(
    "📸 Upload a garbage image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("Classifying..."):
            img_array = preprocess_image(image)
            predictions = model.predict(img_array)[0]
            predicted_idx = np.argmax(predictions)
            predicted_class = CLASSES[predicted_idx]
            confidence = predictions[predicted_idx] * 100

        st.markdown(f"### 🏷️ Predicted Category")
        st.success(f"**{predicted_class}**")
        st.metric("Confidence", f"{confidence:.1f}%")

        st.markdown(f"#### 💡 Recycling Tip")
        st.info(TIPS[predicted_class])

    st.markdown("---")
    st.markdown("#### 📊 Prediction Probabilities")
    prob_dict = {CLASSES[i]: float(predictions[i]) for i in range(len(CLASSES))}
    st.bar_chart(prob_dict)