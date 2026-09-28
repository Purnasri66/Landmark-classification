import streamlit as st
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import (
    preprocess_input,
    decode_predictions
)

st.set_page_config(
    page_title="Landmark Image Classification",
    page_icon="🏛️",
    layout="centered"
)

st.title("🏛️ Landmark Image Classification")
st.write("Image classification using pre-trained ResNet50")

@st.cache_resource
def load_model():
    return ResNet50(weights="imagenet")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a landmark image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Classify Image"):

        # Resize image
        resized = image.resize((224, 224))

        # Convert to NumPy array
        array = np.asarray(resized, dtype=np.float32)

        # Add batch dimension
        batch = np.expand_dims(array, axis=0)

        # ResNet50 preprocessing
        processed = preprocess_input(batch)

        # Prediction
        predictions = model.predict(
            processed,
            verbose=0
        )

        # Decode top-5 predictions
        results = decode_predictions(
            predictions,
            top=5
        )[0]

        st.subheader("Top-5 Predictions")

        for rank, (_, label, probability) in enumerate(
            results,
            start=1
        ):
            st.write(
                f"**{rank}. {label.replace('_', ' ').title()}** "
                f"— {probability * 100:.2f}%"
            )
