import streamlit as st
from PIL import Image
import numpy as np

from predict import predict_image

st.set_page_config(page_title="Plant Disease Detector", layout="wide")

st.title("🌿 Plant Disease Detector")
st.markdown("Upload a plant image and detect diseases using AI.")

st.divider()

st.sidebar.title("ℹ️ About")
st.sidebar.write(
    "This AI model detects plant diseases from leaf images."
)

st.header("📥 Upload Image")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "png", "jpeg"])


if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button("🔍 Analyze Plant"):
            with st.spinner("Analyzing image... 🌿"):
                disease, confidence = predict_image(image)

            st.divider()

            st.header("📊 Results")

            score = confidence * 100

            st.subheader("🦠 Detected Disease")
            st.write(f"### {disease}")

            st.subheader("📈 Confidence")
            st.progress(int(score))
            st.write(f"{score:.2f}% confidence")

            if score > 80:
                st.success("✅ High confidence prediction")
            elif score > 50:
                st.warning("⚠️ Moderate confidence")
            else:
                st.error("❌ Low confidence")