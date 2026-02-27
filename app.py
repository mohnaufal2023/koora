import streamlit as st
import joblib
import cv2
import numpy as np
from src.feature_extraction import extract_features

# ==========================
# Load Model
# ==========================
model = joblib.load("model.pkl")

st.set_page_config(page_title="Deteksi Gambar AI", layout="centered")

st.title("🖼️ Deteksi Gambar AI vs Real")
st.write("Upload gambar untuk mengetahui apakah gambar tersebut AI atau REAL.")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert ke OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Gambar Uploaded", use_column_width=True)

    # Resize (WAJIB sama seperti training)
    image = cv2.resize(image, (256, 256))

    # Ekstraksi fitur
    features = extract_features(image)
    features = np.array(features).reshape(1, -1)

    # Prediksi
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    label = "AI 🤖" if prediction == 1 else "REAL 📷"
    confidence = max(probabilities) * 100

    st.subheader(f"Hasil Prediksi: {label}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Progress bar confidence
    st.progress(int(confidence))