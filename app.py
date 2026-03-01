import streamlit as st
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from src.feature_extraction import extract_features

# ==========================
# Load Model
# ==========================
model = joblib.load("model.pkl")

st.set_page_config(page_title="Deteksi Gambar AI", layout="wide")

st.title("🖼️ Sistem Deteksi Gambar AI vs Real")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image_resized = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    col1, col2 = st.columns(2)

    # ======================
    # KOLOM 1 (Gambar & Edge)
    # ======================
    with col1:
        st.subheader("📷 Gambar")
        st.image(image, use_container_width=True)

        st.subheader("🧩 Canny Edge")
        edges = cv2.Canny(gray, 100, 200)
        st.image(edges, use_container_width=True)

    # ======================
    # KOLOM 2 (Histogram)
    # ======================
    with col2:
        st.subheader("📊 Histogram RGB")

        fig_rgb, ax = plt.subplots()
        colors = ('b', 'g', 'r')

        for i, color in enumerate(colors):
            hist = cv2.calcHist([image_resized], [i], None, [256], [0, 256])
            ax.plot(hist, color=color)

        ax.set_xlim([0, 256])
        st.pyplot(fig_rgb)

        st.subheader("📈 Histogram Grayscale")

        fig_gray, ax2 = plt.subplots()
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
        ax2.plot(hist_gray, color='black')
        ax2.set_xlim([0, 256])
        st.pyplot(fig_gray)

    st.markdown("---")

    # ======================
    # Fitur Tekstur
    # ======================
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0][0]
    correlation = graycoprops(glcm, 'correlation')[0][0]
    energy = graycoprops(glcm, 'energy')[0][0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0][0]
    edge_ratio = np.sum(edges > 0) / edges.size

    st.subheader("🔎 Nilai Fitur yang Digunakan")

    col3, col4, col5 = st.columns(3)

    col3.metric("Contrast", f"{contrast:.2f}")
    col3.metric("Correlation", f"{correlation:.3f}")

    col4.metric("Energy", f"{energy:.4f}")
    col4.metric("Homogeneity", f"{homogeneity:.3f}")

    col5.metric("Edge Ratio", f"{edge_ratio:.3f}")

    st.markdown("---")

    # ======================
    # Prediksi
    # ======================
    features = extract_features(image_resized)
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    label = "AI 🤖" if prediction == 1 else "REAL 📷"
    confidence = max(probabilities) * 100

    st.subheader("🎯 Hasil Prediksi")
    st.success(f"{label} (Confidence: {confidence:.2f}%)")

    st.progress(int(confidence))

    # Probabilitas detail
    st.subheader("📊 Probabilitas Kelas")
    st.bar_chart({
        "REAL": probabilities[0],
        "AI": probabilities[1]
    })