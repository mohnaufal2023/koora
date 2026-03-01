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

st.set_page_config(page_title="Deteksi Gambar AI", layout="centered")

st.title("🖼️ Deteksi Gambar AI vs Real")
st.write("Upload gambar untuk mengetahui apakah gambar tersebut AI atau REAL.")

# ==========================
# Fungsi Histogram RGB
# ==========================
def plot_rgb_histogram(image):
    colors = ('b', 'g', 'r')
    fig, ax = plt.subplots()

    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=color)
        ax.set_xlim([0, 256])

    ax.set_title("Histogram Warna RGB")
    ax.set_xlabel("Intensitas Piksel")
    ax.set_ylabel("Jumlah Piksel")

    return fig


# ==========================
# Fungsi Histogram Grayscale
# ==========================
def plot_gray_histogram(gray):
    fig, ax = plt.subplots()

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    ax.plot(hist, color='black')
    ax.set_xlim([0, 256])

    ax.set_title("Histogram Grayscale (Distribusi Intensitas Piksel)")
    ax.set_xlabel("Intensitas Piksel")
    ax.set_ylabel("Jumlah Piksel")

    return fig


uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Gambar Uploaded", use_container_width=True)

    # ==============================
    # Histogram RGB
    # ==============================
    st.subheader("📊 Histogram Warna RGB")
    fig_rgb = plot_rgb_histogram(image)
    st.pyplot(fig_rgb)

    # Resize sesuai training
    image = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ==============================
    # Histogram Grayscale
    # ==============================
    st.subheader("📈 Distribusi Intensitas Piksel (Grayscale)")
    fig_gray = plot_gray_histogram(gray)
    st.pyplot(fig_gray)

    # ==============================
    # Visualisasi Canny Edge
    # ==============================
    st.subheader("🧩 Visualisasi Deteksi Tepi (Canny)")
    edges = cv2.Canny(gray, 100, 200)
    st.image(edges, caption="Hasil Canny Edge Detection", use_container_width=True)

    edge_ratio = np.sum(edges > 0) / edges.size

    # ==============================
    # Hitung GLCM (Transparansi)
    # ==============================
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

    # ==============================
    # Tampilkan Nilai Fitur
    # ==============================
    st.subheader("🔎 Nilai Fitur yang Dihitung")

    st.write(f"Contrast     : {contrast:.4f}")
    st.write(f"Correlation  : {correlation:.4f}")
    st.write(f"Energy       : {energy:.4f}")
    st.write(f"Homogeneity  : {homogeneity:.4f}")
    st.write(f"Edge Ratio   : {edge_ratio:.4f}")

    # ==============================
    # Ekstraksi fitur untuk model
    # ==============================
    features = extract_features(image)
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    label = "AI 🤖" if prediction == 1 else "REAL 📷"
    confidence = max(probabilities) * 100

    st.subheader(f"🎯 Hasil Prediksi: {label}")
    st.write(f"Confidence: {confidence:.2f}%")
    st.progress(int(confidence))