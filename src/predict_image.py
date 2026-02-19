import os
import sys
import cv2
import joblib
import numpy as np
from feature_extraction import extract_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


def main():
    if len(sys.argv) != 2:
        print("Usage: py src/predict_image.py <path_to_image>")
        return

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print("File tidak ditemukan.")
        return

    image = cv2.imread(image_path)

    if image is None:
        print("Gagal membaca gambar.")
        return

    image = cv2.resize(image, (256, 256))

    features = extract_features(image)
    features = np.array(features).reshape(1, -1)

    model = joblib.load(MODEL_PATH)

    prediction = model.predict(features)[0]

    label = "REAL" if prediction == 0 else "AI"

    print("\n=== HASIL PREDIKSI ===")
    print(f"Gambar: {image_path}")
    print(f"Prediksi: {label}")


if __name__ == "__main__":
    main()
