import os
import cv2
import csv
import numpy as np
from skimage.feature import graycomatrix, graycoprops


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "features")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_features(image, bins=32):
    features = []

    # ======================
    # 1️⃣ Fitur Warna (RGB)
    # ======================
    for i in range(3):  # B, G, R
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    # ======================
    # 2️⃣ Fitur Tekstur (GLCM)
    # ======================
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

    features.extend([contrast, correlation, energy, homogeneity])

    # ======================
    # 3️⃣ Fitur Tepi (Canny)
    # ======================
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.sum(edges > 0) / edges.size
    features.append(edge_ratio)

    # ======================
    # 4️⃣ Distribusi Intensitas (Grayscale Histogram)
    # ======================
    gray_hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    gray_hist = cv2.normalize(gray_hist, gray_hist).flatten()
    features.extend(gray_hist)

    return features



def process_dataset(split, csv_name):
    rows = []

    for label_name, label_value in [("ai", 1), ("real", 0)]:
        folder = os.path.join(PROCESSED_DIR, split, label_name)

        for file in os.listdir(folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(folder, file)
                image = cv2.imread(path)

                if image is None:
                    continue

                features = extract_features(image)

                features.append(label_value)
                rows.append(features)

    csv_path = os.path.join(OUTPUT_DIR, csv_name)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main():
    process_dataset("train", "train_features.csv")
    process_dataset("test", "test_features.csv")


if __name__ == "__main__":
    main()
