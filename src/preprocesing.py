import os
import cv2

IMAGE_SIZE = (256, 256)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")


def preprocess_image(input_path, output_path):
    image = cv2.imread(input_path)
    if image is None:
        return

    image = cv2.resize(image, IMAGE_SIZE)
    cv2.imwrite(output_path, image)


def process_folder(split, label):
    input_dir = os.path.join(DATASET_DIR, split, label)
    output_dir = os.path.join(OUTPUT_DIR, split, label)

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            preprocess_image(input_path, output_path)


def main():
    for split in ["train", "test"]:
        for label in ["ai", "real"]:
            process_folder(split, label)


if __name__ == "__main__":
    main()
