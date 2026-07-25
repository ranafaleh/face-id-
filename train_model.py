"""
2_train_model.py
-----------------
Trains an LBPH (Local Binary Patterns Histograms) face recognizer on
every image inside dataset/<name>/*.jpg.

LBPH works well for small, personal datasets like this one (unlike deep
learning models which need thousands of images). It compares the texture
pattern of a face rather than raw pixels, which makes it fairly robust
to lighting changes.

Usage:
    python 2_train_model.py
"""

import cv2
import numpy as np
import os
from PIL import Image

DATASET_DIR = "dataset"
TRAINER_PATH = os.path.join("trainer", "trainer.yml")

def get_images_and_labels(dataset_dir):
    face_samples = []
    ids = []

    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        for file_name in os.listdir(person_dir):
            if not file_name.lower().endswith((".jpg", ".png")):
                continue

            # filename format: <id>.<sample_number>.jpg
            person_id = int(file_name.split(".")[0])

            img_path = os.path.join(person_dir, file_name)
            pil_img = Image.open(img_path).convert("L")  # grayscale
            img_np = np.array(pil_img, "uint8")

            face_samples.append(img_np)
            ids.append(person_id)

    return face_samples, ids

def main():
    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print("[ERROR] No dataset found. Run 1_register_face.py first.")
        return

    print("[INFO] Reading images and training model...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = get_images_and_labels(DATASET_DIR)

    if len(faces) == 0:
        print("[ERROR] No face images found in dataset.")
        return

    recognizer.train(faces, np.array(ids))
    os.makedirs("trainer", exist_ok=True)
    recognizer.save(TRAINER_PATH)

    print(f"[INFO] Trained on {len(faces)} images for {len(set(ids))} person(s).")
    print(f"[INFO] Model saved to '{TRAINER_PATH}'.")

if __name__ == "__main__":
    main()
