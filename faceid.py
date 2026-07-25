import cv2
import os
import argparse
import numpy as np
 
def augment(face_img):
    """Generate a handful of variations of one face crop."""
    variants = [face_img]
 
    # horizontal flip
    variants.append(cv2.flip(face_img, 1))
 
    # slight rotations
    h, w = face_img.shape
    center = (w // 2, h // 2)
    for angle in (-10, -5, 5, 10):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face_img, matrix, (w, h))
        variants.append(rotated)
 
    # brightness changes
    for beta in (-30, 30):
        bright = cv2.convertScaleAbs(face_img, alpha=1.0, beta=beta)
        variants.append(bright)
 
    # slight blur (simulates camera softness/distance)
    variants.append(cv2.GaussianBlur(face_img, (3, 3), 0))
 
    return variants
 
def main():
    parser = argparse.ArgumentParser(description="Register a face from an image file.")
    parser.add_argument("--image", required=True, nargs="+", help="Path(s) to source image(s) (jpg/png). Pass one or several.")
    parser.add_argument("--name", required=True, help="Name to associate with this face")
    parser.add_argument("--id", required=True, type=int, help="Unique numeric ID for this face")
    args = parser.parse_args()
 
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)
 
    save_dir = os.path.join("dataset", args.name)
    os.makedirs(save_dir, exist_ok=True)
 
    # don't overwrite samples already saved for this person in a previous run
    existing_files = [f for f in os.listdir(save_dir) if f.startswith(f"{args.id}.")]
    next_index = len(existing_files) + 1
 
    total_saved = 0
    for image_path in args.image:
        if not os.path.exists(image_path):
            print(f"[WARN] Image not found, skipping: {image_path}")
            continue
 
        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARN] Could not read image, skipping: {image_path}")
            continue
 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
 
        if len(faces) == 0:
            print(f"[WARN] No face detected in '{image_path}', skipping.")
            continue
 
        # use the largest detected face in the image
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = gray[y:y + h, x:x + w]
        face_crop = cv2.resize(face_crop, (200, 200))
 
        variants = augment(face_crop)
        for variant in variants:
            file_path = os.path.join(save_dir, f"{args.id}.{next_index}.jpg")
            cv2.imwrite(file_path, variant)
            next_index += 1
            total_saved += 1
 
        print(f"[INFO] '{image_path}': detected face, saved {len(variants)} training samples.")
 
    if total_saved == 0:
        print("[ERROR] No samples were saved. Nothing to train on.")
        return
 
    print(f"[INFO] Total {total_saved} training samples now in '{save_dir}'.")
 
    # update labels.txt same as the webcam version
    labels_file = "labels.txt"
    existing = {}
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            for line in f:
                pid, pname = line.strip().split(",", 1)
                existing[int(pid)] = pname
    existing[args.id] = args.name
    with open(labels_file, "w") as f:
        for pid, pname in sorted(existing.items()):
            f.write(f"{pid},{pname}\n")
 
    print("[INFO] Now run: python 2_train_model.py")
 
if __name__ == "__main__":
    main()
 
