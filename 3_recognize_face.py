import cv2
import os
 
TRAINER_PATH = os.path.join("trainer", "trainer.yml")
LABELS_PATH = "labels.txt"
 
# Lower = stricter match required to unlock. Tune this for your setup.
# Typical usable range is 45-70 depending on lighting/camera quality.
CONFIDENCE_THRESHOLD = 60
 
def load_labels(path):
    labels = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                pid, pname = line.strip().split(",", 1)
                labels[int(pid)] = pname
    return labels
 
def main():
    if not os.path.exists(TRAINER_PATH):
        print("[ERROR] No trained model found. Run 2_train_model.py first.")
        return
 
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)
 
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)
 
    labels = load_labels(LABELS_PATH)
 
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
 
    print("[INFO] Starting recognition. Press 'q' to quit.")
 
    while True:
        ok, frame = cam.read()
        if not ok:
            break
 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
 
        unlocked = False
 
        for (x, y, w, h) in faces:
            face_crop = gray[y:y + h, x:x + w]
            face_crop = cv2.resize(face_crop, (200, 200))
 
            person_id, confidence = recognizer.predict(face_crop)
            # confidence: 0 = perfect match, higher = less similar
 
            if confidence < CONFIDENCE_THRESHOLD and person_id in labels:
                name = labels[person_id]
                unlocked = True
                color = (0, 200, 0)
                text = f"{name} ({round(100 - confidence, 1)}% match) - UNLOCKED"
            else:
                color = (0, 0, 255)
                text = "Unknown - LOCKED"
 
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
 
        # Big banner like the iPhone lock screen animation
        banner_color = (0, 200, 0) if unlocked else (0, 0, 255)
        banner_text = "UNLOCKED" if unlocked else "LOCKED"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), banner_color, -1)
        cv2.putText(frame, banner_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
 
        cv2.imshow("Face ID Demo - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cam.release()
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()
 