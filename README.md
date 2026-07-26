# Face ID Clone — OpenCV Face Recognition

# A simplified, iPhone-Face-ID-style face recognition system built with OpenCV. Instead of scanning a live face, this version registers a face from a single photo (in this case, a picture of Lana Del Rey), trains a recognizer on it, then unlocks a live webcam feed only when that same photo is held up to the camera.

# How this differs from real Face ID:
Apple's Face ID uses an infrared depth-sensing camera (TrueDepth) to build a 3D map of a face, which is what makes it hard to fool with a photo. This project uses a normal 2D webcam and a single reference photo, so it's a learning demo, not a secure authentication system.

# Files in this repo

# 1-faceid.py
Enrollment. Takes a source image (# lana del ray.jpg), detects the face in it with a Haar Cascade classifier, then generates 9 augmented variants (flip, small rotations, brightness changes) so there's enough data to train on from just one photo. Saves them to # dataset/ folder (not uploaded here — see note below) and records the name/ID pair in # labels.txt.

# 2-train_model.py
Training. Reads every image in dataset/ and trains an LBPH (Local Binary Patterns Histograms) face recognizer on them. Saves the trained model as trainer.yml.

# 3-3_recognize_face.py
Recognition / unlock. Opens the webcam, detects any face in the live feed, and asks the trained model if it matches. If the match confidence passes a threshold, it shows a green UNLOCKED banner; otherwise, red LOCKED.

# 4-labels.txt
Simple id,name mapping (e.g. 1,LanaDelRey) so the recognizer can turn a numeric prediction back into a name.

# 5-trainer.yml
The trained LBPH model output by train_model.py — this is what 3_recognize_face.py loads to make predictions.

# 6-lana del ray.jpg
The single reference photo used to enroll the face.

# 7-faceid video.mp4
Screen recording of the project running.


# How it works (pipeline)
 1-Enroll — faceid.py finds the face in lana del ray.jpg using a Haar Cascade detector (a classic, lightweight OpenCV face detector, built in — no extra downloads needed), crops it, converts it to grayscale, and creates 9 slightly varied copies of it so the model sees more than just one exact image.

 2-Train — train_model.py feeds all 9 images into an LBPH recognizer. LBPH compares the texture pattern of a face rather than raw pixels, which makes it reasonably tolerant of lighting differences — useful since it needs to work with a small, single-photo dataset.

3-Recognize — 3_recognize_face.py runs the webcam live. For every face it detects, it asks the trained model "how close is this to what you learned?" (a distance score, where lower = more similar). If it's close enough, the screen flashes green UNLOCKED.


# Setup
pip install opencv-contrib-python Pillow numpy

# Usage

# Step 1 — enroll the face from the photo
python faceid.py --image "lana del ray.jpg" --name LanaDelRey --id 1

# Step 2 — train the recognizer
python train_model.py

# Step 3 — run the live webcam unlock demo
python 3_recognize_face.py


# Known limitations
Not secure: since it's 2D-only (no depth sensing like real Face ID), a photo held up to the camera is literally the intended "unlock method" here — this project is a demo of the recognition pipeline, not a security system.

Lighting sensitive: LBPH accuracy can drop in very dark or very bright conditions, or with glare off a phone screen/printed photo.

Single source photo: using more reference photos (different angles/ lighting) of the same face would make the match more robust.
