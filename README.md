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
