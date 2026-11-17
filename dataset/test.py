import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def normalize_landmarks(hand_landmarks):
    points = np.array([[p.x, p.y, p.z] for p in hand_landmarks])
    wrist = points[0]
    points = points - wrist
    scale = np.linalg.norm(points[9])
    points = points / scale
    return points.flatten()

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1)
landmarker = HandLandmarker.create_from_options(options)

model = joblib.load('asl_letter_model.pkl')

test_folder = 'asl_alphabet_test/asl_alphabet_test'  # adjust if nested differently
correct = 0
total = 0

for filename in os.listdir(test_folder):
    # Expecting filenames like "A_test.jpg" — true label is the first character
    if filename.lower().startswith(('space', 'nothing', 'del')):
        continue # we excluded these from training, skip them here too
    true_label = filename[0].upper()
    if true_label in ('J', 'Z'):
        continue
    image = cv2.imread(os.path.join(test_folder, filename))
    if image is None:
        continue

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = landmarker.detect(mp_image)

    if len(result.hand_landmarks) == 0:
        print(f"{filename}: no hand detected, skipping")
        continue

    normalized = normalize_landmarks(result.hand_landmarks[0])
    prediction = model.predict([normalized])[0]

    total += 1
    if prediction == true_label:
        correct += 1
    print(f"{filename}: true={true_label}, predicted={prediction}, {'correct' if prediction == true_label else 'WRONG'}")

print(f"\nIndependent test accuracy: {correct}/{total} = {correct/total:.2%}")