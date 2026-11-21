import cv2
import mediapipe as mp
import os
import csv
import numpy as np

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
# IMAGE mode this time, not VIDEO — each photo is independent, no "previous frame" to track from
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,min_hand_detection_confidence= 0.5,
    min_hand_presence_confidence=0.5)

landmarker = HandLandmarker.create_from_options(options)

dataset_path = 'asl_alphabet_train/asl_alphabet_train'  # folder containing A/, B/, C/ ... subfolders
images_per_letter = 3000
letters = [chr(c) for c in range(ord('A'), ord('Z')+1) if chr(c) not in ('J', 'Z')]

output_file = open('landmarks.csv', 'w', newline='')
writer = csv.writer(output_file)

# Header row: label, then x0,y0,z0,x1,y1,z1... for 21 points
header = ['letter']
for i in range(21):
    header += [f'x{i}', f'y{i}', f'z{i}']
writer.writerow(header)

for letter in letters:
    folder = os.path.join(dataset_path, letter)
    files = os.listdir(folder)[:images_per_letter]

    print(f"Processing {letter}: {len(files)} images")

    for filename in files:
        image_path = os.path.join(folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        result = landmarker.detect(mp_image)  # detect(), not detect_for_video() — IMAGE mode

        if len(result.hand_landmarks) == 0:
            continue  # no hand found in this image, skip it

        hand = result.hand_landmarks[0]
        normalized = normalize_landmarks(hand)
        row = [letter] + list(normalized)
        writer.writerow(row)

output_file.close()
print("Done. Saved to landmarks.csv")