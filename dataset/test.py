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
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3)
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
    def extract_geometric_features(X_63):
        """
        Takes (N, 63) normalized landmarks and appends 14 relational distance features
        to separate M/N/S/T/A/E, C/O, and U/V. Returns (N, 77).
        """
        pts = X_63.reshape(-1, 21, 3)
    
        def dist(i, j):
            return np.linalg.norm(pts[:, i] - pts[:, j], axis=1, keepdims=True)

        # 1. Thumb tip (4) to fingertips (8, 12, 16, 20) -> fixes C vs O and E
        thumb_to_tips = [dist(4, idx) for idx in [8, 12, 16, 20]]
    
        # 2. Thumb tip (4) to middle knuckles / PIP joints (6, 10, 14, 18) -> fixes M, N, S, T, A
        thumb_to_pips = [dist(4, idx) for idx in [6, 10, 14, 18]]
    
        # 3. Adjacent fingertip spread (8-12, 12-16) -> fixes U vs V vs W
        finger_spread = [dist(8, 12), dist(12, 16)]
    
        # 4. Finger curl ratios: tip-to-wrist(0) / MCP-to-wrist(0) for 4 fingers
        curl_ratios = []
        for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]:
            ratio = dist(tip, 0) / (dist(mcp, 0) + 1e-6)
            curl_ratios.append(ratio)

        extra = np.hstack(thumb_to_tips + thumb_to_pips + finger_spread + curl_ratios)
        return np.hstack([X_63, extra])


    features = extract_geometric_features(normalized.reshape(1, 63))

    prediction = model.predict(features)[0]
    total += 1
    if prediction == true_label:
        correct += 1
    print(f"{filename}: true={true_label}, predicted={prediction}, {'correct' if prediction == true_label else 'WRONG'}")

print(f"\nIndependent test accuracy: {correct}/{total} = {correct/total:.2%}")