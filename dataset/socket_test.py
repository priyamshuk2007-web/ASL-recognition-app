import cv2
import mediapipe as mp
import socket
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

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8)

landmarker = HandLandmarker.create_from_options(options)

# Connect to Java, same as your socket test
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 5000))
print("Connected to Java!")

camera = cv2.VideoCapture(0)
frame_count = 0

while True:
    success, frame = camera.read()
    if not success:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_count += 1
    result = landmarker.detect_for_video(mp_image, frame_count)

    if len(result.hand_landmarks) >= 2:
            client.sendall(b"BOTH\n")   # tells Java "two hands in this frame"
            cv2.putText(frame, "BOTH", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif len(result.hand_landmarks) == 1:
            hand = result.hand_landmarks[0]

            normalized = normalize_landmarks(hand)
            line = ",".join(str(v) for v in normalized) + "\n"

            client.sendall(line.encode())
            cv2.putText(frame, "SENT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
            client.sendall(b"NONE\n")   # tells Java "no hand in this frame"
    cv2.imshow("Sending landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
client.close()