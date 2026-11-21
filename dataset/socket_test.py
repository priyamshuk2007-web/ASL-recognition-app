import time
import sys
import struct
import cv2
import mediapipe as mp
import socket
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 21 hand skeleton joint connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (0, 17)                                   # Palm base
]

def normalize_landmarks(hand_landmarks, is_left_hand=False):
    points = np.array([[p.x, p.y, p.z] for p in hand_landmarks], dtype=np.float32)
    points = points - points[0]
    scale = np.linalg.norm(points[9])
    if scale < 1e-6:
        scale = 1e-6
    points = points / scale
    if is_left_hand:
        # Mirror x-coordinates for opposite (left) hand to match the right-hand model
        points[:, 0] = -points[:, 0]
    return points.flatten()

def draw_hand_skeleton(frame, hand_landmarks, is_left=False):
    h, w, _ = frame.shape
    coords = []
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))

    # Bone connections
    bone_color = (147, 51, 234) if is_left else (16, 185, 129)  # Purple for Left, Emerald for Right
    for p1, p2 in HAND_CONNECTIONS:
        cv2.line(frame, coords[p1], coords[p2], (220, 220, 220), 2)

    # Joint points
    for i, (cx, cy) in enumerate(coords):
        is_tip = i in (4, 8, 12, 16, 20)
        joint_color = (0, 215, 255) if is_tip else bone_color
        radius = 5 if is_tip else 4
        cv2.circle(frame, (cx, cy), radius, joint_color, -1)
        cv2.circle(frame, (cx, cy), radius + 1, (0, 0, 0), 1)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmarker = HandLandmarker.create_from_options(options)

# Connection with retry loop
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
connected = False
for attempt in range(30):
    try:
        client.connect(('localhost', 5000))
        connected = True
        break
    except (ConnectionRefusedError, OSError):
        time.sleep(0.5)

if not connected:
    print("Could not connect to Java server on port 5000.")
    sys.exit(1)

print("Connected to Java!")

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("ERROR: Could not open camera (index 0).")
    try:
        client.close()
    except Exception:
        pass
    sys.exit(1)

# Set preferred resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_ts = 0
failed_reads = 0

try:
    while True:
        success, frame = camera.read()
        if not success:
            failed_reads += 1
            if failed_reads > 50:
                print("ERROR: Lost connection to webcam.")
                break
            time.sleep(0.01)
            continue
        failed_reads = 0

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        ts = int(time.time() * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts

        result = landmarker.detect_for_video(mp_image, ts)
        num_hands = len(result.hand_landmarks)

        meta_str = "NONE"

        if num_hands >= 2:
            meta_str = "BOTH"
            for hand in result.hand_landmarks:
                draw_hand_skeleton(frame, hand, is_left=False)
            # HUD overlay on frame
            cv2.rectangle(frame, (10, 10), (320, 50), (0, 0, 0), -1)
            cv2.putText(frame, "BOTH HANDS (SPACE)", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)
        elif num_hands == 1:
            hand = result.hand_landmarks[0]
            hand_label = "Right"
            if result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
                hand_label = result.handedness[0][0].category_name

            is_left = (hand_label == "Left")
            draw_hand_skeleton(frame, hand, is_left=is_left)

            normalized = normalize_landmarks(hand, is_left_hand=is_left)
            csv_str = ",".join(f"{v:.6f}" for v in normalized)
            meta_str = f"HAND:{hand_label}:{csv_str}"

            # Visual overlay
            status_text = f"{hand_label} Hand{' [Mirrored]' if is_left else ''}"
            text_color = (255, 150, 0) if is_left else (16, 220, 100)
            cv2.rectangle(frame, (10, 10), (330, 50), (20, 20, 20), -1)
            cv2.putText(frame, status_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)
        else:
            meta_str = "NONE"
            cv2.rectangle(frame, (10, 10), (280, 50), (20, 20, 20), -1)
            cv2.putText(frame, "No hand detected", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 2)

        # Compress frame as JPEG (quality 75 gives crisp visuals with tiny payload ~30KB)
        ret, jpeg_buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret:
            continue
        jpeg_bytes = jpeg_buf.tobytes()

        # Pack packet: [4-byte meta_len][4-byte jpeg_len][meta_bytes][jpeg_bytes]
        meta_bytes = meta_str.encode('utf-8')
        packet = struct.pack('>II', len(meta_bytes), len(jpeg_bytes)) + meta_bytes + jpeg_bytes

        try:
            client.sendall(packet)
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("Java connection closed.")
            break

finally:
    camera.release()
    try:
        client.close()
    except Exception:
        pass
