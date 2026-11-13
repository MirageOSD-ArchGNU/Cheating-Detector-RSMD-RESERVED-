#!/usr/bin/env python3
"""
MirageIME WEBCHEAT - Real-time AI Proctoring & Cheating Detection Engine
Optimized for Linux / CachyOS - Multi-Human, Hand Tracking, Head Pose, & Persistent Phone Alerts
"""

import cv2
import sys
import time
import argparse
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

class MirageWebCheatEngine:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.35):
        self.conf_thresh = conf_thresh
        print("[+] Initializing MirageIME WEBCHEAT Pro Engine with Persistent Warnings...")

        # Persistent alert timer variables (keeps red warning active for 4 seconds after trigger)
        self.last_suspicious_time = 0
        self.alert_hold_duration = 4.0

        # Initialize YOLOv8 for human & phone/object detection
        self.yolo_model = None
        if HAS_YOLO:
            try:
                self.yolo_model = YOLO(model_path)
                print("[+] YOLOv8 model loaded successfully.")
            except Exception as e:
                print(f"[!] Warning: YOLO model failed to load ({e}).")

        # Initialize MediaPipe Solutions Safely
        self.mp_hands = None
        self.mp_face_mesh = None
        self.mp_draw = None

        if HAS_MEDIAPIPE and hasattr(mp, 'solutions'):
            self.mp_draw = mp.solutions.drawing_utils
            try:
                if hasattr(mp.solutions, 'hands'):
                    self.mp_hands = mp.solutions.hands.Hands(
                        max_num_hands=4,
                        min_detection_confidence=0.4,
                        min_tracking_confidence=0.4
                    )
                    print("[+] MediaPipe Hands (finger tracking) initialized.")
            except Exception as e:
                print(f"[!] MediaPipe Hands init error: {e}")

            try:
                if hasattr(mp.solutions, 'face_mesh'):
                    self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                        max_num_faces=3,
                        refine_landmarks=True,
                        min_detection_confidence=0.4,
                        min_tracking_confidence=0.4
                    )
                    print("[+] MediaPipe FaceMesh & Head Tracking initialized.")
            except Exception as e:
                print(f"[!] MediaPipe FaceMesh init error: {e}")

    def process_frame(self, frame):
        h, w, _ = frame.shape
        current_time = time.time()
        cheating_score = 0.0
        status_text = "STATUS: NORMAL"
        alerts = []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        human_count = 0
        boxes_to_draw = [] # Format: (xmin, ymin, xmax, ymax, label, color)

        # 1. Run YOLOv8 Object Detection (Explicit Phone & Prohibited Items Tracking)
        if self.yolo_model is not None:
            results = self.yolo_model(frame, verbose=False, conf=self.conf_thresh)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.yolo_model.names[cls_id]
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)

                    if cls_name == 'person':
                        human_count += 1
                        boxes_to_draw.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], f"Person #{human_count} ({conf:.2f})", (0, 255, 0)))
                    elif cls_name in ['cell phone', 'mobile']:
                        cheating_score += 0.8  # Immediate high penalty for phone use
                        alerts.append(f"PHONE DETECTED! ({conf:.2f})")
                        boxes_to_draw.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], f"PHONE DETECTED! ({conf:.2f})", (0, 0, 255)))
                    elif cls_name in ['laptop', 'book', 'tablet', 'remote']:
                        cheating_score += 0.5
                        alerts.append(f"Prohibited Object: {cls_name}")
                        boxes_to_draw.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], f"SUS: {cls_name}", (0, 0, 255)))

        if human_count > 1:
            cheating_score += 0.6 * (human_count - 1)
            alerts.append(f"Multiple Humans Detected ({human_count})")

        # 2. Run MediaPipe Face Mesh & Head Pose Estimation (Yaw / Pitch tracking)
        if self.mp_face_mesh is not None:
            face_results = self.mp_face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    face_3d = []
                    face_2d = []

                    landmark_indices = [1, 33, 263, 61, 291, 152]
                    for idx in landmark_indices:
                        lm = face_landmarks.landmark[idx]
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z * 3000])

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = w
                    cam_matrix = np.array([
                        [focal_length, 0, w / 2],
                        [0, focal_length, h / 2],
                        [0, 0, 1]
                    ])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    if success:
                        rmat, _ = cv2.Rodrigues(rot_vec)
                        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                        pitch = angles[0] * 360
                        yaw = angles[1] * 360

                        if yaw < -22.0:
                            cheating_score += 0.4
                            alerts.append("Head Turned Left (Looking Away)")
                        elif yaw > 22.0:
                            cheating_score += 0.4
                            alerts.append("Head Turned Right (Looking Away)")
                        elif pitch < -18.0:
                            cheating_score += 0.3
                            alerts.append("Head Looking Down")
                        elif pitch > 20.0:
                            cheating_score += 0.3
                            alerts.append("Head Looking Up")

                    if self.mp_draw and hasattr(mp.solutions, 'face_mesh'):
                        self.mp_draw.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
                        )

        # 3. Run MediaPipe Hands & Finger Tracking
        if self.mp_hands is not None:
            hand_results = self.mp_hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                num_hands = len(hand_results.multi_hand_landmarks)
                if num_hands > 2:
                    cheating_score += 0.25
                    alerts.append(f"Multiple Hands ({num_hands})")

                for hand_landmarks in hand_results.multi_hand_landmarks:
                    if self.mp_draw and hasattr(mp.solutions, 'hands'):
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=3),
                            self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                        )

                    wrist_y = hand_landmarks.landmark[0].y
                    if wrist_y < 0.35:
                        cheating_score += 0.3
                        alerts.append("Hand Raised / Suspicious Gesture")

        # Clamp score between 0.0 and 1.0
        cheating_score = min(1.0, max(0.0, cheating_score))

        # Persistent Warning Logic: Keep warning active for 4 seconds after trigger
        if cheating_score >= 0.4 or alerts or human_count > 1:
            self.last_suspicious_time = current_time

        is_warning_active = (current_time - self.last_suspicious_time) < self.alert_hold_duration

        # Determine Box Color & Status
        if is_warning_active:
            global_box_color = (0, 0, 255)  # Bright Red
            status_text = "STATUS: SUSPICIOUS / CHEATING DETECTED!"
        else:
            global_box_color = (0, 255, 0)  # Green
            status_text = "STATUS: NORMAL"

        # Draw all detected human/object bounding boxes
        if boxes_to_draw:
            for (xmin, ymin, xmax, ymax, label, col) in boxes_to_draw:
                draw_col = (0, 0, 255) if is_warning_active and "Person" in label else col
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), draw_col, 3)
                cv2.putText(frame, label, (xmin, max(20, ymin - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_col, 2)
        else:
            cv2.rectangle(frame, (10, 10), (w - 10, h - 10), global_box_color, 2)

        # Draw HUD Header Banner
        overlay_bg = frame.copy()
        cv2.rectangle(overlay_bg, (15, 15), (w - 15, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay_bg, 0.65, frame, 0.35, 0, frame)

        display_score = max(cheating_score, 0.95 if is_warning_active else 0.0)
        cv2.putText(frame, status_text, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, global_box_color, 2)
        cv2.putText(frame, f"Cheating Probability: {display_score*100:.1f}% | Humans: {human_count}", (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if alerts:
            alert_str = " | ".join(alerts[:2])
            cv2.putText(frame, f"Alerts: {alert_str}", (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.putText(frame, "Press 'S' to save snapshot | 'Q' to quit", (25, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

def main():
    parser = argparse.ArgumentParser(description="MirageIME WEBCHEAT Proctoring Engine")
    parser.add_argument("--source", type=int, default=0, help="Webcam device index")
    parser.add_argument("--width", type=int, default=1280, help="Frame width (Default HD)")
    parser.add_argument("--height", type=int, default=720, help="Frame height (Default HD)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    print(f"[+] Opening camera source #{args.source}...")
    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[!] Error opening webcam {args.source}. Trying index 2...")
        cap = cv2.VideoCapture(2)
        if not cap.isOpened():
            print("[!] Error: Could not open any camera device.")
            sys.exit(1)

    engine = MirageWebCheatEngine(model_path=args.model, conf_thresh=args.conf)

    # Create resizable HD window
    window_name = "MirageIME WEBCHEAT - AI Proctoring"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    print("[+] MirageIME WEBCHEAT running. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[!] Stream ended or frame drop.")
            break

        processed_frame = engine.process_frame(frame)
        cv2.imshow(window_name, processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            filename = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"[+] Saved snapshot to {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("[+] Session closed.")

if __name__ == "__main__":
    main()
