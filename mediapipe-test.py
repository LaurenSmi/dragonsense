import csv
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision


BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "anna-perg.mp4"
MODEL_PATH = BASE_DIR / "pose_landmarker_full.task"
OUTPUT_VIDEO_PATH = BASE_DIR / "annotated_video.mp4"
OUTPUT_CSV_PATH = BASE_DIR / "elbow_angles.csv"
OUTPUT_PLOT_PATH = BASE_DIR / "elbow_angle_plot.png"

MIN_VISIBILITY = 0.5
ELBOW_LANDMARKS = {
    "left": (11, 13, 15),
    "right": (12, 14, 16),
}
ELBOW_COLORS = {
    "left": (255, 80, 80),
    "right": (80, 160, 255),
}

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def calculate_elbow_angle(shoulder, elbow, wrist):
    """Return the shoulder-elbow-wrist angle in degrees, or None if unreliable."""
    if any(landmark.visibility < MIN_VISIBILITY for landmark in (shoulder, elbow, wrist)):
        return None

    upper_arm = np.array([shoulder.x - elbow.x, shoulder.y - elbow.y])
    forearm = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
    lengths = np.linalg.norm(upper_arm) * np.linalg.norm(forearm)
    if lengths == 0:
        return None

    cosine = np.dot(upper_arm, forearm) / lengths
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    pose_landmark_style = vision.drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = vision.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in detection_result.pose_landmarks:
        vision.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style,
        )

        for side, (shoulder_index, elbow_index, wrist_index) in ELBOW_LANDMARKS.items():
            shoulder = pose_landmarks[shoulder_index]
            elbow = pose_landmarks[elbow_index]
            wrist = pose_landmarks[wrist_index]
            angle = calculate_elbow_angle(shoulder, elbow, wrist)
            if angle is None:
                continue

            height, width = annotated_image.shape[:2]
            elbow_point = (int(elbow.x * width), int(elbow.y * height))
            cv2.circle(annotated_image, elbow_point, 8, ELBOW_COLORS[side], -1)
            cv2.putText(
                annotated_image,
                f"{side.title()} elbow: {angle:.1f} deg",
                (elbow_point[0] + 10, elbow_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                ELBOW_COLORS[side],
                2,
                cv2.LINE_AA,
            )

    return annotated_image


def save_angle_plot(angle_samples, fps):
    width, height = 1280, 720
    graph = np.full((height, width, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 100, 70, 1230, 620

    cv2.putText(graph, "Elbow angle over time", (left, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 30, 30), 2, cv2.LINE_AA)

    max_time = angle_samples[-1][0] if angle_samples else 0
    max_time = max(max_time, 1 / fps)

    for angle in range(0, 181, 30):
        y = bottom - int((angle / 180) * (bottom - top))
        cv2.line(graph, (left, y), (right, y), (220, 220, 220), 1)
        cv2.putText(graph, str(angle), (48, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)

    for tick in range(6):
        x = left + int((tick / 5) * (right - left))
        tick_time = max_time * tick / 5
        cv2.line(graph, (x, top), (x, bottom), (235, 235, 235), 1)
        cv2.putText(graph, f"{tick_time:.1f}", (x - 18, bottom + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)

    cv2.line(graph, (left, top), (left, bottom), (40, 40, 40), 2)
    cv2.line(graph, (left, bottom), (right, bottom), (40, 40, 40), 2)
    cv2.putText(graph, "Time (seconds)", (left + 450, bottom + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(graph, "Angle (degrees)", (15, top - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2, cv2.LINE_AA)

    for side, color in ELBOW_COLORS.items():
        previous_point = None
        side_index = 1 if side == "left" else 2
        for sample in angle_samples:
            time_seconds, angle = sample[0], sample[side_index]
            if angle is None:
                previous_point = None
                continue

            x = left + int((time_seconds / max_time) * (right - left))
            y = bottom - int((angle / 180) * (bottom - top))
            point = (x, y)
            if previous_point is not None:
                cv2.line(graph, previous_point, point, color, 2, cv2.LINE_AA)
            previous_point = point

    cv2.putText(graph, "Left elbow", (right - 250, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, ELBOW_COLORS["left"], 2, cv2.LINE_AA)
    cv2.putText(graph, "Right elbow", (right - 110, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, ELBOW_COLORS["right"], 2, cv2.LINE_AA)

    if not cv2.imwrite(str(OUTPUT_PLOT_PATH), graph):
        raise RuntimeError(f"Could not save elbow angle plot to {OUTPUT_PLOT_PATH}")


def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Input video not found: {VIDEO_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Pose model not found: {MODEL_PATH}")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        output_segmentation_masks=False,
    )

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_width <= 0 or frame_height <= 0:
        cap.release()
        raise RuntimeError(f"Input video has invalid metadata: {VIDEO_PATH}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH), fourcc, fps, (frame_width, frame_height)
    )
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {OUTPUT_VIDEO_PATH}")

    angle_samples = []
    frame_index = 0
    print("Processing video and tracking elbow angles...")

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            with OUTPUT_CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["time_seconds", "left_elbow_degrees", "right_elbow_degrees"])

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    time_seconds = frame_index / fps
                    frame_timestamp_ms = int(time_seconds * 1000)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                    left_angle = None
                    right_angle = None
                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks[0]
                        left_angle = calculate_elbow_angle(
                            landmarks[11], landmarks[13], landmarks[15]
                        )
                        right_angle = calculate_elbow_angle(
                            landmarks[12], landmarks[14], landmarks[16]
                        )
                        annotated_frame = draw_landmarks_on_image(rgb_frame, result)
                        output_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    else:
                        output_frame = frame

                    angle_samples.append((time_seconds, left_angle, right_angle))
                    csv_writer.writerow([
                        f"{time_seconds:.6f}",
                        "" if left_angle is None else f"{left_angle:.2f}",
                        "" if right_angle is None else f"{right_angle:.2f}",
                    ])
                    out.write(output_frame)
                    frame_index += 1
    finally:
        cap.release()
        out.release()

    if not angle_samples:
        raise RuntimeError(f"No frames could be read from input video: {VIDEO_PATH}")

    save_angle_plot(angle_samples, fps)
    print(f"Annotated video saved to {OUTPUT_VIDEO_PATH}")
    print(f"Elbow angle data saved to {OUTPUT_CSV_PATH}")
    print(f"Elbow angle plot saved to {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()
