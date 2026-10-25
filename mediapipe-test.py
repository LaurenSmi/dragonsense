import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_dir = Path(r"C:\Users\Lauren\Capstone")
video_path = str(base_dir / "anna-perg.mp4")

# Using the downloaded model bundle path
model_path = str(base_dir / "pose_landmarker_full.task")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  pose_landmark_style = vision.drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = vision.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in pose_landmarks_list:
    vision.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  return annotated_image

# Configure PoseLandmarker Options
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=True
)

# Initialize VideoCapture
cap = cv2.VideoCapture(video_path)

# Fetch video metadata
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter to save output
output_path = str(base_dir / "annotated_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print("Processing video frames...")

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame from OpenCV BGR format to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Get current frame's timestamp in milliseconds
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Run pose detection for the current video frame
        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # Annotate landmarks
        if pose_landmarker_result.pose_landmarks:
            annotated_frame = draw_landmarks_on_image(rgb_frame, pose_landmarker_result)
            # Convert back to BGR for OpenCV VideoWriter
            output_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        else:
            output_frame = frame

        # Write annotated frame to output file
        out.write(output_frame)

cap.release()
out.release()
print(f"Finished processing! Annotated video saved to {output_path}.")