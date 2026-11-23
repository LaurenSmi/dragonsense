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
OUTPUT_TORSO_PLOT_PATH = BASE_DIR / "torso_lean_plot.png"
STROKE_SNAPSHOT_DIR = BASE_DIR / "stroke_snapshots"
STROKE_MOMENTS_CSV_PATH = BASE_DIR / "stroke_moments.csv"

MIN_VISIBILITY = 0.5
FORWARD_IMAGE_DIRECTION = 1  # The paddler in the input video faces right.
ELBOW_LANDMARKS = {
    "left": (11, 13, 15),
    "right": (12, 14, 16),
}
SHOULDER_LANDMARKS = (11, 12)
HIP_LANDMARKS = (23, 24)
ELBOW_COLORS = {
    "left": (255, 80, 80),
    "right": (80, 160, 255),
}

TORSO_COLOR = (50, 180, 50)
STROKE_PHASES = ("Setup", "Catch", "Pull", "Exit", "Recovery")
HEAD_MAX_DOWN_ANGLE_DEGREES = 55
SETUP_MAX_TORSO_LEAN_DEGREES = 15
SETUP_LEFT_HAND_HEIGHT_TOLERANCE_TORSO_RATIO = 0.05
CATCH_BOTTOM_ARM_MIN_DEGREES = 150
RECOVERY_RIGHT_ELBOW_MIN_DEGREES = 140
TOP_HAND_EYE_LEVEL_TOLERANCE_TORSO_RATIO = 0.3
MIN_HAND_HIP_DISTANCE_TORSO_RATIO = 0.55
PADDLE_MIN_POSITIVE_ANGLE_DEGREES = 3
PADDLE_MAX_CLEAR_ANGLE_DEGREES = 75
MIN_BACKWARD_HAND_TRAVEL_FRAME_RATIO = 0.015
MIN_TORSO_BACKWARD_CHANGE_DEGREES = 3
MIN_SIT_UP_CHANGE_DEGREES = 3

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def calculate_elbow_angle(shoulder, elbow, wrist, image_width, image_height):
    """Return the pixel-corrected shoulder-elbow-wrist angle in degrees."""
    if any(landmark.visibility < MIN_VISIBILITY for landmark in (shoulder, elbow, wrist)):
        return None

    # MediaPipe normalizes x by the image width and y by its height. In a
    # non-square frame, one normalized unit therefore represents different
    # pixel distances on each axis. Scale each coordinate independently first
    # so both vector components use the same pixel-based coordinate system.
    shoulder_xy = np.array([shoulder.x * image_width, shoulder.y * image_height])
    elbow_xy = np.array([elbow.x * image_width, elbow.y * image_height])
    wrist_xy = np.array([wrist.x * image_width, wrist.y * image_height])

    # Build both arm-segment vectors from the elbow. The dot-product formula
    # below then gives the included angle at the joint in image pixel space.
    upper_arm = shoulder_xy - elbow_xy
    forearm = wrist_xy - elbow_xy
    lengths = np.linalg.norm(upper_arm) * np.linalg.norm(forearm)
    if lengths == 0:
        return None

    cosine = np.dot(upper_arm, forearm) / lengths
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def visible_midpoint(landmarks, landmark_indices, image_width, image_height):
    """Return the pixel midpoint of visible landmarks, or None if none qualify."""
    visible_points = [
        (landmarks[index].x * image_width, landmarks[index].y * image_height)
        for index in landmark_indices
        if landmarks[index].visibility >= MIN_VISIBILITY
    ]
    if not visible_points:
        return None

    return np.mean(visible_points, axis=0)


def calculate_torso_lean_angle(landmarks, image_width, image_height):
    """Return torso lean in degrees: forward positive, backward negative."""
    shoulder_midpoint = visible_midpoint(
        landmarks, SHOULDER_LANDMARKS, image_width, image_height
    )
    hip_midpoint = visible_midpoint(
        landmarks, HIP_LANDMARKS, image_width, image_height
    )
    if shoulder_midpoint is None or hip_midpoint is None:
        return None, None, None

    # Compare the shoulder midpoint to the hip midpoint in pixel coordinates.
    # atan2 handles either direction of lean; the input video's paddler faces
    # right, so a rightward shoulder displacement is positive (forward).
    delta_x = shoulder_midpoint[0] - hip_midpoint[0]
    delta_y = shoulder_midpoint[1] - hip_midpoint[1]
    angle = math.degrees(math.atan2(
        FORWARD_IMAGE_DIRECTION * delta_x, -delta_y
    ))
    return angle, shoulder_midpoint, hip_midpoint


def calculate_head_down_angle(landmarks, image_width, image_height):
    """Estimate head-down pitch from the ear midpoint toward the nose."""
    nose = landmarks[0]
    if nose.visibility < MIN_VISIBILITY:
        return None

    visible_ears = [
        landmarks[index]
        for index in (7, 8)
        if landmarks[index].visibility >= MIN_VISIBILITY
    ]
    if not visible_ears:
        return None

    # For this right-facing side view, a nose lower than the ear midpoint
    # yields a positive angle; face yaw and occlusion can make this estimate
    # unavailable or imperfect, so it is only a form cue.
    ear_x = sum(ear.x for ear in visible_ears) / len(visible_ears) * image_width
    ear_y = sum(ear.y for ear in visible_ears) / len(visible_ears) * image_height
    nose_x = nose.x * image_width
    nose_y = nose.y * image_height
    forward_dx = FORWARD_IMAGE_DIRECTION * (nose_x - ear_x)
    downward_dy = nose_y - ear_y
    if forward_dx < 1:
        return None
    return math.degrees(math.atan2(downward_dy, forward_dx))


def landmark_pixel_point(landmark, image_width, image_height):
    """Return one visible landmark as an (x, y) pixel point."""
    if landmark.visibility < MIN_VISIBILITY:
        return None
    return landmark.x * image_width, landmark.y * image_height


def calculate_hand_line_angle(left_wrist, right_wrist):
    """Return hand-line slope in degrees; positive slopes down to image-right."""
    if left_wrist is None or right_wrist is None:
        return None

    # Sort endpoints by image x so the signed angle is independent of which
    # anatomical hand is on the left. This representation is -90..+90 degrees;
    # it is a side-view paddle/hand-line proxy, not a 0..180 joint angle.
    leftmost, rightmost = sorted((left_wrist, right_wrist), key=lambda point: point[0])
    delta_x = rightmost[0] - leftmost[0]
    if delta_x < 1:
        return None
    return math.degrees(math.atan2(rightmost[1] - leftmost[1], delta_x))


def add_form_check(checks, label, passed, correction, measurable=True):
    """Append a reviewable pass, correction, or unavailable form check."""
    if not measurable:
        checks.append(("REVIEW", f"{label}: cannot assess from visible landmarks"))
    elif passed:
        checks.append(("OK", f"{label}: looks consistent"))
    else:
        checks.append(("FIX", f"{label}: {correction}"))


def get_top_hand_y(sample):
    """Return the higher (smaller image-y) visible wrist, if available."""
    wrist_points = [point for point in (sample[4], sample[7]) if point is not None]
    return min((point[1] for point in wrist_points), default=None)


def right_wrist_hip_offset(sample):
    """Return signed forward distance from right wrist to hip; negative is past."""
    right_wrist = sample[7]
    hip_point = sample[6]
    if right_wrist is None or hip_point is None:
        return None

    # The paddler faces right, so a wrist farther left than the hip midpoint
    # has crossed behind the hip during the backward pull.
    return FORWARD_IMAGE_DIRECTION * (right_wrist[0] - hip_point[0])


def hands_hip_distance_ratio(sample):
    """Return hand-midpoint to hip distance as a fraction of torso length."""
    hand_midpoint = sample[5]
    hip_midpoint = sample[6]
    torso_length = sample[10]
    if hand_midpoint is None or hip_midpoint is None or not torso_length:
        return None

    return math.hypot(
        hand_midpoint[0] - hip_midpoint[0],
        hand_midpoint[1] - hip_midpoint[1],
    ) / torso_length


def build_phase_checks(angle_samples, moment_by_phase, phase, frame_width, fps):
    """Create phase-specific form cues and concise corrections for a snapshot."""
    frame_index = moment_by_phase[phase]
    sample = angle_samples[frame_index]
    checks = []

    head_down_angle = sample[8]
    add_form_check(
        checks,
        "Head forward/up",
        head_down_angle is not None and head_down_angle <= HEAD_MAX_DOWN_ANGLE_DEGREES,
        "lift the head and look forward",
        measurable=head_down_angle is not None,
    )

    hand_line_angle = sample[11]
    positive_hand_line = (
        hand_line_angle is not None
        and hand_line_angle >= PADDLE_MIN_POSITIVE_ANGLE_DEGREES
    )

    if phase == "Setup":
        # Setup includes the earlier phase definition as well as the new
        # corrections: torso near upright and anatomical left hand near its
        # highest point before the detected catch.
        torso_lean = sample[3]
        add_form_check(
            checks,
            "Upright setup",
            torso_lean is not None
            and abs(torso_lean) <= SETUP_MAX_TORSO_LEAN_DEGREES,
            "sit more upright for setup",
            measurable=torso_lean is not None,
        )

        catch_index = moment_by_phase["Catch"]
        left_wrist_y = sample[4][1] if sample[4] is not None else None
        torso_length = sample[10]
        stroke_left_hand_y = [
            angle_samples[index][4][1]
            for index in range(frame_index, catch_index + 1)
            if angle_samples[index][4] is not None
        ]
        left_hand_high = (
            left_wrist_y is not None
            and torso_length is not None
            and stroke_left_hand_y
            and left_wrist_y
            <= min(stroke_left_hand_y)
            + SETUP_LEFT_HAND_HEIGHT_TOLERANCE_TORSO_RATIO * torso_length
        )
        add_form_check(
            checks,
            "Paddler's left hand high",
            left_hand_high,
            "raise the paddler's left hand toward its highest setup position",
            measurable=left_wrist_y is not None and torso_length is not None
            and bool(stroke_left_hand_y),
        )

        add_form_check(
            checks,
            "Hand-line angle",
            positive_hand_line,
            "set the hands so their line slopes down toward image-right",
            measurable=hand_line_angle is not None,
        )

    elif phase == "Catch":
        # "Bottom arm" means the arm whose wrist is lower in the image. Top
        # hand height is compared with detected eye level, normalized by torso
        # length so the tolerance scales with the paddler's apparent size.
        left_wrist = sample[4]
        right_wrist = sample[7]
        wrists = [
            (side, point)
            for side, point in (("left", left_wrist), ("right", right_wrist))
            if point is not None
        ]
        if wrists:
            bottom_side, _ = max(wrists, key=lambda item: item[1][1])
            bottom_elbow_angle = sample[1] if bottom_side == "left" else sample[2]
        else:
            bottom_elbow_angle = None
        add_form_check(
            checks,
            "Bottom arm",
            bottom_elbow_angle is not None
            and bottom_elbow_angle >= CATCH_BOTTOM_ARM_MIN_DEGREES,
            f"straighten the bottom arm to at least {CATCH_BOTTOM_ARM_MIN_DEGREES} degrees",
            measurable=bottom_elbow_angle is not None,
        )

        top_hand_y = get_top_hand_y(sample)
        eye_point = sample[9]
        torso_length = sample[10]
        hand_near_eye_level = (
            top_hand_y is not None
            and eye_point is not None
            and torso_length is not None
            and abs(top_hand_y - eye_point[1])
            <= TOP_HAND_EYE_LEVEL_TOLERANCE_TORSO_RATIO * torso_length
        )
        add_form_check(
            checks,
            "Top hand at eye level",
            hand_near_eye_level,
            "bring the top hand closer to eye height",
            measurable=(
                top_hand_y is not None and eye_point is not None and torso_length is not None
            ),
        )
        add_form_check(
            checks,
            "Paddle/hand-line angle",
            positive_hand_line,
            "keep the hand line sloping down toward image-right",
            measurable=hand_line_angle is not None,
        )

    elif phase == "Pull":
        # Backward is image-left for this right-facing paddler. Require both a
        # minimum hand displacement and a reduction in forward torso lean.
        hand_hip_ratio = hands_hip_distance_ratio(sample)
        add_form_check(
            checks,
            "Hands stay at least 0.55 torso from hips",
            hand_hip_ratio is not None
            and hand_hip_ratio >= MIN_HAND_HIP_DISTANCE_TORSO_RATIO,
            "keep the hands at least 0.55 torso lengths from the hip point during pull",
            measurable=hand_hip_ratio is not None,
        )

        wrist_offset = right_wrist_hip_offset(sample)
        add_form_check(
            checks,
            "Right wrist is at or forward of hip",
            wrist_offset is not None and wrist_offset >= 0,
            "stop the backward pull when the right wrist reaches the hip",
            measurable=wrist_offset is not None,
        )

        catch_index = moment_by_phase["Catch"]
        start = angle_samples[catch_index]
        hand_start = start[5]
        hand_end = sample[5]
        torso_start = start[3]
        torso_end = sample[3]
        hands_back = (
            hand_start is not None
            and hand_end is not None
            and FORWARD_IMAGE_DIRECTION * (hand_end[0] - hand_start[0])
            <= -MIN_BACKWARD_HAND_TRAVEL_FRAME_RATIO * frame_width
        )
        torso_back = (
            torso_start is not None
            and torso_end is not None
            and torso_end - torso_start <= -MIN_TORSO_BACKWARD_CHANGE_DEGREES
        )
        add_form_check(
            checks,
            "Hands and torso move back together",
            hands_back and torso_back,
            "drive the hands back as the torso returns toward upright",
            measurable=(
                hand_start is not None
                and hand_end is not None
                and torso_start is not None
                and torso_end is not None
            ),
        )

    elif phase == "Exit":
        # A taller torso means less forward lean than at the pull snapshot.
        # Check the paddler's right wrist directly against the hip midpoint;
        # do not rely on the two-hand midpoint, which can hide one wrist going
        # behind the hip when the other hand remains forward.
        pull_lean = angle_samples[moment_by_phase["Pull"]][3]
        exit_lean = sample[3]
        sitting_taller = (
            pull_lean is not None
            and exit_lean is not None
            and exit_lean <= pull_lean - MIN_SIT_UP_CHANGE_DEGREES
        )
        add_form_check(
            checks,
            "Torso sits taller",
            sitting_taller,
            "sit a little taller as the stroke finishes",
            measurable=pull_lean is not None and exit_lean is not None,
        )

        hand_midpoint = sample[5]
        hip_midpoint = sample[6]
        torso_length = sample[10]
        hand_hip_ratio = hands_hip_distance_ratio(sample)
        if hand_midpoint is not None and hip_midpoint is not None and torso_length:
            hands_at_hips = (
                hand_hip_ratio is not None
                and hand_hip_ratio >= MIN_HAND_HIP_DISTANCE_TORSO_RATIO
            )
            hands_not_past_hips = (
                FORWARD_IMAGE_DIRECTION * (hand_midpoint[0] - hip_midpoint[0])
                >= -0.15 * torso_length
            )
            hands_near_hips = hands_at_hips and hands_not_past_hips
            measurable_hands = True
        else:
            hands_near_hips = False
            measurable_hands = False
        add_form_check(
            checks,
            "Hands finish at hips",
            hands_near_hips,
            "finish with hands at the hips, not behind them",
            measurable=measurable_hands,
        )

        wrist_offset = right_wrist_hip_offset(sample)
        add_form_check(
            checks,
            "Right wrist is at or forward of hip",
            wrist_offset is not None and wrist_offset >= 0,
            "finish at the hip; do not pull the right wrist behind it",
            measurable=wrist_offset is not None,
        )

        # The hand-line helper uses a signed -90..+90 degree slope. A positive
        # line is down toward image-right; 180 degrees is not part of this
        # undirected line-angle convention, so reject near-vertical lines.
        paddle_in_exit_range = (
            hand_line_angle is not None
            and PADDLE_MIN_POSITIVE_ANGLE_DEGREES <= hand_line_angle
            <= PADDLE_MAX_CLEAR_ANGLE_DEGREES
        )
        add_form_check(
            checks,
            "Paddle/hand-line angle",
            paddle_in_exit_range,
            "keep a clear positive hand-line slope; avoid a nearly vertical line",
            measurable=hand_line_angle is not None,
        )

    elif phase == "Recovery":
        # MediaPipe's right elbow is anatomical, not whichever elbow appears
        # on image-right. Check the higher wrist's vertical trend over a short
        # recent window to detect a top hand that is dropping.
        right_elbow_angle = sample[2]
        add_form_check(
            checks,
            "Right elbow",
            right_elbow_angle is not None
            and right_elbow_angle >= RECOVERY_RIGHT_ELBOW_MIN_DEGREES,
            f"keep the right elbow at least {RECOVERY_RIGHT_ELBOW_MIN_DEGREES} degrees open",
            measurable=right_elbow_angle is not None,
        )

        torso_lean = sample[3]
        add_form_check(
            checks,
            "Forward torso lean",
            torso_lean is not None and torso_lean > 0,
            "maintain a forward torso lean during recovery",
            measurable=torso_lean is not None,
        )

        exit_index = moment_by_phase["Exit"]
        recent_start = max(exit_index, frame_index - max(1, round(fps * 0.15)))
        top_y_start = get_top_hand_y(angle_samples[recent_start])
        top_y_now = get_top_hand_y(sample)
        torso_length = sample[10]
        top_hand_not_dropping = (
            top_y_start is not None
            and top_y_now is not None
            and torso_length is not None
            and top_y_now - top_y_start <= 0.03 * torso_length
        )
        add_form_check(
            checks,
            "Top hand stays up",
            top_hand_not_dropping,
            "keep the top hand level or rising through recovery",
            measurable=(
                top_y_start is not None and top_y_now is not None and torso_length is not None
            ),
        )

    return checks


def smooth_signal(values, radius):
    """Apply a centered moving average while preserving missing samples."""
    smoothed = []
    for index, value in enumerate(values):
        if value is None:
            smoothed.append(None)
            continue

        neighbors = [
            neighbor
            for neighbor in values[max(0, index - radius):index + radius + 1]
            if neighbor is not None
        ]
        smoothed.append(sum(neighbors) / len(neighbors) if neighbors else None)
    return smoothed


def detect_catch_indices(angle_samples, fps):
    """Find prominent local maxima in torso lean to mark successive catches."""
    torso_angles = [sample[3] for sample in angle_samples]
    smoothed_angles = smooth_signal(torso_angles, max(1, round(fps * 0.08)))
    prominence_window = max(2, round(fps * 0.35))
    minimum_spacing = max(1, round(fps * 0.6))
    candidates = []

    for index in range(1, len(smoothed_angles) - 1):
        angle = smoothed_angles[index]
        if angle is None:
            continue

        previous = smoothed_angles[index - 1]
        following = smoothed_angles[index + 1]
        if previous is None or following is None or angle < previous or angle < following:
            continue

        left_values = [
            value for value in smoothed_angles[max(0, index - prominence_window):index]
            if value is not None
        ]
        right_values = [
            value for value in smoothed_angles[index + 1:index + prominence_window + 1]
            if value is not None
        ]
        if not left_values or not right_values:
            continue

        prominence = angle - max(min(left_values), min(right_values))
        if prominence >= 6:
            candidates.append((index, angle))

    # When two neighboring frames both look like a peak, retain the stronger
    # one. The spacing prevents small tracking wiggles becoming extra strokes.
    selected = []
    for index, angle in sorted(candidates, key=lambda candidate: candidate[1], reverse=True):
        if all(abs(index - kept_index) >= minimum_spacing for kept_index in selected):
            selected.append(index)
    return sorted(selected)


def select_setup_index(angle_samples, start_index, catch_index, smoothed_left_wrist_y):
    """Pick a near-upright frame with the paddler's left hand held high."""
    candidates = [
        index
        for index in range(start_index, catch_index)
        if smoothed_left_wrist_y[index] is not None and angle_samples[index][3] is not None
    ]
    if not candidates:
        return None

    near_upright = [
        index for index in candidates if abs(angle_samples[index][3]) <= 15
    ]
    if near_upright:
        candidates = near_upright
        return min(candidates, key=lambda index: smoothed_left_wrist_y[index])

    # If the paddler never gets within 15 degrees of upright in this interval,
    # prefer the most upright available frame, breaking ties by hand height.
    return min(
        candidates,
        key=lambda index: (
            abs(angle_samples[index][3]),
            smoothed_left_wrist_y[index],
        ),
    )


def select_stroke_moments(angle_samples, fps):
    """Return ordered phase/frame pairs for each complete detected stroke."""
    catch_indices = detect_catch_indices(angle_samples, fps)
    if len(catch_indices) < 2:
        raise RuntimeError(
            "Could not detect at least two catches; no complete stroke can be captured."
        )

    smooth_radius = max(1, round(fps * 0.08))
    left_wrist_y = smooth_signal(
        [sample[4][1] if sample[4] is not None else None for sample in angle_samples],
        smooth_radius,
    )
    hand_x = smooth_signal(
        [sample[5][0] if sample[5] is not None else None for sample in angle_samples],
        smooth_radius,
    )
    hand_y = smooth_signal(
        [sample[5][1] if sample[5] is not None else None for sample in angle_samples],
        smooth_radius,
    )
    hip_points = [sample[6] for sample in angle_samples]
    stroke_intervals = []

    for catch_index, next_catch_index in zip(catch_indices, catch_indices[1:]):
        # Pull is the point after catch where the averaged hands come closest
        # to the hip midpoint.
        pull_candidates = [
            index
            for index in range(catch_index, next_catch_index)
            if hand_x[index] is not None and hip_points[index] is not None
        ]
        if not pull_candidates:
            continue

        pull_index = min(
            pull_candidates,
            key=lambda index: math.hypot(
                angle_samples[index][5][0] - hip_points[index][0],
                angle_samples[index][5][1] - hip_points[index][1],
            ),
        )

        # With the paddler facing right, the hands move left during the pull.
        # Their leftmost position is the end of backward travel.
        backward_candidates = [
            index for index in range(pull_index, next_catch_index)
            if hand_x[index] is not None
        ]
        if not backward_candidates:
            continue
        backward_end = min(backward_candidates, key=lambda index: hand_x[index])

        # Exit is the first point after backward travel stops where the hands
        # also begin rising (image y decreases). Fall back to the reversal
        # point if visibility or motion makes that combined cue unavailable.
        exit_index = backward_end
        for index in range(backward_end + 2, next_catch_index):
            if (
                hand_x[index] is not None
                and hand_x[index - 2] is not None
                and hand_y[index] is not None
                and hand_y[index - 2] is not None
                and hand_x[index] >= hand_x[index - 2] - 3
                and hand_y[index] <= hand_y[index - 2] - 3
            ):
                exit_index = index
                break

        forward_candidates = [
            index for index in range(backward_end, next_catch_index)
            if hand_x[index] is not None
        ]
        if forward_candidates:
            forward_end = max(forward_candidates, key=lambda index: hand_x[index])
            halfway_x = (hand_x[backward_end] + hand_x[forward_end]) / 2
            recovery_index = next(
                (
                    index for index in range(exit_index, forward_end + 1)
                    if hand_x[index] is not None and hand_x[index] >= halfway_x
                ),
                (exit_index + forward_end) // 2,
            )
        else:
            recovery_index = (exit_index + next_catch_index) // 2

        stroke_intervals.append({
            "catch": catch_index,
            "next_catch": next_catch_index,
            "pull": pull_index,
            "exit": exit_index,
            "recovery": recovery_index,
        })

    if not stroke_intervals:
        raise RuntimeError("Could not identify hands and hips for any complete stroke.")

    moments = []
    for cycle_index, interval in enumerate(stroke_intervals, start=1):
        if cycle_index == 1:
            setup_start = 0
        else:
            previous_recovery = stroke_intervals[cycle_index - 2]["recovery"]
            setup_start = previous_recovery + max(1, round(fps * 0.08))
        setup_index = select_setup_index(
            angle_samples, setup_start, interval["catch"], left_wrist_y
        )
        if setup_index is not None:
            moments.append((cycle_index, "Setup", setup_index))
        moments.extend([
            (cycle_index, "Catch", interval["catch"]),
            (cycle_index, "Pull", interval["pull"]),
            (cycle_index, "Exit", interval["exit"]),
            (cycle_index, "Recovery", interval["recovery"]),
        ])

    return moments


def save_stroke_snapshots(angle_samples, fps):
    """Save annotated video frames and a timestamp index for detected phases."""
    moments = select_stroke_moments(angle_samples, fps)
    STROKE_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    expected_snapshot_names = set()
    video = cv2.VideoCapture(str(OUTPUT_VIDEO_PATH))
    if not video.isOpened():
        raise RuntimeError(f"Could not reopen annotated video: {OUTPUT_VIDEO_PATH}")

    try:
        with STROKE_MOMENTS_CSV_PATH.open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            csv_writer = csv.writer(csv_file)
            frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            csv_writer.writerow([
                "cycle",
                "phase",
                "frame_index",
                "time_seconds",
                "torso_lean_degrees_forward_positive",
                "head_down_angle_degrees",
                "right_elbow_degrees",
                "right_wrist_hip_offset_pixels_positive_forward",
                "hands_to_hip_distance_torso_ratio",
                "hand_line_angle_degrees_down_to_image_right",
                "form_checks",
                "corrections",
                "snapshot_path",
            ])

            moments_by_cycle = {}
            for cycle_index, phase, frame_index in moments:
                moments_by_cycle.setdefault(cycle_index, {})[phase] = frame_index

            for cycle_index, phase, frame_index in moments:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = video.read()
                if not success:
                    raise RuntimeError(
                        f"Could not read annotated frame {frame_index} for {phase}."
                    )

                time_seconds = angle_samples[frame_index][0]
                torso_angle = angle_samples[frame_index][3]
                sample = angle_samples[frame_index]
                checks = build_phase_checks(
                    angle_samples,
                    moments_by_cycle[cycle_index],
                    phase,
                    frame_width,
                    fps,
                )
                corrections = [text for status, text in checks if status != "OK"]
                label = f"Cycle {cycle_index:02d} - {phase} - {time_seconds:.2f}s"
                cv2.rectangle(frame, (12, 12), (690, 58), (255, 255, 255), -1)
                cv2.putText(
                    frame,
                    label,
                    (24, 44),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (20, 20, 20),
                    2,
                    cv2.LINE_AA,
                )

                # Keep the phase-specific notes below the live measurement
                # panel already drawn on the right side of the video.
                panel_x = max(20, frame_width - 870)
                panel_y = 470
                line_height = 30
                panel_bottom = panel_y + 52 + line_height * len(checks)
                cv2.rectangle(
                    frame,
                    (panel_x, panel_y),
                    (frame_width - 18, min(panel_bottom, frame.shape[0] - 12)),
                    (255, 255, 255),
                    -1,
                )
                cv2.putText(
                    frame,
                    "Form checks (2D estimates)",
                    (panel_x + 14, panel_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (25, 25, 25),
                    2,
                    cv2.LINE_AA,
                )
                for check_index, (status, text) in enumerate(checks):
                    status_color = {
                        "OK": (30, 130, 30),
                        "FIX": (0, 90, 220),
                        "REVIEW": (0, 150, 200),
                    }[status]
                    cv2.putText(
                        frame,
                        f"[{status}] {text}",
                        (panel_x + 14, panel_y + 58 + check_index * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        status_color,
                        1,
                        cv2.LINE_AA,
                    )

                snapshot_path = STROKE_SNAPSHOT_DIR / (
                    f"cycle_{cycle_index:02d}_{phase.lower()}_"
                    f"frame_{frame_index:04d}.jpg"
                )
                expected_snapshot_names.add(snapshot_path.name)
                if not cv2.imwrite(str(snapshot_path), frame):
                    raise RuntimeError(f"Could not save snapshot: {snapshot_path}")

                csv_writer.writerow([
                    cycle_index,
                    phase,
                    frame_index,
                    f"{time_seconds:.6f}",
                    "" if torso_angle is None else f"{torso_angle:.2f}",
                    "" if sample[8] is None else f"{sample[8]:.2f}",
                    "" if sample[2] is None else f"{sample[2]:.2f}",
                    (
                        ""
                        if right_wrist_hip_offset(sample) is None
                        else f"{right_wrist_hip_offset(sample):.2f}"
                    ),
                    (
                        ""
                        if hands_hip_distance_ratio(sample) is None
                        else f"{hands_hip_distance_ratio(sample):.3f}"
                    ),
                    "" if sample[11] is None else f"{sample[11]:.2f}",
                    " | ".join(f"{status}: {text}" for status, text in checks),
                    " | ".join(corrections),
                    str(snapshot_path),
                ])
    finally:
        video.release()

    # Remove only stale files that match this script's exact snapshot naming
    # pattern; leave any other files the user placed in the output folder.
    valid_phase_names = {phase.lower() for phase in STROKE_PHASES}
    for stale_path in STROKE_SNAPSHOT_DIR.glob("cycle_*_*.jpg"):
        parts = stale_path.stem.split("_")
        is_generated_snapshot = (
            len(parts) == 5
            and parts[0] == "cycle"
            and parts[1].isdigit()
            and parts[2] in valid_phase_names
            and parts[3] == "frame"
            and parts[4].isdigit()
        )
        if is_generated_snapshot and stale_path.name not in expected_snapshot_names:
            stale_path.unlink()

    return moments


def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    image_height, image_width = annotated_image.shape[:2]
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

        torso_angle, shoulder_midpoint, hip_midpoint = calculate_torso_lean_angle(
            pose_landmarks, image_width, image_height
        )
        if torso_angle is not None:
            shoulder_point = tuple(np.rint(shoulder_midpoint).astype(int))
            hip_point = tuple(np.rint(hip_midpoint).astype(int))
            cv2.line(
                annotated_image, hip_point, shoulder_point, TORSO_COLOR, 3, cv2.LINE_AA
            )
            cv2.putText(
                annotated_image,
                f"Torso lean: {torso_angle:+.1f} deg",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                TORSO_COLOR,
                2,
                cv2.LINE_AA,
            )

        for side, (shoulder_index, elbow_index, wrist_index) in ELBOW_LANDMARKS.items():
            shoulder = pose_landmarks[shoulder_index]
            elbow = pose_landmarks[elbow_index]
            wrist = pose_landmarks[wrist_index]
            angle = calculate_elbow_angle(
                shoulder, elbow, wrist, image_width, image_height
            )
            if angle is None:
                continue

            elbow_point = (int(elbow.x * image_width), int(elbow.y * image_height))
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

        left_elbow_angle = calculate_elbow_angle(
            pose_landmarks[11], pose_landmarks[13], pose_landmarks[15],
            image_width, image_height,
        )
        right_elbow_angle = calculate_elbow_angle(
            pose_landmarks[12], pose_landmarks[14], pose_landmarks[16],
            image_width, image_height,
        )
        head_down_angle = calculate_head_down_angle(
            pose_landmarks, image_width, image_height
        )
        left_wrist = landmark_pixel_point(
            pose_landmarks[15], image_width, image_height
        )
        right_wrist = landmark_pixel_point(
            pose_landmarks[16], image_width, image_height
        )
        hand_midpoint = (
            tuple(np.mean([point for point in (left_wrist, right_wrist) if point is not None], axis=0))
            if left_wrist is not None or right_wrist is not None
            else None
        )
        hand_line_angle = calculate_hand_line_angle(left_wrist, right_wrist)
        eye_midpoint = visible_midpoint(
            pose_landmarks, (2, 5), image_width, image_height
        )
        wrist_candidates = [
            (side, point)
            for side, point in (("Left", left_wrist), ("Right", right_wrist))
            if point is not None
        ]
        if wrist_candidates:
            bottom_side, _ = max(wrist_candidates, key=lambda item: item[1][1])
            bottom_arm_angle = (
                left_elbow_angle if bottom_side == "Left" else right_elbow_angle
            )
        else:
            bottom_side, bottom_arm_angle = None, None

        measurements = [
            ("Head down", head_down_angle, f"target <= {HEAD_MAX_DOWN_ANGLE_DEGREES} deg"),
            ("Torso lean", torso_angle, "forward is +"),
            ("Left elbow", left_elbow_angle, f"catch bottom >= {CATCH_BOTTOM_ARM_MIN_DEGREES} deg"),
            ("Right elbow", right_elbow_angle, f"recovery >= {RECOVERY_RIGHT_ELBOW_MIN_DEGREES} deg"),
            (
                "Bottom arm",
                bottom_arm_angle,
                f"{bottom_side or 'unknown'}; catch >= {CATCH_BOTTOM_ARM_MIN_DEGREES} deg",
            ),
            (
                "Hand-line angle",
                hand_line_angle,
                f"setup/catch > {PADDLE_MIN_POSITIVE_ANGLE_DEGREES}; exit < {PADDLE_MAX_CLEAR_ANGLE_DEGREES} deg",
            ),
        ]
        panel_x = max(20, image_width - 520)
        panel_y = 28
        line_height = 26
        panel_height = 48 + line_height * (len(measurements) + 3)
        cv2.rectangle(
            annotated_image,
            (panel_x, panel_y),
            (image_width - 16, min(panel_y + panel_height, image_height - 12)),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            annotated_image,
            "Form measurements (2D)",
            (panel_x + 12, panel_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
        y = panel_y + 58
        for label, value, target in measurements:
            rendered_value = "not visible" if value is None else f"{value:+.1f} deg"
            cv2.putText(
                annotated_image,
                f"{label}: {rendered_value}",
                (panel_x + 12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.49,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_image,
                target,
                (panel_x + 250, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.41,
                (90, 90, 90),
                1,
                cv2.LINE_AA,
            )
            y += line_height

        torso_length = (
            float(np.linalg.norm(shoulder_midpoint - hip_midpoint))
            if shoulder_midpoint is not None and hip_midpoint is not None
            else None
        )
        hip_offset = None
        if right_wrist is not None and hip_midpoint is not None:
            hip_offset = FORWARD_IMAGE_DIRECTION * (
                right_wrist[0] - hip_midpoint[0]
            )
        if hand_midpoint is not None and hip_midpoint is not None and torso_length:
            hand_hip_ratio = math.hypot(
                hand_midpoint[0] - hip_midpoint[0],
                hand_midpoint[1] - hip_midpoint[1],
            ) / torso_length
        else:
            hand_hip_ratio = None
        if hand_midpoint is not None and eye_midpoint is not None and torso_length:
            top_hand_eye_ratio = abs(
                get_top_hand_y((None, None, None, None, left_wrist, None, None, right_wrist))
                - eye_midpoint[1]
            ) / torso_length
        else:
            top_hand_eye_ratio = None

        extra_rows = [
            (
                "Right wrist - hip",
                "not visible" if hip_offset is None else f"{hip_offset:+.0f} px",
                "negative = past hip",
            ),
            (
                "Hands - hip distance",
                "not visible" if hand_hip_ratio is None else f"{hand_hip_ratio:.2f} torso",
                f"pull/exit target >= {MIN_HAND_HIP_DISTANCE_TORSO_RATIO:.2f} torso",
            ),
            (
                "Top hand - eye",
                "not visible" if top_hand_eye_ratio is None else f"{top_hand_eye_ratio:.2f} torso",
                f"catch target <= {TOP_HAND_EYE_LEVEL_TOLERANCE_TORSO_RATIO:.2f}",
            ),
        ]
        for label, value, target in extra_rows:
            cv2.putText(
                annotated_image,
                f"{label}: {value}",
                (panel_x + 12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.49,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_image,
                target,
                (panel_x + 250, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.41,
                (90, 90, 90),
                1,
                cv2.LINE_AA,
            )
            y += line_height

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


def save_torso_lean_plot(angle_samples, fps):
    width, height = 1280, 720
    graph = np.full((height, width, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 100, 70, 1230, 620

    cv2.putText(graph, "Torso lean over time (forward positive)", (left, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)

    max_time = angle_samples[-1][0] if angle_samples else 0
    max_time = max(max_time, 1 / fps)

    for angle in range(-90, 91, 30):
        y = bottom - int(((angle + 90) / 180) * (bottom - top))
        color = (160, 160, 160) if angle == 0 else (220, 220, 220)
        thickness = 2 if angle == 0 else 1
        cv2.line(graph, (left, y), (right, y), color, thickness)
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
    cv2.putText(graph, "Torso lean (degrees)", (15, top - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2, cv2.LINE_AA)

    previous_point = None
    for sample in angle_samples:
        time_seconds, angle = sample[0], sample[3]
        if angle is None:
            previous_point = None
            continue

        x = left + int((time_seconds / max_time) * (right - left))
        y = bottom - int(((angle + 90) / 180) * (bottom - top))
        point = (x, y)
        if previous_point is not None:
            cv2.line(graph, previous_point, point, TORSO_COLOR, 2, cv2.LINE_AA)
        previous_point = point

    if not cv2.imwrite(str(OUTPUT_TORSO_PLOT_PATH), graph):
        raise RuntimeError(f"Could not save torso lean plot to {OUTPUT_TORSO_PLOT_PATH}")


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
                csv_writer.writerow([
                    "time_seconds",
                    "left_elbow_degrees",
                    "right_elbow_degrees",
                    "torso_lean_degrees_forward_positive",
                ])

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
                    torso_lean_angle = None
                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks[0]
                        left_angle = calculate_elbow_angle(
                            landmarks[11], landmarks[13], landmarks[15],
                            frame_width, frame_height,
                        )
                        right_angle = calculate_elbow_angle(
                            landmarks[12], landmarks[14], landmarks[16],
                            frame_width, frame_height,
                        )
                        torso_lean_angle, shoulder_midpoint, hip_midpoint = calculate_torso_lean_angle(
                            landmarks, frame_width, frame_height
                        )
                        head_down_angle = calculate_head_down_angle(
                            landmarks, frame_width, frame_height
                        )
                        eye_midpoint = visible_midpoint(
                            landmarks, (2, 5), frame_width, frame_height
                        )
                        torso_length = (
                            float(np.linalg.norm(shoulder_midpoint - hip_midpoint))
                            if shoulder_midpoint is not None and hip_midpoint is not None
                            else None
                        )
                        left_wrist_point = landmark_pixel_point(
                            landmarks[15], frame_width, frame_height
                        )
                        right_wrist_point = landmark_pixel_point(
                            landmarks[16], frame_width, frame_height
                        )
                        visible_wrist_points = [
                            point for point in (left_wrist_point, right_wrist_point)
                            if point is not None
                        ]
                        hand_midpoint = (
                            tuple(np.mean(visible_wrist_points, axis=0))
                            if visible_wrist_points else None
                        )
                        hand_line_angle = calculate_hand_line_angle(
                            left_wrist_point, right_wrist_point
                        )
                        hip_midpoint = visible_midpoint(
                            landmarks, HIP_LANDMARKS, frame_width, frame_height
                        )
                        annotated_frame = draw_landmarks_on_image(rgb_frame, result)
                        output_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    else:
                        head_down_angle = None
                        eye_midpoint = None
                        torso_length = None
                        left_wrist_point = None
                        right_wrist_point = None
                        hand_midpoint = None
                        hand_line_angle = None
                        hip_midpoint = None
                        output_frame = frame

                    angle_samples.append((
                        time_seconds,
                        left_angle,
                        right_angle,
                        torso_lean_angle,
                        left_wrist_point,
                        hand_midpoint,
                        hip_midpoint,
                        right_wrist_point,
                        head_down_angle,
                        eye_midpoint,
                        torso_length,
                        hand_line_angle,
                    ))
                    csv_writer.writerow([
                        f"{time_seconds:.6f}",
                        "" if left_angle is None else f"{left_angle:.2f}",
                        "" if right_angle is None else f"{right_angle:.2f}",
                        "" if torso_lean_angle is None else f"{torso_lean_angle:.2f}",
                    ])
                    out.write(output_frame)
                    frame_index += 1
    finally:
        cap.release()
        out.release()

    if not angle_samples:
        raise RuntimeError(f"No frames could be read from input video: {VIDEO_PATH}")

    save_angle_plot(angle_samples, fps)
    save_torso_lean_plot(angle_samples, fps)
    captured_moments = save_stroke_snapshots(angle_samples, fps)
    print(f"Annotated video saved to {OUTPUT_VIDEO_PATH}")
    print(f"Elbow angle data saved to {OUTPUT_CSV_PATH}")
    print(f"Elbow angle plot saved to {OUTPUT_PLOT_PATH}")
    print(f"Torso lean plot saved to {OUTPUT_TORSO_PLOT_PATH}")
    print(
        f"Saved {len(captured_moments)} stroke snapshots to {STROKE_SNAPSHOT_DIR}"
    )
    print(f"Stroke moment timestamps saved to {STROKE_MOMENTS_CSV_PATH}")


if __name__ == "__main__":
    main()
