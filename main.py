from datetime import datetime
from pathlib import Path

import cv2

from qr_detector import QRDetector, draw_qr_box
from qr_matcher import (
    QRMatcher,
    draw_center_point,
    draw_error_text,
    get_centering_error,
    get_qr_center,
)


BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def put_status_text(frame, text, color=WHITE):
    cv2.putText(
        frame,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )


def put_help_text(frame):
    text = "ZXing QR mode | s save | r reset | d debug | q quit"
    cv2.putText(
        frame,
        text,
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        WHITE,
        2,
        cv2.LINE_AA,
    )


def save_debug_frame(frame, detector):
    debug_dir = Path("work") / "debug_frames"
    debug_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = debug_dir / f"qr_debug_{timestamp}.png"
    cv2.imwrite(str(debug_path), frame)

    debug_info = detector.debug_scan(frame)

    print("[DEBUG] Saved frame:", debug_path)
    print(f"[DEBUG] Frame size: {debug_info['frame_width']}x{debug_info['frame_height']}")
    print(f"[DEBUG] ZXing QR count: {debug_info['zxing_count']}")
    print(f"[DEBUG] OpenCV QR count: {debug_info['opencv_count']}")
    print(f"[DEBUG] ZXing decoded text: {debug_info['zxing_texts']}")
    print(f"[DEBUG] OpenCV decoded text: {debug_info['opencv_texts']}")


def log_once(log_state, key, message):
    """Print repeated camera-loop messages only when their state changes."""
    if log_state.get(key) != message:
        print(message)
        log_state[key] = message


def first_non_empty_detection(detections):
    for detection in detections:
        if detection.data.strip():
            return detection
    return None


def draw_before_target_saved(frame, detections, log_state):
    valid_count = 0

    for detection in detections:
        if detection.data.strip():
            valid_count += 1
            draw_qr_box(frame, detection.points, BLUE)
        else:
            print("[WARN] QR detected, but decoded data is empty. Skipping it.")

    if valid_count > 0:
        put_status_text(frame, f"QR Found: {valid_count} - press 's' to save target", BLUE)
        log_once(
            log_state,
            "qr_state",
            f"[INFO] QR detected: {valid_count} decoded QR(s) visible from {len(detections)} total detection(s).",
        )
    else:
        put_status_text(frame, "No decoded QR found", RED)
        log_once(log_state, "qr_state", "[INFO] No QR detected.")


def draw_after_target_saved(frame, detections, matcher, log_state):
    match_found = False
    decoded_count = 0

    for detection in detections:
        if not detection.data.strip():
            print("[WARN] QR detected, but decoded data is empty. Skipping it.")
            continue

        decoded_count += 1

        if matcher.is_match(detection.data):
            match_found = True
            draw_qr_box(frame, detection.points, GREEN)

            qr_center = get_qr_center(detection.points)
            error_x, error_y = get_centering_error(qr_center, frame.shape)

            draw_center_point(frame, qr_center, GREEN)
            draw_error_text(frame, error_x, error_y)
            put_status_text(frame, "Target match found", GREEN)
            log_once(
                log_state,
                "match_state",
                f"[INFO] Match found. error_x={error_x}, error_y={error_y}",
            )
        else:
            draw_qr_box(frame, detection.points, RED)

    if decoded_count == 0:
        put_status_text(frame, "No decoded QR found", RED)
        log_once(log_state, "match_state", "[INFO] No QR detected.")
    elif not match_found:
        put_status_text(frame, f"No target match visible ({decoded_count} QR)", RED)
        log_once(
            log_state,
            "match_state",
            f"[INFO] QR detected: {decoded_count} decoded QR(s), but no target match found.",
        )


def main():
    print("[INFO] Starting QR matching project.")
    print("[INFO] Controls: s = save target, r = reset target, d = debug frame, q = quit")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("[ERROR] No camera found. Check that your webcam is connected and not used by another app.")
        return

    print("[INFO] Camera opened successfully.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("[INFO] Requested camera resolution: 1280x720.")

    detector = QRDetector()
    matcher = QRMatcher()
    log_state = {}

    while True:
        frame_ok, frame = camera.read()

        if not frame_ok:
            print("[ERROR] Could not read from camera.")
            break

        detections = detector.detect(frame)

        if not detections:
            put_status_text(frame, "No QR detected", RED)
            log_once(log_state, "qr_state", "[INFO] No QR detected.")
        elif matcher.has_target():
            draw_after_target_saved(frame, detections, matcher, log_state)
        else:
            draw_before_target_saved(frame, detections, log_state)

        put_help_text(frame)
        cv2.imshow("QR Matcher", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            detection = first_non_empty_detection(detections)

            if detection is None:
                print("[WARN] Cannot save target: no decoded QR data is visible.")
            elif matcher.save_target(detection.data):
                print(f"[INFO] Target saved: {matcher.target_data}")
                log_state.clear()
            else:
                print("[WARN] Cannot save target: decoded QR data is empty.")

        elif key == ord("r"):
            matcher.reset_target()
            log_state.clear()
            print("[INFO] Target reset. Scan a QR and press 's' to save a new target.")

        elif key == ord("d"):
            save_debug_frame(frame, detector)

        elif key == ord("q"):
            print("[INFO] Quitting QR matcher.")
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
