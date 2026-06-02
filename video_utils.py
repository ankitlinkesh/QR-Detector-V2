from dataclasses import dataclass

import cv2


@dataclass
class VideoFrame:
    image: object
    frame_index: int
    timestamp_sec: float | None


def iter_video_frames(video_path, frame_step=5, max_frames=None):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    processed_count = 0

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            if frame_index % frame_step == 0:
                timestamp = frame_index / fps if fps and fps > 0 else None
                yield VideoFrame(image=frame, frame_index=frame_index, timestamp_sec=timestamp)
                processed_count += 1

                if max_frames is not None and processed_count >= max_frames:
                    break

            frame_index += 1
    finally:
        capture.release()
