from dataclasses import dataclass

import cv2
import numpy as np
import zxingcpp


@dataclass
class QRDetection:
    """One QR code found in a camera frame."""

    data: str
    points: np.ndarray
    source: str = "camera"


class QRDetector:
    """QR detector that uses ZXing first and OpenCV as a fallback."""

    def __init__(self):
        self.opencv_detector = cv2.QRCodeDetector()

    def detect(self, frame):
        """
        Return a list of QRDetection objects found in the frame.

        ZXing is the primary decoder because it is more reliable when several
        QR codes are visible. OpenCV remains as a fallback when ZXing finds
        nothing.
        """
        zxing_detections = self._detect_with_zxing(frame)

        if zxing_detections:
            return self._remove_duplicate_detections(zxing_detections)

        return self._remove_duplicate_detections(self._detect_with_opencv(frame))

    def debug_scan(self, frame):
        """Run both detector backends and return counts/texts for troubleshooting."""
        zxing_detections = self._detect_with_zxing(frame)
        opencv_detections = self._detect_with_opencv(frame)

        return {
            "frame_width": frame.shape[1],
            "frame_height": frame.shape[0],
            "zxing_count": len(zxing_detections),
            "opencv_count": len(opencv_detections),
            "zxing_texts": [detection.data for detection in zxing_detections],
            "opencv_texts": [detection.data for detection in opencv_detections],
        }

    def _detect_with_zxing(self, frame):
        barcodes = zxingcpp.read_barcodes(frame, formats=zxingcpp.BarcodeFormat.QRCode)
        detections = []

        for barcode in barcodes:
            if not barcode.valid:
                continue

            detections.append(
                QRDetection(
                    data=barcode.text,
                    points=_points_from_zxing_position(barcode.position),
                    source="zxing",
                )
            )

        return detections

    def _detect_with_opencv(self, frame):
        found, decoded_info, points, _ = self.opencv_detector.detectAndDecodeMulti(frame)

        if not found or points is None:
            return []

        detections = []
        for data, qr_points in zip(decoded_info, points):
            detections.append(QRDetection(data=data, points=qr_points.astype("float32"), source="opencv"))

        return detections

    def _remove_duplicate_detections(self, detections):
        unique_detections = []

        for detection in detections:
            duplicate_index = self._find_duplicate_index(unique_detections, detection)

            if duplicate_index is None:
                unique_detections.append(detection)
            elif detection.data.strip() and not unique_detections[duplicate_index].data.strip():
                unique_detections[duplicate_index] = detection

        return unique_detections

    def _find_duplicate_index(self, detections, new_detection):
        new_center = _points_center(new_detection.points)

        for index, detection in enumerate(detections):
            center = _points_center(detection.points)
            distance = ((new_center[0] - center[0]) ** 2 + (new_center[1] - center[1]) ** 2) ** 0.5

            if distance < 25:
                return index

        return None


def draw_qr_box(frame, points, color, thickness=3):
    """Draw a four-sided box around a QR code."""
    corner_points = points.astype(int)

    for index in range(len(corner_points)):
        start_point = tuple(corner_points[index])
        end_point = tuple(corner_points[(index + 1) % len(corner_points)])
        cv2.line(frame, start_point, end_point, color, thickness)


def _points_center(points):
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def _points_from_zxing_position(position):
    return np.array(
        [
            [position.top_left.x, position.top_left.y],
            [position.top_right.x, position.top_right.y],
            [position.bottom_right.x, position.bottom_right.y],
            [position.bottom_left.x, position.bottom_left.y],
        ],
        dtype="float32",
    )
