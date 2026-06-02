from dataclasses import dataclass, field
from time import perf_counter

import cv2
import numpy as np

from preprocessing import (
    ImageVariant,
    crop_with_padding,
    generate_crop_variants,
    generate_fallback_tile_variants,
    generate_full_frame_variants,
    perspective_rectify,
)
from text_filter import clean_text, normalize_text

try:
    import zxingcpp
except ImportError:
    zxingcpp = None


@dataclass
class QRDetection:
    data: str
    normalized_data: str
    points: np.ndarray
    center: list[int]
    width: float
    height: float
    area: float
    source: str
    confidence: float | None
    preprocessing_variant: str
    frame_index: int | None = None
    input_path: str | None = None
    merged_sources: list[str] = field(default_factory=list)


@dataclass
class QRCandidate:
    points: np.ndarray
    source: str
    score: float


class QRDetector:
    def __init__(self, mode="robust", debug=False, profile=False, max_variants_per_frame=160, max_candidates=24):
        if zxingcpp is None:
            raise RuntimeError("zxing-cpp is not installed. Run: python -m pip install -r requirements.txt")

        self.mode = mode
        self.debug = debug
        self.profile = profile
        self.max_variants_per_frame = max_variants_per_frame
        self.max_candidates = max_candidates
        self.opencv_detector = cv2.QRCodeDetector()
        self.last_candidates = []
        self.last_profile = {}

    def detect(self, image) -> list[QRDetection]:
        profile = DetectionProfile()
        start = perf_counter()

        if self.mode == "fast":
            detections = self._decode_full_frame(image, "fast", profile)
            self.last_candidates = []
            self._finish_profile(profile, start, detections)
            return merge_duplicate_detections(detections)

        detections = self._decode_full_frame(image, "robust", profile)
        candidates = self._find_candidates(image, profile)
        self.last_candidates = candidates_to_dicts(candidates)
        detections.extend(self._decode_candidates(image, candidates, profile))

        if not detections:
            detections.extend(self._decode_fallback_tiles(image, profile))

        merged = merge_duplicate_detections(detections)
        self._finish_profile(profile, start, merged, candidate_count=len(candidates))
        return merged

    def _decode_full_frame(self, image, mode, profile):
        detections = []

        for variant in generate_full_frame_variants(image, mode):
            detections.extend(self._detect_with_zxing(variant, profile))

            if mode == "robust":
                detections.extend(self._detect_with_opencv_decode(variant, profile))

            if mode == "fast" and detections:
                break

        return detections

    def _find_candidates(self, image, profile):
        candidates = []
        candidates.extend(self._find_opencv_candidates(image, profile))
        candidates.extend(self._find_contour_candidates(image, profile))
        return merge_candidate_duplicates(candidates, self.max_candidates)

    def _find_opencv_candidates(self, image, profile):
        candidates = []
        profile.candidate_attempts += 2

        found_multi, points_multi = self.opencv_detector.detectMulti(image)
        if found_multi and points_multi is not None:
            for points in points_multi:
                candidates.append(QRCandidate(points=points.astype("float32"), source="opencv_detect_multi", score=1000 + contour_area(points)))

        found_single, points_single = self.opencv_detector.detect(image)
        if found_single and points_single is not None:
            for points in points_single.reshape(-1, 4, 2):
                candidates.append(QRCandidate(points=points.astype("float32"), source="opencv_detect_single", score=900 + contour_area(points)))

        return candidates

    def _find_contour_candidates(self, image, profile):
        locator_width = 1280
        height, width = image.shape[:2]
        scale = min(1.0, locator_width / width)
        locator = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
        gray = cv2.cvtColor(locator, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 60, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        profile.candidate_attempts += 1

        candidates = []
        image_area = locator.shape[0] * locator.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * 0.00025 or area > image_area * 0.12:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            points = approx.reshape(4, 2).astype("float32") / scale
            x, y, box_width, box_height = cv2.boundingRect(points)
            if box_width < 40 or box_height < 40:
                continue

            aspect = box_width / max(1, box_height)
            if aspect < 0.35 or aspect > 2.8:
                continue

            rectangularity = area / max(1, box_width * box_height * scale * scale)
            if rectangularity < 0.25:
                continue

            score = float(area) + 200 * (1 - abs(1 - min(aspect, 1 / aspect)))
            candidates.append(QRCandidate(points=points, source="contour_quad", score=score))

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates[: self.max_candidates]

    def _decode_candidates(self, image, candidates, profile):
        detections = []

        for candidate_index, candidate in enumerate(candidates, start=1):
            crop, crop_points, offset_x, offset_y = crop_with_padding(image, candidate.points)
            if crop.size == 0:
                continue

            candidate_variants = []
            candidate_variants.append(
                ImageVariant(
                    image=crop,
                    name=f"candidate_{candidate_index}_crop",
                    source=f"zxing_{candidate.source}_crop",
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
            )

            try:
                rectified = perspective_rectify(crop, crop_points)
                candidate_variants.append(
                    ImageVariant(
                        image=rectified,
                        name=f"candidate_{candidate_index}_rectified",
                        source=f"zxing_{candidate.source}_rectified",
                    )
                )
            except cv2.error:
                pass

            for base_variant in candidate_variants:
                for variant in generate_crop_variants(base_variant.image, prefix=base_variant.name):
                    if profile.decode_attempts >= self.max_variants_per_frame:
                        return detections

                    variant.source = base_variant.source if "rectified" in base_variant.source else variant.source
                    variant.offset_x = base_variant.offset_x
                    variant.offset_y = base_variant.offset_y
                    detections.extend(self._detect_with_zxing(variant, profile))
                    detections.extend(self._detect_with_opencv_decode(variant, profile))

                    if detections and self.mode == "fast":
                        return detections

        return detections

    def _decode_fallback_tiles(self, image, profile):
        detections = []

        for variant in generate_fallback_tile_variants(image, max_tiles=12):
            if profile.decode_attempts >= self.max_variants_per_frame:
                break
            detections.extend(self._detect_with_zxing(variant, profile))

        return detections

    def _detect_with_zxing(self, variant: ImageVariant, profile, override_points=None) -> list[QRDetection]:
        profile.decode_attempts += 1
        barcodes = zxingcpp.read_barcodes(variant.image, formats=zxingcpp.BarcodeFormat.QRCode)
        detections = []

        for barcode in barcodes:
            if not barcode.valid:
                continue

            points = override_points if override_points is not None else map_points_to_original(_points_from_zxing_position(barcode.position), variant)
            detection = make_detection(
                data=barcode.text,
                points=points,
                source=variant.source,
                preprocessing_variant=variant.name,
                confidence=None,
            )
            if detection is not None:
                detections.append(detection)

        return detections

    def _detect_with_opencv_decode(self, variant: ImageVariant, profile, override_points=None) -> list[QRDetection]:
        profile.decode_attempts += 1
        found, decoded_info, points, _ = self.opencv_detector.detectAndDecodeMulti(variant.image)

        if not found or points is None:
            return []

        detections = []
        for data, qr_points in zip(decoded_info, points):
            points = map_points_to_original(qr_points.astype("float32"), variant)
            detection = make_detection(
                data=data,
                points=points,
                source=f"opencv_{variant.source}",
                preprocessing_variant=variant.name,
                confidence=None,
            )
            if detection is not None:
                detections.append(detection)

        return detections

    def _finish_profile(self, profile, start, detections, candidate_count=0):
        profile.elapsed_sec = perf_counter() - start
        self.last_profile = {
            "elapsed_sec": round(profile.elapsed_sec, 3),
            "candidate_count": candidate_count,
            "candidate_attempts": profile.candidate_attempts,
            "decode_attempts": profile.decode_attempts,
            "detections": len(detections),
        }

        if self.profile or self.debug:
            print(
                "[PROFILE] "
                f"time={self.last_profile['elapsed_sec']}s "
                f"candidates={candidate_count} "
                f"candidate_attempts={profile.candidate_attempts} "
                f"decode_attempts={profile.decode_attempts} "
                f"detections={len(detections)}"
            )


@dataclass
class DetectionProfile:
    candidate_attempts: int = 0
    decode_attempts: int = 0
    elapsed_sec: float = 0.0


def make_detection(data, points, source, preprocessing_variant, confidence):
    cleaned = clean_text(data)
    normalized = normalize_text(cleaned)

    if not normalized:
        return None

    x, y, width, height = cv2.boundingRect(points.astype("float32"))
    area = float(width * height)
    center = [int(x + width / 2), int(y + height / 2)]

    return QRDetection(
        data=data,
        normalized_data=normalized,
        points=points.astype("float32"),
        center=center,
        width=float(width),
        height=float(height),
        area=area,
        source=source,
        confidence=confidence,
        preprocessing_variant=preprocessing_variant,
        merged_sources=[source],
    )


def map_points_to_original(points, variant: ImageVariant):
    mapped_points = points.astype("float32") / variant.scale
    mapped_points[:, 0] += variant.offset_x
    mapped_points[:, 1] += variant.offset_y
    return mapped_points

def candidates_to_dicts(candidates):
    return [
        {
            "points": candidate.points.astype(float).round(2).tolist(),
            "center": [round(value, 2) for value in candidate_center(candidate.points)],
            "source": candidate.source,
            "score": round(candidate.score, 2),
        }
        for candidate in candidates
    ]

def merge_duplicate_detections(detections):
    merged = []

    for detection in detections:
        duplicate_index = find_duplicate_index(merged, detection)

        if duplicate_index is None:
            merged.append(detection)
            continue

        existing = merged[duplicate_index]
        best = choose_best_detection(existing, detection)
        best.merged_sources = sorted(set(existing.merged_sources + detection.merged_sources))
        merged[duplicate_index] = best

    return merged


def merge_candidate_duplicates(candidates, max_candidates):
    merged = []

    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(candidate_iou(candidate.points, existing.points) > 0.25 or point_distance(candidate_center(candidate.points), candidate_center(existing.points)) < 45 for existing in merged):
            continue
        merged.append(candidate)
        if len(merged) >= max_candidates:
            break

    return merged


def find_duplicate_index(detections, new_detection):
    for index, detection in enumerate(detections):
        if detection.normalized_data != new_detection.normalized_data:
            continue

        center_distance = point_distance(detection.center, new_detection.center)
        overlap = candidate_iou(detection.points, new_detection.points)

        if center_distance < 40 or overlap > 0.3:
            return index

    return None


def choose_best_detection(left, right):
    left_score = detection_score(left)
    right_score = detection_score(right)
    return right if right_score > left_score else left


def detection_score(detection):
    score = detection.area

    if detection.source.startswith("zxing"):
        score += 100000

    if "original" in detection.preprocessing_variant or "rectified" in detection.preprocessing_variant:
        score += 10000

    if "candidate" in detection.source:
        score += 5000

    return score


def point_distance(left, right):
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5


def candidate_center(points):
    return [float(points[:, 0].mean()), float(points[:, 1].mean())]


def contour_area(points):
    return float(cv2.contourArea(points.astype("float32")))


def candidate_iou(left_points, right_points):
    left_box = bounding_box(left_points)
    right_box = bounding_box(right_points)

    x1 = max(left_box[0], right_box[0])
    y1 = max(left_box[1], right_box[1])
    x2 = min(left_box[2], right_box[2])
    y2 = min(left_box[3], right_box[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, left_box[2] - left_box[0]) * max(0, left_box[3] - left_box[1])
    right_area = max(0, right_box[2] - right_box[0]) * max(0, right_box[3] - right_box[1])
    union = left_area + right_area - intersection

    if union == 0:
        return 0

    return intersection / union


def bounding_box(points):
    x_values = points[:, 0]
    y_values = points[:, 1]
    return [float(x_values.min()), float(y_values.min()), float(x_values.max()), float(y_values.max())]


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




