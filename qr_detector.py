from dataclasses import dataclass, field
from time import perf_counter

import cv2
import numpy as np

from preprocessing import (
    ImageVariant,
    apply_clahe,
    crop_with_padding,
    generate_crop_variants,
    generate_fallback_tile_variants,
    generate_full_frame_variants,
    perspective_rectify,
    to_bgr,
)
from text_filter import clean_text, normalize_text

try:
    import zxingcpp
except ImportError:
    zxingcpp = None


LOCATOR_WIDTH = 1280
MAX_NEAR_MISSES = 80


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


@dataclass
class DetectionProfile:
    candidate_attempts: int = 0
    decode_attempts: int = 0
    elapsed_sec: float = 0.0


class QRDetector:
    def __init__(self, mode="smart", debug=False, profile=False, max_variants_per_frame=160, max_candidates=24):
        if zxingcpp is None:
            raise RuntimeError("zxing-cpp is not installed. Run: python -m pip install -r requirements.txt")

        self.mode = normalize_mode(mode)
        self.debug = debug
        self.profile = profile
        self.max_variants_per_frame = max_variants_per_frame
        self.max_candidates = max_candidates
        self.opencv_detector = cv2.QRCodeDetector()
        self.last_candidates = []
        self.last_raw_candidates = []
        self.last_raw_opencv_candidates = []
        self.last_raw_contour_candidates = []
        self.last_near_misses = []
        self.last_failure_reason = None
        self.last_profile = {}

    def detect(self, image) -> list[QRDetection]:
        self._reset_last_run()
        profile = DetectionProfile()
        start = perf_counter()

        if self.mode == "fast":
            detections = self._decode_full_frame(image, "fast", profile)
            merged = merge_duplicate_detections(detections)
            self._finish_profile(profile, start, merged, candidate_count=0)
            return merged

        detections = self._decode_full_frame(image, "smart", profile)
        candidates = self._find_candidates(image, profile, quick=bool(detections))
        self.last_candidates = candidates_to_dicts(candidates)
        candidates_to_decode = filter_undecoded_candidates(candidates, detections)
        detections.extend(self._decode_candidates(image, candidates_to_decode, profile, deep=False))

        if not detections:
            detections.extend(self._decode_fallback_tiles(image, profile))

        merged = merge_duplicate_detections(detections)
        self._finish_profile(profile, start, merged, candidate_count=len(candidates))
        return merged

    def decode_known_candidates(self, image, candidate_dicts, max_variants=None) -> list[QRDetection]:
        old_budget = self.max_variants_per_frame
        if max_variants is not None:
            self.max_variants_per_frame = max_variants

        self._reset_last_run()
        profile = DetectionProfile()
        start = perf_counter()

        candidates = []
        for candidate in candidate_dicts:
            points = np.array(candidate["points"], dtype="float32")
            source = candidate.get("source", "temporal_candidate")
            score = float(candidate.get("score", 0)) + 5000
            candidates.append(QRCandidate(points=points, source=f"temporal_{source}", score=score))

        self.last_candidates = candidates_to_dicts(candidates)
        detections = self._decode_candidates(image, candidates, profile, deep=True)
        merged = merge_duplicate_detections(detections)
        self._finish_profile(profile, start, merged, candidate_count=len(candidates))
        self.max_variants_per_frame = old_budget
        return merged

    def _reset_last_run(self):
        self.last_candidates = []
        self.last_raw_candidates = []
        self.last_raw_opencv_candidates = []
        self.last_raw_contour_candidates = []
        self.last_near_misses = []
        self.last_failure_reason = None
        self.last_profile = {}

    def _decode_full_frame(self, image, mode, profile):
        detections = []

        for variant in generate_full_frame_variants(image, mode):
            detections.extend(self._detect_with_zxing(variant, profile, binarizers=basic_binarizers(), record_errors=False))

            if mode == "smart" and profile.decode_attempts < self.max_variants_per_frame:
                detections.extend(self._detect_with_opencv_decode(variant, profile))

            if detections:
                break

        return detections

    def _find_candidates(self, image, profile, quick=False):
        candidates = []
        raw_opencv_candidates = []
        raw_contour_candidates = []

        for locator_index, locator_variant in enumerate(self._generate_locator_variants(image)):
            opencv_candidates = self._find_opencv_candidates(locator_variant, profile)
            raw_opencv_candidates.extend(opencv_candidates)
            candidates.extend(opencv_candidates)

            if quick:
                break

            contour_candidates = self._find_contour_candidates(locator_variant, profile)
            raw_contour_candidates.extend(contour_candidates)
            candidates.extend(contour_candidates)

        self.last_raw_opencv_candidates = candidates_to_dicts(raw_opencv_candidates)
        self.last_raw_contour_candidates = candidates_to_dicts(raw_contour_candidates)
        self.last_raw_candidates = candidates_to_dicts(raw_opencv_candidates + raw_contour_candidates)
        return merge_candidate_duplicates(candidates, self.max_candidates)

    def _generate_locator_variants(self, image):
        height, width = image.shape[:2]
        scale = min(1.0, LOCATOR_WIDTH / width)
        locator = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
        gray = cv2.cvtColor(locator, cv2.COLOR_BGR2GRAY)

        yield ImageVariant(image=locator, name="locator_original", source="locator_original", scale=scale)
        yield ImageVariant(image=to_bgr(gray), name="locator_gray", source="locator_gray", scale=scale)
        yield ImageVariant(image=to_bgr(apply_clahe(gray)), name="locator_clahe", source="locator_clahe", scale=scale)

        block_size = 31 if min(locator.shape[:2]) >= 31 else max(3, min(locator.shape[:2]) | 1)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 3)
        yield ImageVariant(image=to_bgr(adaptive), name="locator_adaptive", source="locator_adaptive", scale=scale)

        if scale < 1.0:
            fullres_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            yield ImageVariant(image=to_bgr(fullres_gray), name="locator_fullres_gray", source="locator_fullres_gray", scale=1.0)
            yield ImageVariant(
                image=to_bgr(apply_clahe(fullres_gray)),
                name="locator_fullres_clahe",
                source="locator_fullres_clahe",
                scale=1.0,
            )
            fullres_block_size = 31 if min(image.shape[:2]) >= 31 else max(3, min(image.shape[:2]) | 1)
            fullres_adaptive = cv2.adaptiveThreshold(
                fullres_gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                fullres_block_size,
                3,
            )
            yield ImageVariant(
                image=to_bgr(fullres_adaptive),
                name="locator_fullres_adaptive",
                source="locator_fullres_adaptive",
                scale=1.0,
            )

    def _find_opencv_candidates(self, variant, profile):
        candidates = []
        profile.candidate_attempts += 2

        found_multi, points_multi = self.opencv_detector.detectMulti(variant.image)
        if found_multi and points_multi is not None:
            for points in points_multi:
                mapped = map_points_to_original(points.astype("float32"), variant)
                candidates.append(
                    QRCandidate(
                        points=mapped,
                        source=f"opencv_detect_multi_{variant.name}",
                        score=1_200_000 + candidate_shape_score(mapped),
                    )
                )

        found_single, points_single = self.opencv_detector.detect(variant.image)
        if found_single and points_single is not None:
            for points in points_single.reshape(-1, 4, 2):
                mapped = map_points_to_original(points.astype("float32"), variant)
                candidates.append(
                    QRCandidate(
                        points=mapped,
                        source=f"opencv_detect_single_{variant.name}",
                        score=1_100_000 + candidate_shape_score(mapped),
                    )
                )

        return candidates

    def _find_contour_candidates(self, variant, profile):
        gray = cv2.cvtColor(variant.image, cv2.COLOR_BGR2GRAY)
        if "adaptive" not in variant.name:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

        edges = cv2.Canny(gray, 45, 170)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        profile.candidate_attempts += 1

        candidates = []
        image_area = variant.image.shape[0] * variant.image.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)
            min_area = max(16.0, image_area * 0.000035)
            if area < min_area or area > image_area * 0.18:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            locator_points = approx.reshape(4, 2).astype("float32")
            points = map_points_to_original(locator_points, variant)
            x, y, box_width, box_height = cv2.boundingRect(points)
            if box_width < 10 or box_height < 10:
                continue

            aspect = box_width / max(1, box_height)
            if aspect < 0.35 or aspect > 2.8:
                continue

            locator_box_area = max(1, box_width * box_height * variant.scale * variant.scale)
            rectangularity = area / locator_box_area
            if rectangularity < 0.2:
                continue

            score = candidate_shape_score(points, rectangularity=rectangularity)
            candidates.append(QRCandidate(points=points, source=f"contour_quad_{variant.name}", score=score))

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates[: self.max_candidates]

    def _decode_candidates(self, image, candidates, profile, deep=False):
        detections = []

        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate_start_count = len(detections)
            candidate_decoded = False
            padding_ratio = 0.8 if deep else 0.55
            min_padding = 70 if deep else 42
            crop, crop_points, offset_x, offset_y = crop_with_padding(image, candidate.points, padding_ratio=padding_ratio, min_padding=min_padding)
            if crop.size == 0:
                continue

            base_variants = [
                (
                    ImageVariant(
                        image=crop,
                        name=f"candidate_{candidate_index}_crop",
                        source=f"zxing_{candidate.source}_crop",
                        offset_x=offset_x,
                        offset_y=offset_y,
                    ),
                    None,
                )
            ]

            for output_size, border in rectification_specs(candidate, deep):
                try:
                    rectified = perspective_rectify(crop, crop_points, output_size=output_size, border=border)
                    base_variants.append(
                        (
                            ImageVariant(
                                image=rectified,
                                name=f"candidate_{candidate_index}_rectified_{output_size}",
                                source=f"zxing_{candidate.source}_rectified_{output_size}",
                            ),
                            candidate.points,
                        )
                    )
                except cv2.error:
                    pass

            for base_variant, override_points in base_variants:
                for variant in generate_crop_variants(base_variant.image, prefix=base_variant.name):
                    if not crop_variant_allowed(variant.name, deep):
                        continue

                    if candidate_decoded:
                        break

                    if profile.decode_attempts >= self.max_variants_per_frame:
                        return detections

                    variant.source = f"{base_variant.source}_{variant.source.replace('zxing_candidate_', '')}"
                    variant.offset_x = base_variant.offset_x
                    variant.offset_y = base_variant.offset_y
                    detections.extend(
                        self._detect_with_zxing(
                            variant,
                            profile,
                            override_points=override_points,
                            binarizers=candidate_binarizers(variant.name, deep=deep),
                            record_errors=True,
                            try_downscale=False,
                        )
                    )

                    if profile.decode_attempts >= self.max_variants_per_frame:
                        return detections

                    detections.extend(self._detect_with_opencv_decode(variant, profile, override_points=override_points))
                    candidate_decoded = len(detections) > candidate_start_count

                if candidate_decoded:
                    break

        return detections

    def _decode_fallback_tiles(self, image, profile):
        detections = []

        for variant in generate_fallback_tile_variants(image, max_tiles=24):
            if profile.decode_attempts >= self.max_variants_per_frame:
                break
            detections.extend(
                self._detect_with_zxing(
                    variant,
                    profile,
                    binarizers=basic_binarizers(),
                    record_errors=True,
                    try_downscale=True,
                )
            )

        return detections

    def _detect_with_zxing(self, variant: ImageVariant, profile, override_points=None, binarizers=None, record_errors=False, try_downscale=True) -> list[QRDetection]:
        detections = []
        binarizers = binarizers or basic_binarizers()

        for binarizer in binarizers:
            if profile.decode_attempts >= self.max_variants_per_frame:
                break

            profile.decode_attempts += 1
            try:
                barcodes = zxingcpp.read_barcodes(
                    variant.image,
                    formats=zxingcpp.BarcodeFormat.QRCode,
                    binarizer=binarizer,
                    try_downscale=try_downscale,
                    return_errors=record_errors,
                )
            except Exception as error:
                if record_errors:
                    self._record_near_miss(variant.source, variant.name, str(error), points=override_points)
                continue

            for barcode in barcodes:
                points = override_points if override_points is not None else points_from_barcode(barcode, variant)

                if not barcode.valid:
                    if record_errors:
                        self._record_near_miss(
                            source=variant.source,
                            variant_name=variant.name,
                            error=str(getattr(barcode, "error", "decode_error")),
                            points=points,
                        )
                    continue

                if points is None:
                    continue

                detection = make_detection(
                    data=barcode.text,
                    points=points,
                    source=f"{variant.source}_{binarizer_name(binarizer)}",
                    preprocessing_variant=variant.name,
                    confidence=None,
                )
                if detection is not None:
                    detections.append(detection)

        return detections

    def _detect_with_opencv_decode(self, variant: ImageVariant, profile, override_points=None) -> list[QRDetection]:
        if profile.decode_attempts >= self.max_variants_per_frame:
            return []

        profile.decode_attempts += 1
        found, decoded_info, points, _ = self.opencv_detector.detectAndDecodeMulti(variant.image)

        if not found or points is None:
            return []

        detections = []
        for data, qr_points in zip(decoded_info, points):
            mapped_points = override_points if override_points is not None else map_points_to_original(qr_points.astype("float32"), variant)
            detection = make_detection(
                data=data,
                points=mapped_points,
                source=f"opencv_{variant.source}",
                preprocessing_variant=variant.name,
                confidence=None,
            )
            if detection is not None:
                detections.append(detection)

        return detections

    def _record_near_miss(self, source, variant_name, error, points=None):
        reason = "checksum_near_miss" if "ChecksumError" in str(error) else "decoder_near_miss"
        near_miss = {
            "reason": reason,
            "source": source,
            "preprocessing_variant": variant_name,
            "error": str(error),
            "points": None,
            "center": None,
        }

        if points is not None:
            points_array = np.array(points, dtype="float32")
            near_miss["points"] = points_array.astype(float).round(2).tolist()
            near_miss["center"] = [round(value, 2) for value in candidate_center(points_array)]

        signature = (
            near_miss["reason"],
            near_miss["source"],
            near_miss["preprocessing_variant"],
            tuple(round(value / 20) for value in near_miss["center"]) if near_miss["center"] else None,
        )
        for existing in self.last_near_misses:
            existing_signature = (
                existing["reason"],
                existing["source"],
                existing["preprocessing_variant"],
                tuple(round(value / 20) for value in existing["center"]) if existing["center"] else None,
            )
            if existing_signature == signature:
                return

        if len(self.last_near_misses) < MAX_NEAR_MISSES:
            self.last_near_misses.append(near_miss)

    def _finish_profile(self, profile, start, detections, candidate_count=0):
        profile.elapsed_sec = perf_counter() - start
        self.last_failure_reason = classify_failure(detections, candidate_count, self.last_near_misses)
        self.last_profile = {
            "elapsed_sec": round(profile.elapsed_sec, 3),
            "candidate_count": candidate_count,
            "candidate_attempts": profile.candidate_attempts,
            "decode_attempts": profile.decode_attempts,
            "near_misses": len(self.last_near_misses),
            "failure_reason": self.last_failure_reason,
            "detections": len(detections),
        }

        if self.profile or self.debug:
            print(
                "[PROFILE] "
                f"time={self.last_profile['elapsed_sec']}s "
                f"candidates={candidate_count} "
                f"candidate_attempts={profile.candidate_attempts} "
                f"decode_attempts={profile.decode_attempts} "
                f"near_misses={len(self.last_near_misses)} "
                f"reason={self.last_failure_reason} "
                f"detections={len(detections)}"
            )


def normalize_mode(mode):
    if mode == "robust":
        return "smart"
    return mode


def basic_binarizers():
    return [zxingcpp.Binarizer.LocalAverage]


def candidate_binarizers(variant_name, deep=False):
    if deep:
        return [
            zxingcpp.Binarizer.LocalAverage,
            zxingcpp.Binarizer.GlobalHistogram,
            zxingcpp.Binarizer.FixedThreshold,
            zxingcpp.Binarizer.BoolCast,
        ]

    if any(token in variant_name for token in ("clahe", "adaptive", "otsu")):
        return [zxingcpp.Binarizer.LocalAverage, zxingcpp.Binarizer.GlobalHistogram]

    return [zxingcpp.Binarizer.LocalAverage]


def binarizer_name(binarizer):
    return str(binarizer).split(".")[-1].lower()


def crop_variant_allowed(variant_name, deep):
    if deep:
        return True

    if "upscaled_3x" in variant_name or "upscaled_4x" in variant_name:
        return False

    return True


def rectification_specs(candidate, deep):
    if not deep:
        return [(640, 48)]

    return [(640, 48), (900, 72), (1200, 96)]


def classify_failure(detections, candidate_count, near_misses):
    if detections:
        return "decoded"

    if any(near_miss["reason"] == "checksum_near_miss" for near_miss in near_misses):
        return "checksum_near_miss"

    if candidate_count > 0:
        return "candidate_no_decode"

    return "no_candidate"


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


def points_from_barcode(barcode, variant):
    try:
        points = _points_from_zxing_position(barcode.position)
    except Exception:
        return None
    return map_points_to_original(points, variant)


def candidates_to_dicts(candidates):
    candidate_dicts = []
    for candidate in candidates:
        x1, y1, x2, y2 = bounding_box(candidate.points)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        candidate_dicts.append(
            {
                "points": candidate.points.astype(float).round(2).tolist(),
                "center": [round(value, 2) for value in candidate_center(candidate.points)],
                "source": candidate.source,
                "score": round(candidate.score, 2),
                "width": round(width, 2),
                "height": round(height, 2),
                "area": round(width * height, 2),
            }
        )
    return candidate_dicts


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
        if any(candidate_iou(candidate.points, existing.points) > 0.2 or point_distance(candidate_center(candidate.points), candidate_center(existing.points)) < 55 for existing in merged):
            continue
        merged.append(candidate)
        if len(merged) >= max_candidates:
            break

    return merged


def filter_undecoded_candidates(candidates, detections):
    if not detections:
        return candidates

    remaining = []
    for candidate in candidates:
        if candidate.source.startswith("contour") and candidate.score < 80_000:
            continue

        if any(candidate_iou(candidate.points, detection.points) > 0.2 or point_distance(candidate_center(candidate.points), detection.center) < 90 for detection in detections):
            continue
        remaining.append(candidate)

    return remaining


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


def candidate_shape_score(points, rectangularity=1.0):
    x1, y1, x2, y2 = bounding_box(points)
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    area = box_width * box_height
    aspect = box_width / box_height
    aspect_balance = 1 - abs(1 - min(aspect, 1 / aspect))
    return float(area * (0.6 + rectangularity) + 2500 * aspect_balance)


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
