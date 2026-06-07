from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

try:
    import zxingcpp
except ImportError:
    zxingcpp = None


@dataclass
class RecoveryVariant:
    name: str
    image: np.ndarray
    preserves_geometry: bool = True


@dataclass
class RankedFrame:
    frame_index: int
    score: float
    image: np.ndarray
    sharpness: float
    contrast: float
    brightness: float
    qr_visibility: float


class FrameRanker:
    def __init__(self):
        self.opencv_detector = cv2.QRCodeDetector()

    def rank(self, frames, top_n=10):
        ranked = []

        for frame_index, frame in frames:
            ranked.append(self.score_frame(frame_index, frame))

        ranked.sort(key=lambda frame: frame.score, reverse=True)
        return ranked[:top_n]

    def score_frame(self, frame_index, frame):
        gray = ensure_gray(frame)
        sharpness = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 500.0, 1.0)
        contrast = min(float(gray.std()) / 80.0, 1.0)
        brightness = max(0.0, 1.0 - abs(float(gray.mean()) - 127.0) / 127.0)
        qr_visibility = self.qr_visibility_score(gray)
        score = round(0.40 * sharpness + 0.25 * contrast + 0.20 * qr_visibility + 0.15 * brightness, 4)
        return RankedFrame(
            frame_index=frame_index,
            score=score,
            image=frame,
            sharpness=round(sharpness, 4),
            contrast=round(contrast, 4),
            brightness=round(brightness, 4),
            qr_visibility=round(qr_visibility, 4),
        )

    def qr_visibility_score(self, gray):
        try:
            found, points = self.opencv_detector.detect(gray)
            if found and points is not None:
                return 1.0
        except cv2.error:
            pass

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        square_count = 0
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx) != 4:
                continue

            area = cv2.contourArea(contour)
            if area <= 100:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            aspect = width / max(height, 1)
            if 0.7 < aspect < 1.3:
                square_count += 1

        if square_count >= 3:
            return min(square_count / 12.0, 1.0)

        return min(square_count * 0.15, 1.0)


def generate_recovery_variants(image):
    gray = ensure_gray(image)
    yield RecoveryVariant("original", image)
    yield RecoveryVariant("grayscale", gray)

    hist_eq = cv2.equalizeHist(gray)
    clahe = apply_recovery_clahe(gray)
    adaptive = adaptive_threshold(gray)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    yield RecoveryVariant("histogram_equalization", hist_eq)
    yield RecoveryVariant("clahe", clahe)
    yield RecoveryVariant("adaptive_threshold", adaptive)
    yield RecoveryVariant("otsu_threshold", otsu)
    yield RecoveryVariant("gaussian_blur_reduction", gaussian_blur_reduction(gray))
    yield RecoveryVariant("median_filter", cv2.medianBlur(gray, 3))
    yield RecoveryVariant("bilateral_filter", cv2.bilateralFilter(gray, 9, 75, 75))
    yield RecoveryVariant("sharpen", sharpen(gray))
    yield RecoveryVariant("unsharp_mask", unsharp_mask(gray))
    yield RecoveryVariant("contrast_stretch", contrast_stretch(gray))
    yield RecoveryVariant("brightness_correction", brightness_correction(gray))
    yield RecoveryVariant("gamma_low", gamma_correction(gray, 0.7))
    yield RecoveryVariant("gamma_high", gamma_correction(gray, 1.5))
    yield RecoveryVariant("morph_open", cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
    yield RecoveryVariant("morph_close", cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)))
    yield RecoveryVariant("erosion", cv2.erode(otsu, np.ones((2, 2), np.uint8), iterations=1))
    yield RecoveryVariant("dilation", cv2.dilate(otsu, np.ones((2, 2), np.uint8), iterations=1))
    yield RecoveryVariant("edge_enhance", edge_enhance(gray))
    yield RecoveryVariant("noise_removal", noise_removal(gray))
    yield RecoveryVariant("deblur", deblur(gray))
    yield RecoveryVariant("super_resolution_sim", upscale_and_sharpen(gray, scale=2))

    for angle in (90, 180, 270, 45, 135, 225, 315):
        yield RecoveryVariant(f"rotation_{angle}", rotate_image(gray, angle), preserves_geometry=False)

    for scale in (1.5, 2, 3, 4):
        label = str(scale).replace(".0", "").replace(".", "_")
        yield RecoveryVariant(f"scale_{label}x", resize_by_scale(gray, scale), preserves_geometry=False)

    yield RecoveryVariant("grayscale_clahe_sharpen", sharpen(clahe))
    yield RecoveryVariant("otsu_resize_2x", resize_by_scale(otsu, 2), preserves_geometry=False)
    yield RecoveryVariant("clahe_otsu_sharpen", sharpen(otsu_threshold(clahe)))
    yield RecoveryVariant("denoise_clahe_adaptive", adaptive_threshold(apply_recovery_clahe(noise_removal(gray))))
    yield RecoveryVariant("brightness_contrast_sharpen", sharpen(contrast_stretch(brightness_correction(gray))))
    yield RecoveryVariant("superres_clahe", apply_recovery_clahe(upscale_and_sharpen(gray, scale=2)), preserves_geometry=False)


def decode_recovery_variants(image, frame_index=None, max_variants=None, max_side=None):
    if zxingcpp is None:
        return []

    results = []
    seen_texts = set()
    start = perf_counter()
    working_image, base_preserves_geometry, resize_scale = prepare_recovery_image(image, max_side)

    for variant_index, variant in enumerate(generate_recovery_variants(working_image), start=1):
        if max_variants is not None and variant_index > max_variants:
            break

        try:
            barcodes = zxingcpp.read_barcodes(variant.image)
        except Exception:
            continue

        for barcode in barcodes:
            text = str(getattr(barcode, "text", "")).strip()
            if not text or text in seen_texts:
                continue

            seen_texts.add(text)
            results.append(
                {
                    "text": text,
                    "format": barcode_format_name(barcode),
                    "variant": variant.name,
                    "frame_index": frame_index,
                    "variant_index": variant_index,
                    "processing_time": round(perf_counter() - start, 4),
                    "source": "zxing_recovery_whole_frame",
                    "points": points_from_barcode(barcode) if variant.preserves_geometry and base_preserves_geometry else None,
                    "preserves_geometry": variant.preserves_geometry and base_preserves_geometry,
                    "recovery_resize_scale": resize_scale,
                }
            )

    return results



def prepare_recovery_image(image, max_side=None):
    if max_side is None or max_side <= 0:
        return image, True, 1.0

    height, width = image.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_side:
        return image, True, 1.0

    scale = max_side / float(largest_side)
    resized = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, False, scale
def ensure_gray(image):
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_recovery_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def adaptive_threshold(gray):
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)


def otsu_threshold(gray):
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


def sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def gaussian_blur_reduction(gray):
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def unsharp_mask(gray):
    blurred = cv2.GaussianBlur(gray, (9, 9), 10.0)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def contrast_stretch(gray):
    p2, p98 = np.percentile(gray, (2, 98))
    if p98 - p2 < 1:
        return gray
    return np.clip((gray - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)


def brightness_correction(gray):
    mean = float(gray.mean())
    beta = 127.0 - mean
    return cv2.convertScaleAbs(gray, alpha=1.0, beta=beta)


def gamma_correction(gray, gamma):
    inv_gamma = 1.0 / max(gamma, 0.01)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(gray, table)


def edge_enhance(gray):
    edges = cv2.Canny(gray, 50, 150)
    return cv2.addWeighted(gray, 0.8, edges, 0.2, 0)


def noise_removal(gray):
    return cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)


def deblur(gray):
    kernel = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]])
    return cv2.filter2D(gray, -1, kernel)


def upscale_and_sharpen(gray, scale=2):
    upscaled = resize_by_scale(gray, scale)
    return sharpen(upscaled)


def resize_by_scale(image, scale):
    height, width = image.shape[:2]
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)


def rotate_image(image, angle):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_width / 2.0) - center[0]
    matrix[1, 2] += (new_height / 2.0) - center[1]
    return cv2.warpAffine(image, matrix, (new_width, new_height), borderValue=255)


def barcode_format_name(barcode):
    try:
        return str(barcode.format).replace("BarcodeFormat.", "")
    except Exception:
        return "unknown"


def points_from_barcode(barcode):
    try:
        position = barcode.position
        return [
            [float(position.top_left.x), float(position.top_left.y)],
            [float(position.top_right.x), float(position.top_right.y)],
            [float(position.bottom_right.x), float(position.bottom_right.y)],
            [float(position.bottom_left.x), float(position.bottom_left.y)],
        ]
    except Exception:
        return None

