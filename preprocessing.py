from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ImageVariant:
    image: np.ndarray
    name: str
    source: str
    scale: float = 1.0
    offset_x: int = 0
    offset_y: int = 0


def generate_preprocessing_variants(image, mode="robust"):
    """Compatibility wrapper used by debug output."""
    yield from generate_full_frame_variants(image, mode)

    if mode == "robust":
        yield from generate_fallback_tile_variants(image, max_tiles=12)


def generate_full_frame_variants(image, mode="robust"):
    yield ImageVariant(image=image, name="original", source="zxing_full_original")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    yield ImageVariant(image=to_bgr(gray), name="grayscale", source="zxing_full_grayscale")

    if mode == "fast":
        return

    clahe_gray = apply_clahe(gray)
    yield ImageVariant(image=to_bgr(clahe_gray), name="clahe", source="zxing_full_clahe")
    yield ImageVariant(image=sharpen(image), name="sharpened", source="zxing_full_sharpened")
    yield ImageVariant(
        image=cv2.convertScaleAbs(image, alpha=1.35, beta=10),
        name="contrast",
        source="zxing_full_contrast",
    )


def generate_crop_variants(crop, prefix="crop"):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    yield ImageVariant(image=crop, name=f"{prefix}_original", source="zxing_candidate_crop")
    yield ImageVariant(image=to_bgr(gray), name=f"{prefix}_grayscale", source="zxing_candidate_gray")
    yield ImageVariant(image=to_bgr(apply_clahe(gray)), name=f"{prefix}_clahe", source="zxing_candidate_clahe")
    yield ImageVariant(image=sharpen(crop), name=f"{prefix}_sharpened", source="zxing_candidate_sharpened")

    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield ImageVariant(image=to_bgr(adaptive), name=f"{prefix}_adaptive_threshold", source="zxing_candidate_adaptive")
    yield ImageVariant(image=to_bgr(otsu), name=f"{prefix}_otsu_threshold", source="zxing_candidate_otsu")

    for scale in (2, 3, 4):
        upscaled = resize_by_scale(crop, scale)
        upscaled_gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        yield ImageVariant(
            image=upscaled,
            name=f"{prefix}_upscaled_{scale}x",
            source=f"zxing_candidate_upscale_{scale}x",
            scale=float(scale),
        )
        yield ImageVariant(
            image=to_bgr(apply_clahe(upscaled_gray)),
            name=f"{prefix}_upscaled_{scale}x_clahe",
            source=f"zxing_candidate_upscale_{scale}x_clahe",
            scale=float(scale),
        )
        yield ImageVariant(
            image=sharpen(upscaled),
            name=f"{prefix}_upscaled_{scale}x_sharpen",
            source=f"zxing_candidate_upscale_{scale}x_sharpen",
            scale=float(scale),
        )


def generate_fallback_tile_variants(image, max_tiles=24, overlap=0.35):
    image_height, image_width = image.shape[:2]
    tile_sizes = [(900, 900), (700, 700), (520, 520)]
    yielded = 0

    for tile_width, tile_height in tile_sizes:
        step_x = max(1, int(tile_width * (1 - overlap)))
        step_y = max(1, int(tile_height * (1 - overlap)))

        for y in tile_starts(image_height, tile_height, step_y):
            for x in tile_starts(image_width, tile_width, step_x):
                if yielded >= max_tiles:
                    return

                tile = image[y : min(y + tile_height, image_height), x : min(x + tile_width, image_width)]
                if tile.size == 0:
                    continue

                yielded += 1
                name = f"fallback_tile_{tile_width}x{tile_height}_{x}_{y}"
                yield ImageVariant(
                    image=tile,
                    name=name,
                    source=f"zxing_fallback_tile_{tile_width}x{tile_height}",
                    offset_x=x,
                    offset_y=y,
                )


def crop_with_padding(image, points, padding_ratio=0.35, min_padding=24):
    height, width = image.shape[:2]
    x, y, box_width, box_height = cv2.boundingRect(points.astype("float32"))
    padding = max(min_padding, int(max(box_width, box_height) * padding_ratio))

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + box_width + padding)
    y2 = min(height, y + box_height + padding)

    crop = image[y1:y2, x1:x2]
    translated_points = points.astype("float32").copy()
    translated_points[:, 0] -= x1
    translated_points[:, 1] -= y1

    return crop, translated_points, x1, y1


def perspective_rectify(image, points, output_size=640, border=48):
    ordered_points = order_points(points.astype("float32"))
    destination = np.array(
        [
            [border, border],
            [output_size - border - 1, border],
            [output_size - border - 1, output_size - border - 1],
            [border, output_size - border - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(ordered_points, destination)
    return cv2.warpPerspective(image, matrix, (output_size, output_size))


def order_points(points):
    ordered = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]

    return ordered


def tile_starts(length, tile_length, step):
    if length <= tile_length:
        return [0]

    starts = list(range(0, length - tile_length + 1, step))
    final_start = length - tile_length

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def resize_by_scale(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
