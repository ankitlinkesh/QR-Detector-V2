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


def normalize_mode(mode):
    return "smart" if mode == "robust" else mode


def generate_preprocessing_variants(image, mode="smart"):
    """Compatibility wrapper used by debug output."""
    mode = normalize_mode(mode)
    yield from generate_full_frame_variants(image, mode)

    if mode == "smart":
        yield from generate_fallback_tile_variants(image, max_tiles=24)


def generate_full_frame_variants(image, mode="smart"):
    mode = normalize_mode(mode)
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
    clahe_gray = apply_clahe(gray)

    yield ImageVariant(image=crop, name=f"{prefix}_original", source="zxing_candidate_crop")
    yield ImageVariant(image=to_bgr(gray), name=f"{prefix}_grayscale", source="zxing_candidate_gray")
    yield ImageVariant(image=to_bgr(clahe_gray), name=f"{prefix}_clahe", source="zxing_candidate_clahe")
    yield ImageVariant(image=sharpen(crop), name=f"{prefix}_sharpened", source="zxing_candidate_sharpened")

    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield ImageVariant(image=to_bgr(adaptive), name=f"{prefix}_adaptive_threshold", source="zxing_candidate_adaptive")
    yield ImageVariant(image=to_bgr(otsu), name=f"{prefix}_otsu_threshold", source="zxing_candidate_otsu")

    yield ImageVariant(
        image=to_bgr(add_quiet_zone(make_white_background(gray))),
        name=f"{prefix}_white_bg_otsu_quiet",
        source="zxing_candidate_white_bg_otsu_quiet",
    )
    yield ImageVariant(
        image=to_bgr(add_quiet_zone(make_white_background(adaptive))),
        name=f"{prefix}_adaptive_white_quiet",
        source="zxing_candidate_adaptive_white_quiet",
    )
    yield ImageVariant(
        image=to_bgr(add_quiet_zone(make_white_background(clahe_gray))),
        name=f"{prefix}_clahe_white_quiet",
        source="zxing_candidate_clahe_white_quiet",
    )

    for inner_variant in inner_square_crop_variants(crop, prefix):
        yield inner_variant

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
    records = []
    seen = set()

    for size_index, (tile_width, tile_height) in enumerate(tile_sizes):
        step_x = max(1, int(tile_width * (1 - overlap)))
        step_y = max(1, int(tile_height * (1 - overlap)))
        x_starts = tile_starts(image_width, tile_width, step_x)
        y_starts = tile_starts(image_height, tile_height, step_y)

        for anchor_index, (x, y) in enumerate(tile_anchor_starts(image_width, image_height, tile_width, tile_height)):
            add_tile_record(records, seen, image, x, y, tile_width, tile_height, size_index, anchor_index)

        for y in y_starts:
            for x in x_starts:
                add_tile_record(records, seen, image, x, y, tile_width, tile_height, size_index, None)

    for record in sorted(records, key=lambda item: item["sort_key"])[:max_tiles]:
        x = record["x"]
        y = record["y"]
        tile_width = record["tile_width"]
        tile_height = record["tile_height"]
        tile = image[y : min(y + tile_height, image_height), x : min(x + tile_width, image_width)]
        if tile.size == 0:
            continue

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


def tile_anchor_starts(image_width, image_height, tile_width, tile_height):
    max_x = max(0, image_width - tile_width)
    max_y = max(0, image_height - tile_height)
    center_x = max_x // 2
    center_y = max_y // 2

    anchors = [
        (center_x, center_y),
        (0, 0),
        (max_x, 0),
        (0, max_y),
        (max_x, max_y),
        (center_x, 0),
        (center_x, max_y),
        (0, center_y),
        (max_x, center_y),
    ]

    return [(clamp(x, 0, max_x), clamp(y, 0, max_y)) for x, y in anchors]


def add_tile_record(records, seen, image, x, y, tile_width, tile_height, size_index, anchor_index):
    image_height, image_width = image.shape[:2]
    x = clamp(x, 0, max(0, image_width - tile_width))
    y = clamp(y, 0, max(0, image_height - tile_height))
    key = (x, y, tile_width, tile_height)

    if key in seen:
        return

    seen.add(key)
    center_x = x + min(tile_width, image_width) / 2
    center_y = y + min(tile_height, image_height) / 2
    frame_center_x = image_width / 2
    frame_center_y = image_height / 2
    center_distance = ((center_x - frame_center_x) ** 2 + (center_y - frame_center_y) ** 2) ** 0.5
    texture = tile_texture_score(image, x, y, tile_width, tile_height)

    if anchor_index is None:
        priority = 100 + size_index
    else:
        priority = anchor_index

    records.append(
        {
            "x": x,
            "y": y,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "sort_key": (priority, size_index, -texture, center_distance),
        }
    )



def make_white_background(gray):
    if len(gray.shape) != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return remove_border_connected_dark(binary)


def remove_border_connected_dark(binary):
    cleaned = binary.copy()
    height, width = cleaned.shape[:2]
    mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    for x in range(width):
        if cleaned[0, x] == 0:
            cv2.floodFill(cleaned, mask, (x, 0), 255)
        if cleaned[height - 1, x] == 0:
            cv2.floodFill(cleaned, mask, (x, height - 1), 255)

    for y in range(height):
        if cleaned[y, 0] == 0:
            cv2.floodFill(cleaned, mask, (0, y), 255)
        if cleaned[y, width - 1] == 0:
            cv2.floodFill(cleaned, mask, (width - 1, y), 255)

    return cleaned


def add_quiet_zone(image, border_ratio=0.22, min_border=18):
    border = max(min_border, int(max(image.shape[:2]) * border_ratio))
    return cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=255)


def inner_square_crop_variants(crop, prefix="crop", max_variants=4):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe_gray = apply_clahe(gray)
    search_images = [gray, clahe_gray, make_white_background(gray)]
    records = []

    for search_index, search_image in enumerate(search_images):
        edges = cv2.Canny(search_image, 35, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image_area = crop.shape[0] * crop.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max(25.0, image_area * 0.002) or area > image_area * 0.65:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.045 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            x, y, width, height = cv2.boundingRect(approx)
            if width < 12 or height < 12:
                continue

            aspect = width / max(height, 1)
            if aspect < 0.65 or aspect > 1.55:
                continue

            score = area + 3000 * (1.0 - abs(1.0 - min(aspect, 1 / aspect)))
            records.append((score, x, y, width, height, search_index))

    seen = set()
    emitted = 0
    for _, x, y, width, height, search_index in sorted(records, reverse=True):
        key = (round(x / 8), round(y / 8), round(width / 8), round(height / 8))
        if key in seen:
            continue
        seen.add(key)

        pad = max(10, int(max(width, height) * 0.45))
        x1 = clamp(x - pad, 0, crop.shape[1])
        y1 = clamp(y - pad, 0, crop.shape[0])
        x2 = clamp(x + width + pad, 0, crop.shape[1])
        y2 = clamp(y + height + pad, 0, crop.shape[0])
        focused = crop[y1:y2, x1:x2]
        if focused.size == 0:
            continue

        focused_gray = cv2.cvtColor(focused, cv2.COLOR_BGR2GRAY)
        cleaned = add_quiet_zone(make_white_background(focused_gray), min_border=22)
        yield ImageVariant(
            image=to_bgr(cleaned),
            name=f"{prefix}_inner_square_{emitted + 1}_white_quiet",
            source=f"zxing_candidate_inner_square_{search_index}_white_quiet",
        )
        yield ImageVariant(
            image=resize_by_scale(to_bgr(cleaned), 2),
            name=f"{prefix}_inner_square_{emitted + 1}_white_quiet_2x",
            source=f"zxing_candidate_inner_square_{search_index}_white_quiet_2x",
            scale=2.0,
        )
        emitted += 1
        if emitted >= max_variants:
            break
def tile_texture_score(image, x, y, tile_width, tile_height):
    tile = image[y : min(y + tile_height, image.shape[0]), x : min(x + tile_width, image.shape[1])]
    if tile.size == 0:
        return 0.0

    thumb = cv2.resize(tile, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    return float(gray.std())


def clamp(value, low, high):
    return max(low, min(high, int(value)))


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
