import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from frame_recovery import FrameRanker, decode_recovery_variants
from output_writer import OutputWriter, build_results
from preprocessing import crop_with_padding, generate_preprocessing_variants
from qr_detector import QRDetector, make_detection
from text_filter import QRTextFilter, TextFilterOptions
from video_utils import iter_video_frames


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser(description="Offline smart QR/barcode detection and text extraction pipeline.")
    parser.add_argument("--input", required=True, help="Input image, video, or folder.")
    parser.add_argument("--mode", choices=["fast", "smart", "robust"], default="smart", help="Detection mode. robust is kept as an alias for smart.")
    parser.add_argument("--frame-step", type=int, default=5, help="Process every Nth frame for videos.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum processed frames per video.")
    parser.add_argument("--output", default="outputs/", help="Output directory.")
    parser.add_argument("--save-crops", action="store_true", help="Save detected QR/barcode crops.")
    parser.add_argument("--save-frames", action="store_true", help="Save processed video frames.")
    parser.add_argument("--equals", default=None, help="Mark detections that exactly match this text.")
    parser.add_argument("--contains", default=None, help="Mark detections containing this text.")
    parser.add_argument("--regex", default=None, help="Mark detections matching this regular expression.")
    parser.add_argument("--min-length", type=int, default=None, help="Minimum decoded text length.")
    parser.add_argument("--max-length", type=int, default=None, help="Maximum decoded text length.")
    parser.add_argument("--unique-text-only", action="store_true", help="Only pass the first detection for each text.")
    parser.add_argument("--debug", action="store_true", help="Print detailed detector progress.")
    parser.add_argument("--report", action="store_true", help="Generate a dark static HTML report.")
    parser.add_argument("--profile", action="store_true", help="Print candidate/decode attempt timing for each item.")
    parser.add_argument("--max-variants-per-frame", type=int, default=160, help="Cap smart decode attempts per frame.")
    parser.add_argument("--max-candidates", type=int, default=24, help="Maximum QR-like candidate regions per frame.")
    parser.add_argument("--no-temporal-rescue", action="store_true", help="Disable the bounded near-miss video rescue pass.")
    parser.add_argument("--rescue-window", type=int, default=2, help="Frames before/after a near-miss to retry in smart video mode.")
    parser.add_argument("--max-rescue-targets", type=int, default=8, help="Maximum candidate tracks for temporal rescue per video.")
    parser.add_argument("--no-frame-recovery", action="store_true", help="Disable ranked whole-frame enhancement recovery for hard videos.")
    parser.add_argument("--recovery-top-frames", type=int, default=10, help="Maximum ranked frames to try with whole-frame recovery.")
    parser.add_argument("--recovery-max-variants", type=int, default=0, help="Maximum whole-frame recovery variants per ranked frame. Use 0 for all variants.")
    parser.add_argument("--recovery-max-side", type=int, default=1600, help="Resize ranked recovery frames so the longest side is at most this many pixels. Use 0 for original size.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}")
        return 1

    if args.frame_step < 1:
        print("[ERROR] --frame-step must be at least 1.")
        return 1

    if args.rescue_window < 0:
        print("[ERROR] --rescue-window must be zero or greater.")
        return 1

    if args.recovery_top_frames < 1:
        print("[ERROR] --recovery-top-frames must be at least 1.")
        return 1

    if args.recovery_max_variants < 0:
        print("[ERROR] --recovery-max-variants must be zero or greater.")
        return 1

    if args.recovery_max_side < 0:
        print("[ERROR] --recovery-max-side must be zero or greater.")
        return 1

    try:
        detector = QRDetector(
            mode=args.mode,
            debug=args.debug,
            profile=args.profile,
            max_variants_per_frame=args.max_variants_per_frame,
            max_candidates=args.max_candidates,
        )
    except RuntimeError as error:
        print(f"[ERROR] {error}")
        return 1

    filter_options = TextFilterOptions(
        equals=args.equals,
        contains=args.contains,
        regex=args.regex,
        min_length=args.min_length,
        max_length=args.max_length,
        unique_text_only=args.unique_text_only,
    )
    text_filter = QRTextFilter(filter_options)
    writer = OutputWriter(args.output, save_crops=args.save_crops, save_frames=args.save_frames, debug=args.debug)

    files = collect_input_files(input_path)
    if not files:
        print(f"[ERROR] No supported image or video files found in: {input_path}")
        return 1

    items = []
    total_frames_processed = 0
    total_files_processed = 0

    print(f"[INFO] Found {len(files)} supported input file(s).")
    if args.mode == "robust":
        print("[INFO] --mode robust is an alias for --mode smart.")

    for file_index, file_path in enumerate(files, start=1):
        print(f"[INFO] Processing {file_index}/{len(files)}: {file_path}")
        suffix = file_path.suffix.lower()

        try:
            if suffix in IMAGE_EXTENSIONS:
                item = process_image(file_path, detector, text_filter, writer, args)
                items.append(item)
                total_files_processed += 1
            elif suffix in VIDEO_EXTENSIONS:
                video_items, frame_count = process_video(file_path, detector, text_filter, writer, args)
                items.extend(video_items)
                total_frames_processed += frame_count
                total_files_processed += 1
        except Exception as error:
            print(f"[WARN] Skipping {file_path}: {error}")

    metadata = {
        "input_path": str(input_path),
        "mode": detector.mode,
        "requested_mode": args.mode,
        "frame_step": args.frame_step,
        "max_frames": args.max_frames,
        "temporal_rescue_enabled": detector.mode == "smart" and not args.no_temporal_rescue,
        "rescue_window": args.rescue_window,
        "max_rescue_targets": args.max_rescue_targets,
        "frame_recovery_enabled": detector.mode == "smart" and not args.no_frame_recovery,
        "recovery_top_frames": args.recovery_top_frames,
        "recovery_max_variants": args.recovery_max_variants,
        "recovery_max_side": args.recovery_max_side,
        "filters_used": {
            "equals": args.equals,
            "contains": args.contains,
            "regex": args.regex,
            "min_length": args.min_length,
            "max_length": args.max_length,
            "unique_text_only": args.unique_text_only,
        },
        "total_files_processed": total_files_processed,
        "total_frames_processed": total_frames_processed,
        "max_variants_per_frame": args.max_variants_per_frame,
        "max_candidates": args.max_candidates,
    }
    results = build_results(metadata, items)
    results_path = writer.save_results_json(results)

    print(f"[INFO] Wrote {results_path}")

    if args.report:
        report_path = writer.save_report_html(results)
        print(f"[INFO] Wrote {report_path}")

    print(f"[INFO] Total detections: {results['metadata']['total_detections']}")
    print(f"[INFO] Unique decoded texts: {results['metadata']['total_unique_decoded_texts']}")
    print(f"[INFO] Near misses: {results['metadata']['total_near_misses']}")
    return 0


def collect_input_files(input_path):
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS else []

    files = []
    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            files.append(path)

    return files


def process_image(file_path, detector, text_filter, writer, args):
    image = cv2.imread(str(file_path))

    if image is None:
        raise ValueError("image could not be read")

    stem = safe_stem(file_path)
    return process_image_data(
        image=image,
        input_path=file_path,
        frame_index=None,
        frame_timestamp_sec=None,
        stem=stem,
        detector=detector,
        text_filter=text_filter,
        writer=writer,
        args=args,
    )


def process_video(file_path, detector, text_filter, writer, args):
    items = []
    frame_count = 0
    rescue_tracks = []
    sampled_frames = []

    try:
        frames = iter_video_frames(file_path, frame_step=args.frame_step, max_frames=args.max_frames)
        for video_frame in frames:
            frame_count += 1
            if args.debug and frame_count % 10 == 0:
                print(f"[DEBUG] Processed {frame_count} sampled frame(s) from {file_path}")

            stem = f"{safe_stem(file_path)}_frame_{video_frame.frame_index:06d}"
            writer.save_frame(video_frame.image, stem)
            sampled_frames.append((video_frame.frame_index, video_frame.image.copy(), video_frame.timestamp_sec))
            item = process_image_data(
                image=video_frame.image,
                input_path=file_path,
                frame_index=video_frame.frame_index,
                frame_timestamp_sec=video_frame.timestamp_sec,
                stem=stem,
                detector=detector,
                text_filter=text_filter,
                writer=writer,
                args=args,
            )
            items.append(item)
            collect_rescue_tracks(rescue_tracks, item, video_frame.image, args.max_rescue_targets)
    except ValueError as error:
        print(f"[WARN] {error}")

    if detector.mode == "smart" and not args.no_temporal_rescue and rescue_tracks:
        rescue_items = run_temporal_rescue(file_path, rescue_tracks, detector, text_filter, writer, args)
        items.extend(rescue_items)

    if detector.mode == "smart" and not args.no_frame_recovery and not video_has_detection(items):
        recovery_items = run_frame_recovery(file_path, sampled_frames, detector, text_filter, writer, args)
        items.extend(recovery_items)

    return items, frame_count


def process_image_data(image, input_path, frame_index, frame_timestamp_sec, stem, detector, text_filter, writer, args):
    detections = detector.detect(image)
    return build_item_from_detections(
        image=image,
        input_path=input_path,
        frame_index=frame_index,
        frame_timestamp_sec=frame_timestamp_sec,
        stem=stem,
        detections=detections,
        detector=detector,
        text_filter=text_filter,
        writer=writer,
        args=args,
        rescue=False,
        rescue_source_frame_index=None,
    )


def build_item_from_detections(
    image,
    input_path,
    frame_index,
    frame_timestamp_sec,
    stem,
    detections,
    detector,
    text_filter,
    writer,
    args,
    rescue,
    rescue_source_frame_index,
):
    enriched_detections = []

    if args.debug and not rescue:
        save_debug_variants(image, stem, writer, detector.mode)

    for index, detection in enumerate(detections, start=1):
        detection.input_path = str(input_path)
        detection.frame_index = frame_index
        enriched = text_filter.enrich_detection(detection)
        enriched["crop_path"] = writer.save_crop(image, enriched, stem, index)
        enriched_detections.append(enriched)

    title = str(input_path.name) if frame_index is None else f"{input_path.name} frame {frame_index}"
    if rescue:
        title = f"{title} rescue"

    candidates = getattr(detector, "last_candidates", [])
    raw_candidates = getattr(detector, "last_raw_candidates", [])
    failure_reason = getattr(detector, "last_failure_reason", None)
    annotated_path = writer.save_annotated(
        image,
        enriched_detections,
        stem,
        title,
        has_filter(args),
        candidates=candidates,
        raw_candidates=raw_candidates,
        failure_reason=failure_reason,
    )

    if args.debug:
        texts = [detection["cleaned_text"] for detection in enriched_detections]
        print(f"[DEBUG] {title}: {len(enriched_detections)} QR/barcode detection(s): {texts}")

    return {
        "input_path": str(input_path),
        "frame_index": frame_index,
        "frame_timestamp_sec": frame_timestamp_sec,
        "annotated_output_path": annotated_path,
        "detections": enriched_detections,
        "profile": getattr(detector, "last_profile", {}),
        "candidates": candidates,
        "raw_candidates": raw_candidates,
        "raw_opencv_candidates": getattr(detector, "last_raw_opencv_candidates", []),
        "raw_contour_candidates": getattr(detector, "last_raw_contour_candidates", []),
        "near_misses": getattr(detector, "last_near_misses", []),
        "failure_reason": failure_reason,
        "rescue": rescue,
        "rescue_source_frame_index": rescue_source_frame_index,
    }


def collect_rescue_tracks(tracks, item, image, max_targets):
    if max_targets <= 0 or item["detections"] or item.get("frame_index") is None:
        return

    targets = []
    for near_miss in item.get("near_misses", []):
        if near_miss.get("points"):
            base_score = 2_000_000 if near_miss.get("reason") == "checksum_near_miss" else 1_500_000
            targets.append(
                {
                    "points": near_miss["points"],
                    "source": near_miss.get("source", "near_miss"),
                    "score": base_score,
                    "frame_index": item["frame_index"],
                    "frame_timestamp_sec": item.get("frame_timestamp_sec"),
                    "reason": near_miss.get("reason"),
                }
            )

    for candidate in item.get("candidates", []):
        if candidate.get("points"):
            targets.append(
                {
                    "points": candidate["points"],
                    "source": candidate.get("source", "candidate"),
                    "score": float(candidate.get("score", 0)),
                    "frame_index": item["frame_index"],
                    "frame_timestamp_sec": item.get("frame_timestamp_sec"),
                    "reason": item.get("failure_reason"),
                }
            )

    for target in targets:
        points = np.array(target["points"], dtype="float32")
        target["center"] = points_center(points)
        target["sharpness"] = candidate_sharpness(image, points)
        target["score"] += target["sharpness"]
        add_or_update_track(tracks, target, max_targets)


def add_or_update_track(tracks, target, max_targets):
    for index, existing in enumerate(tracks):
        if point_distance(existing["center"], target["center"]) < 120:
            if target["score"] > existing["score"]:
                tracks[index] = target
            tracks.sort(key=lambda item: item["score"], reverse=True)
            del tracks[max_targets:]
            return

    tracks.append(target)
    tracks.sort(key=lambda item: item["score"], reverse=True)
    del tracks[max_targets:]


def run_temporal_rescue(file_path, tracks, detector, text_filter, writer, args):
    capture = cv2.VideoCapture(str(file_path))
    if not capture.isOpened():
        return []

    rescue_items = []
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    rescue_budget = min(args.max_variants_per_frame, 120)

    try:
        for track_index, track in enumerate(tracks, start=1):
            print(
                f"[INFO] Temporal rescue {track_index}/{len(tracks)} around frame "
                f"{track['frame_index']} ({track.get('reason')})"
            )
            candidate = {"points": track["points"], "source": track["source"], "score": track["score"]}

            for frame_index in rescue_frame_indices(track["frame_index"], total_frames, args.rescue_window):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue

                detections = detector.decode_known_candidates(frame, [candidate], max_variants=rescue_budget)
                if not detections:
                    continue

                timestamp = frame_index / fps if fps and fps > 0 else None
                stem = f"{safe_stem(file_path)}_frame_{frame_index:06d}_rescue_{track_index:02d}"
                item = build_item_from_detections(
                    image=frame,
                    input_path=file_path,
                    frame_index=frame_index,
                    frame_timestamp_sec=timestamp,
                    stem=stem,
                    detections=detections,
                    detector=detector,
                    text_filter=text_filter,
                    writer=writer,
                    args=args,
                    rescue=True,
                    rescue_source_frame_index=track["frame_index"],
                )
                rescue_items.append(item)
                print(f"[INFO] Temporal rescue decoded {len(detections)} QR(s) on frame {frame_index}.")
                break
    finally:
        capture.release()

    return rescue_items


def run_frame_recovery(file_path, sampled_frames, detector, text_filter, writer, args):
    if not sampled_frames:
        return []

    ranker = FrameRanker()
    ranked = ranker.rank([(frame_index, image) for frame_index, image, _ in sampled_frames], top_n=args.recovery_top_frames)
    frame_by_index = {frame_index: (image, timestamp) for frame_index, image, timestamp in sampled_frames}
    recovery_items = []

    print(f"[INFO] Whole-frame recovery trying top {len(ranked)} ranked frame(s).")
    for ranked_frame in ranked:
        image, timestamp = frame_by_index[ranked_frame.frame_index]
        recovery_results = decode_recovery_variants(
            image,
            frame_index=ranked_frame.frame_index,
            max_variants=recovery_variant_limit(args),
            max_side=recovery_max_side(args),
        )
        if not recovery_results:
            continue

        detections = [recovery_result_to_detection(result, image) for result in recovery_results]
        detections = [detection for detection in detections if detection is not None]
        if not detections:
            continue

        detector.last_candidates = []
        detector.last_raw_candidates = []
        detector.last_raw_opencv_candidates = []
        detector.last_raw_contour_candidates = []
        detector.last_near_misses = []
        detector.last_failure_reason = "decoded"
        detector.last_profile = {
            "elapsed_sec": None,
            "candidate_count": 0,
            "candidate_attempts": 0,
            "decode_attempts": None,
            "near_misses": 0,
            "failure_reason": "decoded",
            "detections": len(detections),
            "frame_recovery": True,
            "frame_score": ranked_frame.score,
        }
        stem = f"{safe_stem(file_path)}_frame_{ranked_frame.frame_index:06d}_frame_recovery"
        item = build_item_from_detections(
            image=image,
            input_path=file_path,
            frame_index=ranked_frame.frame_index,
            frame_timestamp_sec=timestamp,
            stem=stem,
            detections=detections,
            detector=detector,
            text_filter=text_filter,
            writer=writer,
            args=args,
            rescue=True,
            rescue_source_frame_index=None,
        )
        item["frame_recovery"] = True
        item["frame_recovery_score"] = ranked_frame.score
        item["frame_recovery_results"] = recovery_results
        recovery_items.append(item)
        print(f"[INFO] Whole-frame recovery decoded {len(detections)} barcode(s) on frame {ranked_frame.frame_index}.")
        break

    return recovery_items



def recovery_variant_limit(args):
    if args.recovery_max_variants == 0:
        return None
    return args.recovery_max_variants


def recovery_max_side(args):
    if args.recovery_max_side == 0:
        return None
    return args.recovery_max_side

def recovery_result_to_detection(result, image):
    points = result.get("points")
    if points is None:
        height, width = image.shape[:2]
        points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    else:
        points = np.array(points, dtype="float32")

    source = f"{result.get('source', 'zxing_recovery')}_{result.get('format', 'unknown')}"
    return make_detection(
        data=result["text"],
        points=points,
        source=source,
        preprocessing_variant=result.get("variant", "frame_recovery"),
        confidence=None,
    )


def video_has_detection(items):
    return any(item.get("detections") for item in items)


def rescue_frame_indices(anchor, total_frames, window):
    offsets = [0]
    for offset in range(1, window + 1):
        offsets.extend([-offset, offset])

    seen = set()
    for offset in offsets:
        frame_index = anchor + offset
        if frame_index in seen or frame_index < 0 or frame_index >= total_frames:
            continue
        seen.add(frame_index)
        yield frame_index


def candidate_sharpness(image, points):
    try:
        crop, _, _, _ = crop_with_padding(image, points, padding_ratio=0.75, min_padding=60)
    except cv2.error:
        return 0.0

    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var() + gray.std())


def points_center(points):
    return [float(points[:, 0].mean()), float(points[:, 1].mean())]


def point_distance(left, right):
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5


def has_filter(args):
    return any(
        [
            args.equals,
            args.contains,
            args.regex,
            args.min_length is not None,
            args.max_length is not None,
            args.unique_text_only,
        ]
    )


def safe_stem(path):
    return Path(path).stem.replace(" ", "_")


def save_debug_variants(image, stem, writer, mode):
    for index, variant in enumerate(generate_preprocessing_variants(image, mode), start=1):
        if index > 40:
            print(f"[DEBUG] Saved first 40 debug variants for {stem}; remaining variants skipped.")
            break
        writer.save_debug_variant(variant.image, stem, f"{index:02d}_{variant.name}")


if __name__ == "__main__":
    sys.exit(main())
