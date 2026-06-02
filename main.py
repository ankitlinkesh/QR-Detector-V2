import argparse
import sys
from pathlib import Path

import cv2

from output_writer import OutputWriter, build_results
from preprocessing import generate_preprocessing_variants
from qr_detector import QRDetector
from text_filter import QRTextFilter, TextFilterOptions
from video_utils import iter_video_frames


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser(description="Offline robust QR detection and text extraction pipeline.")
    parser.add_argument("--input", required=True, help="Input image, video, or folder.")
    parser.add_argument("--mode", choices=["fast", "robust"], default="robust", help="Detection mode.")
    parser.add_argument("--frame-step", type=int, default=5, help="Process every Nth frame for videos.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum processed frames per video.")
    parser.add_argument("--output", default="outputs/", help="Output directory.")
    parser.add_argument("--save-crops", action="store_true", help="Save detected QR crops.")
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
    parser.add_argument("--max-variants-per-frame", type=int, default=160, help="Cap robust decode attempts per frame.")
    parser.add_argument("--max-candidates", type=int, default=24, help="Maximum QR-like candidate regions per frame.")
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

    try:
        detector = QRDetector(mode=args.mode, debug=args.debug, profile=args.profile, max_variants_per_frame=args.max_variants_per_frame, max_candidates=args.max_candidates)
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
        "mode": args.mode,
        "frame_step": args.frame_step,
        "max_frames": args.max_frames,
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

    try:
        frames = iter_video_frames(file_path, frame_step=args.frame_step, max_frames=args.max_frames)
        for video_frame in frames:
            frame_count += 1
            if args.debug and frame_count % 10 == 0:
                print(f"[DEBUG] Processed {frame_count} sampled frame(s) from {file_path}")

            stem = f"{safe_stem(file_path)}_frame_{video_frame.frame_index:06d}"
            writer.save_frame(video_frame.image, stem)
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
    except ValueError as error:
        print(f"[WARN] {error}")

    return items, frame_count


def process_image_data(image, input_path, frame_index, frame_timestamp_sec, stem, detector, text_filter, writer, args):
    detections = detector.detect(image)
    enriched_detections = []

    if args.debug:
        save_debug_variants(image, stem, writer, args.mode)

    for index, detection in enumerate(detections, start=1):
        detection.input_path = str(input_path)
        detection.frame_index = frame_index
        enriched = text_filter.enrich_detection(detection)
        enriched["crop_path"] = writer.save_crop(image, enriched, stem, index)
        enriched_detections.append(enriched)

    has_filter = any(
        [
            args.equals,
            args.contains,
            args.regex,
            args.min_length is not None,
            args.max_length is not None,
            args.unique_text_only,
        ]
    )
    title = str(input_path.name) if frame_index is None else f"{input_path.name} frame {frame_index}"
    candidates = getattr(detector, "last_candidates", [])
    annotated_path = writer.save_annotated(image, enriched_detections, stem, title, has_filter, candidates=candidates)

    if args.debug:
        texts = [detection["cleaned_text"] for detection in enriched_detections]
        print(f"[DEBUG] {title}: {len(enriched_detections)} QR detection(s): {texts}")

    return {
        "input_path": str(input_path),
        "frame_index": frame_index,
        "frame_timestamp_sec": frame_timestamp_sec,
        "annotated_output_path": annotated_path,
        "detections": enriched_detections,
        "profile": getattr(detector, "last_profile", {}),
        "candidates": candidates,
    }


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



