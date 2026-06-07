import html
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class OutputWriter:
    def __init__(self, output_dir, save_crops=False, save_frames=False, debug=False):
        self.output_dir = Path(output_dir)
        self.annotated_dir = self.output_dir / "annotated"
        self.crops_dir = self.output_dir / "crops"
        self.frames_dir = self.output_dir / "frames"
        self.debug_dir = self.output_dir / "debug"
        self.save_crops_enabled = save_crops
        self.save_frames_enabled = save_frames
        self.debug = debug
        self.ensure_directories()

    def ensure_directories(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        if self.save_crops_enabled:
            self.crops_dir.mkdir(parents=True, exist_ok=True)

        if self.save_frames_enabled:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

    def save_frame(self, image, stem):
        if not self.save_frames_enabled:
            return None

        path = self.frames_dir / f"{stem}.png"
        cv2.imwrite(str(path), image)
        return self.relative(path)

    def save_crop(self, image, detection, stem, index):
        if not self.save_crops_enabled:
            return None

        x, y, width, height = cv2.boundingRect(np.array(detection["points"], dtype="float32"))
        pad = 12
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + width + pad)
        y2 = min(image.shape[0], y + height + pad)
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        path = self.crops_dir / f"{stem}_qr_{index}.png"
        cv2.imwrite(str(path), crop)
        return self.relative(path)

    def save_annotated(self, image, detections, stem, title, has_filter, candidates=None, raw_candidates=None, failure_reason=None):
        annotated = draw_annotations(
            image,
            detections,
            title,
            has_filter,
            candidates=candidates,
            raw_candidates=raw_candidates,
            failure_reason=failure_reason,
        )
        path = self.annotated_dir / f"{stem}.png"
        cv2.imwrite(str(path), annotated)
        return self.relative(path)

    def save_debug_variant(self, image, stem, variant_name):
        if not self.debug:
            return None

        safe_variant = "".join(character if character.isalnum() or character in "-_" else "_" for character in variant_name)
        path = self.debug_dir / f"{stem}_{safe_variant}.png"
        cv2.imwrite(str(path), image)
        return self.relative(path)

    def save_results_json(self, results):
        path = self.output_dir / "results.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
        return self.relative(path)

    def save_report_html(self, results):
        path = self.output_dir / "report.html"
        path.write_text(build_report_html(results), encoding="utf-8")
        return self.relative(path)

    def relative(self, path):
        return Path(path).relative_to(self.output_dir).as_posix()


def draw_annotations(image, detections, title, has_filter, candidates=None, raw_candidates=None, failure_reason=None):
    annotated = image.copy()
    header_height = 74
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (annotated.shape[1], header_height), (8, 8, 8), -1)
    annotated = cv2.addWeighted(overlay, 0.86, annotated, 0.14, 0)

    reason_text = failure_reason or "decoded"
    cv2.putText(annotated, title[:90], (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(
        annotated,
        f"Detections: {len(detections)} | Selected: {len(candidates or [])} | Raw: {len(raw_candidates or [])} | Reason: {reason_text}",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (190, 190, 190),
        2,
        cv2.LINE_AA,
    )

    draw_raw_candidate_boxes(annotated, raw_candidates or [], header_height)
    draw_candidate_boxes(annotated, candidates or [], header_height)

    if not detections:
        cv2.putText(
            annotated,
            f"No QR decoded: {reason_text}",
            (20, header_height + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 80, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    for index, detection in enumerate(detections, start=1):
        points = np.array(detection["points"], dtype=np.int32)
        color = choose_color(detection, has_filter)

        for point_index in range(len(points)):
            start = tuple(points[point_index])
            end = tuple(points[(point_index + 1) % len(points)])
            cv2.line(annotated, start, end, color, 3)

        center = tuple(int(value) for value in detection["center"])
        cv2.circle(annotated, center, 5, color, -1)
        label = f"{index}: {detection['cleaned_text'][:42]}"
        label_origin = (max(8, center[0] + 8), max(header_height + 24, center[1] - 8))
        cv2.putText(annotated, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(annotated, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return annotated



def draw_raw_candidate_boxes(annotated, candidates, header_height):
    for index, candidate in enumerate(candidates, start=1):
        points = np.array(candidate["points"], dtype=np.int32)
        color = (255, 210, 80)
        for point_index in range(len(points)):
            start = tuple(points[point_index])
            end = tuple(points[(point_index + 1) % len(points)])
            cv2.line(annotated, start, end, color, 1)

        center = tuple(int(value) for value in candidate["center"])
        label = f"raw {index}: {candidate['source']} {candidate.get('width', 0):.0f}x{candidate.get('height', 0):.0f}"
        label_origin = (max(8, center[0] + 6), max(header_height + 18, center[1] + 14))
        cv2.putText(annotated, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
def draw_candidate_boxes(annotated, candidates, header_height):
    for index, candidate in enumerate(candidates, start=1):
        points = np.array(candidate["points"], dtype=np.int32)
        color = (0, 165, 255)
        for point_index in range(len(points)):
            start = tuple(points[point_index])
            end = tuple(points[(point_index + 1) % len(points)])
            cv2.line(annotated, start, end, color, 2)

        center = tuple(int(value) for value in candidate["center"])
        label = f"candidate {index}: {candidate['source']}"
        label_origin = (max(8, center[0] + 8), max(header_height + 24, center[1] - 8))
        cv2.putText(annotated, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def choose_color(detection, has_filter):
    if detection["matched"] is True or detection["passes_filter"] is True and has_filter:
        return (0, 220, 0)

    if detection["passes_filter"] is False and has_filter:
        return (0, 0, 255)

    return (255, 220, 0)


def build_results(metadata, items):
    unique_texts = sorted(
        {
            detection["normalized_text"]
            for item in items
            for detection in item["detections"]
            if detection["normalized_text"]
        }
    )
    metadata = dict(metadata)
    metadata["timestamp"] = datetime.now().isoformat(timespec="seconds")
    metadata["total_detections"] = sum(len(item["detections"]) for item in items)
    metadata["total_unique_decoded_texts"] = len(unique_texts)
    metadata["total_near_misses"] = sum(len(item.get("near_misses", [])) for item in items)
    metadata["total_raw_candidates"] = sum(len(item.get("raw_candidates", [])) for item in items)
    metadata["failure_reason_counts"] = count_failure_reasons(items)

    return {
        "metadata": metadata,
        "unique_decoded_texts": unique_texts,
        "items": items,
    }


def count_failure_reasons(items):
    counts = {}

    for item in items:
        reason = item.get("failure_reason") or "unknown"
        counts[reason] = counts.get(reason, 0) + 1

    return counts


def build_report_html(results):
    metadata = results["metadata"]
    items_html = []

    for item in results["items"]:
        detections_html = []
        for detection in item["detections"]:
            status = "PASS" if detection["passes_filter"] else "FAIL"
            if detection["matched"] is True:
                status = "MATCH"
            elif detection["matched"] is False:
                status = "NO MATCH"

            detections_html.append(
                f"""
                <tr>
                  <td>{html.escape(status)}</td>
                  <td><pre>{html.escape(detection["cleaned_text"])}</pre></td>
                  <td>{html.escape(detection["detected_type"])}</td>
                  <td>{html.escape(detection["source"])}</td>
                  <td>{html.escape(detection["preprocessing_variant"])}</td>
                  <td>{html.escape(detection["filter_reason"])}</td>
                </tr>
                """
            )

        raw_count = len(item.get("raw_candidates", []))
        selected_count = len(item.get("candidates", []))
        annotated = item.get("annotated_output_path")
        preview = f'<img src="{html.escape(annotated)}" alt="Annotated result">' if annotated else ""
        items_html.append(
            f"""
            <section class="card">
              <h2>{html.escape(item["input_path"])}</h2>
              <p>Frame: {item.get("frame_index")} | Timestamp: {item.get("frame_timestamp_sec")} | Reason: {html.escape(str(item.get("failure_reason")))} | Raw candidates: {raw_count} | Selected candidates: {selected_count}</p>
              {preview}
              <table>
                <thead>
                  <tr><th>Status</th><th>Text</th><th>Type</th><th>Source</th><th>Variant</th><th>Reason</th></tr>
                </thead>
                <tbody>{''.join(detections_html)}</tbody>
              </table>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>QR Detection Report</title>
  <style>
    body {{ margin: 0; background: #080808; color: #e8e8e8; font-family: Arial, sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    h1 {{ color: #ffffff; }}
    .summary, .card {{ background: #151515; border: 1px solid #2b2b2b; border-radius: 8px; padding: 18px; margin: 18px 0; }}
    .accent {{ color: #40d46a; }}
    img {{ max-width: 100%; border: 1px solid #333; border-radius: 6px; margin: 12px 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border-bottom: 1px solid #303030; padding: 10px; text-align: left; vertical-align: top; }}
    th {{ color: #bdbdbd; }}
    pre {{ white-space: pre-wrap; margin: 0; font-family: Consolas, monospace; }}
  </style>
</head>
<body>
<main>
  <h1>QR Detection Report</h1>
  <section class="summary">
    <p><span class="accent">Input:</span> {html.escape(str(metadata.get("input_path")))}</p>
    <p><span class="accent">Mode:</span> {html.escape(str(metadata.get("mode")))}</p>
    <p><span class="accent">Total detections:</span> {metadata.get("total_detections")}</p>
    <p><span class="accent">Unique decoded texts:</span> {metadata.get("total_unique_decoded_texts")}</p>
    <p><span class="accent">Near misses:</span> {metadata.get("total_near_misses")}</p>
    <p><span class="accent">Raw candidates:</span> {metadata.get("total_raw_candidates")}</p>
    <p><span class="accent">Failure reasons:</span> {html.escape(str(metadata.get("failure_reason_counts")))}</p>
    <p><span class="accent">Filters:</span> {html.escape(str(metadata.get("filters_used")))}</p>
  </section>
  {''.join(items_html)}
</main>
</body>
</html>"""

