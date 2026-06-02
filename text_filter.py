import json
import re
from dataclasses import dataclass


@dataclass
class TextFilterOptions:
    equals: str | None = None
    contains: str | None = None
    regex: str | None = None
    min_length: int | None = None
    max_length: int | None = None
    unique_text_only: bool = False


def clean_text(text):
    if text is None:
        return ""

    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    return "".join(character for character in cleaned if character == "\n" or character == "\t" or ord(character) >= 32)


def normalize_text(text):
    cleaned = clean_text(text)
    normalized_lines = [line.strip() for line in cleaned.splitlines()]
    return "\n".join(normalized_lines).strip().lower()


def split_lines(text):
    return [line for line in clean_text(text).splitlines() if line.strip()]


def extract_key_value_pairs(text):
    pairs = {}

    for line in split_lines(text):
        match = re.match(r"^\s*([^:=]+)\s*[:=]\s*(.+?)\s*$", line)
        if match:
            pairs[match.group(1).strip()] = match.group(2).strip()

    if not pairs and looks_json_like(text):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                pairs = {str(key): str(value) for key, value in parsed.items()}
        except json.JSONDecodeError:
            pass

    return pairs


def detect_text_type(text):
    cleaned = clean_text(text)
    normalized = normalize_text(cleaned)

    if not cleaned:
        return "unknown"

    if normalized.startswith("http://") or normalized.startswith("https://"):
        return "url"

    if re.search(r"\bdelivery[_ -]?zone[_ -]?[a-z0-9]+", normalized):
        return "delivery_zone"

    if looks_json_like(cleaned):
        return "json_like"

    if extract_key_value_pairs(cleaned):
        return "key_value"

    return "plain_text"


def looks_json_like(text):
    stripped = clean_text(text)
    return (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]"))


class QRTextFilter:
    def __init__(self, options: TextFilterOptions):
        self.options = options
        self.regex = re.compile(options.regex) if options.regex else None
        self.equals = normalize_text(options.equals) if options.equals else None
        self.contains = normalize_text(options.contains) if options.contains else None
        self.seen_texts = set()

    def enrich_detection(self, detection):
        raw_text = detection.data
        cleaned_text = clean_text(raw_text)
        normalized = normalize_text(cleaned_text)
        matched = None

        if self.equals is not None:
            matched = normalized == self.equals

        passes_filter, filter_reason = self.evaluate(cleaned_text, normalized)

        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "normalized_text": normalized,
            "lines": split_lines(cleaned_text),
            "possible_key_value_pairs": extract_key_value_pairs(cleaned_text),
            "detected_type": detect_text_type(cleaned_text),
            "matched": matched,
            "passes_filter": passes_filter,
            "filter_reason": filter_reason,
            "points": detection.points.astype(float).round(2).tolist(),
            "center": detection.center,
            "width": detection.width,
            "height": detection.height,
            "area": detection.area,
            "source": detection.source,
            "confidence": detection.confidence,
            "preprocessing_variant": detection.preprocessing_variant,
            "merged_sources": detection.merged_sources,
            "crop_path": None,
        }

    def evaluate(self, cleaned_text, normalized):
        reasons = []
        has_filter = any(
            [
                self.equals is not None,
                self.contains is not None,
                self.regex is not None,
                self.options.min_length is not None,
                self.options.max_length is not None,
                self.options.unique_text_only,
            ]
        )

        if self.equals is not None and normalized != self.equals:
            return False, "does not equal requested text"

        if self.contains is not None and self.contains not in normalized:
            return False, "does not contain requested text"

        if self.regex is not None and not self.regex.search(cleaned_text):
            return False, "does not match regex"

        if self.options.min_length is not None and len(cleaned_text) < self.options.min_length:
            return False, f"shorter than min length {self.options.min_length}"

        if self.options.max_length is not None and len(cleaned_text) > self.options.max_length:
            return False, f"longer than max length {self.options.max_length}"

        if self.options.unique_text_only:
            if normalized in self.seen_texts:
                return False, "duplicate decoded text"
            self.seen_texts.add(normalized)

        if not has_filter:
            return True, "no filters applied"

        if self.equals is not None:
            reasons.append("equals matched")
        if self.contains is not None:
            reasons.append("contains matched")
        if self.regex is not None:
            reasons.append("regex matched")

        return True, ", ".join(reasons) if reasons else "passed filters"
