# QR-Detector

An offline Python pipeline for detecting and decoding QR codes in images, videos, and folders. It uses OpenCV for image/video handling and annotation, ZXing-C++ as the primary QR decoder, and robust preprocessing to improve detection of small, blurry, low-contrast, angled, or far-away QR codes.

Live webcam mode was removed because QR recovery is stronger offline: the pipeline can locate likely QR quadrilaterals, crop and rectify them, enhance only those regions, merge duplicates, save debug outputs, and generate structured results without real-time frame-rate pressure.

This project does not include drone control, MAVLink, Flask, databases, cloud APIs, ML training, model downloads, or web apps.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If using the installed Python directly on this machine:

```powershell
C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe -m pip install -r requirements.txt
```

## Commands

Single image:

```powershell
python main.py --input inputs/images/test.png
```

Video:

```powershell
python main.py --input inputs/videos/test.mp4
```

Folder:

```powershell
python main.py --input inputs/images/
```

Robust mode:

```powershell
python main.py --input inputs/ --mode robust
```

Video frame sampling:

```powershell
python main.py --input inputs/videos/test.mp4 --frame-step 5 --max-frames 200
```

Text contains filter:

```powershell
python main.py --input inputs/images/test.png --contains DELIVERY_ZONE
```

Exact text matching:

```powershell
python main.py --input inputs/images/test.png --equals DELIVERY_ZONE_A3
```

Regex filtering:

```powershell
python main.py --input inputs/images/test.png --regex "DELIVERY_ZONE_[A-Z][0-9]"
```

Save QR crops:

```powershell
python main.py --input inputs/images/test.png --save-crops
```

Generate the dark HTML report:

```powershell
python main.py --input inputs/images/test.png --report
```

## CLI Options

- `--input`: required image, video, or folder.
- `--mode`: `fast` or `robust`. Default is `robust`.
- `--frame-step`: process every Nth video frame. Default is `5`.
- `--max-frames`: optional limit for sampled video frames.
- `--output`: output folder. Default is `outputs/`.
- `--save-crops`: save detected QR crops.
- `--save-frames`: save processed video frames.
- `--equals`: mark exact target text matches.
- `--contains`: mark detections containing text.
- `--regex`: mark detections matching a pattern.
- `--min-length` / `--max-length`: filter decoded text by length.
- `--unique-text-only`: only pass the first detection for each decoded text.
- `--debug`: print detailed progress.
- `--report`: generate `report.html`.

## Output Folder

By default, results are written under `outputs/`:

- `outputs/annotated/`: annotated image/frame results.
- `outputs/crops/`: QR crops when `--save-crops` is enabled.
- `outputs/frames/`: processed video frames when `--save-frames` is enabled.
- `outputs/debug/`: reserved for debug assets.
- `outputs/results.json`: structured machine-readable results.
- `outputs/report.html`: optional dark report when `--report` is enabled.

## results.json

The JSON output includes run metadata, processed image/frame entries, decoded QR text, filter status, match status, QR corner points, centers, sizes, source decoder, preprocessing variant, merged duplicate sources, and crop paths when enabled.

All detections are saved. Filters do not delete results; they mark each detection with `passes_filter` and `filter_reason`.

## Detection Pipeline

1. Accept an image, video, or folder.
2. Extract sampled frames for videos.
3. In `fast` mode, run only cheap full-frame ZXing attempts.
4. In `robust` mode, use a candidate-first pipeline:
   - run a few cheap full-frame decode attempts
   - ask OpenCV to locate QR-like quadrilaterals even if decode fails
   - add contour-based quadrilateral candidates
   - crop each candidate with padding so the quiet border is preserved
   - perspective-rectify candidate crops when 4 corners are available
   - run ZXing first on enhanced crop variants
   - use OpenCV decode fallback on crop variants
   - use a capped fallback tile scan only if candidate decoding finds nothing
5. Merge duplicate detections by normalized text, center distance, and IoU.
6. Normalize/filter decoded text.
7. Save annotations, JSON, crops, frames, candidate metadata, and optional report.

This avoids the old brute-force behavior where robust mode created thousands of whole-frame/tile variants for every 4K video frame.
## Supported Inputs

Images:

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.webp`
- `.tiff`

Videos:

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.webm`

## Limitations

- Extremely tiny QR codes may be impossible to decode.
- Motion blur can destroy QR modules.
- Glare or reflection can break decoding.
- Missing quiet border around the QR reduces reliability.
- Low camera resolution limits recoverability.
- Extreme angles can make decoding fail even after preprocessing.

## Tips

- Use high-resolution input.
- Avoid blur and camera shake.
- Keep the QR quiet border visible.
- Print larger QR codes when possible.
- Use good, even lighting.
- Avoid extreme viewing angles.

