# QR-Detector

An offline Python pipeline for detecting and decoding QR codes and related barcode formats in images, videos, and folders. It uses OpenCV for image/video handling and annotation, ZXing-C++ as the primary decoder, a smart candidate-first QR pipeline, and a ranked whole-frame recovery pass for hard video frames.

Live webcam mode is not included. Offline processing gives the detector time to locate likely QR quadrilaterals, crop and rectify them, retry promising regions, rank video frames, run borrowed whole-frame enhancement recovery including CLAHE, thresholding, sharpening, rotations, scale-ups, and Grayscale+CLAHE+Sharpen, and write structured results.

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

Video in default smart mode:

```powershell
python main.py --input inputs/videos/test.mp4 --mode smart --profile
```

The old robust command still works because `robust` is now an alias for `smart`:

```powershell
python main.py --input inputs/videos/test.mp4 --mode robust --profile
```

Folder:

```powershell
python main.py --input inputs/
```

Video frame sampling:

```powershell
python main.py --input inputs/videos/test.mp4 --frame-step 5 --max-frames 200
```

Disable temporal rescue if you only want the first-pass result:

```powershell
python main.py --input inputs/videos/test.mp4 --mode smart --no-temporal-rescue
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

Generate the dark HTML report with raw and selected candidate outlines:

```powershell
python main.py --input inputs/images/test.png --report
```


Hard video recovery, including borrowed frame-ranking plus the full enhancement/rotation/scale recovery pass:

```powershell
python main.py --input inputs/videos/QR-DET5.mp4 --mode smart --frame-step 2 --recovery-top-frames 10 --recovery-max-variants 0 --recovery-max-side 1600 --profile
```

For `QR-DET5.mp4`, this recovery pass decodes frame `94` as `DataBar Stacked` text `(01)96908453915728`. That file is not decoded as a QR code; ZXing identifies it as a stacked barcode format.
## CLI Options

- `--input`: required image, video, or folder.
- `--mode`: `fast`, `smart`, or `robust`. Default is `smart`; `robust` is kept as an alias.
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
- `--debug`: print detailed progress and save debug variants.
- `--report`: generate `report.html`.
- `--profile`: print detector timing and failure reason per processed item.
- `--max-variants-per-frame`: cap smart decode attempts per frame.
- `--max-candidates`: cap QR-like candidate regions per frame.
- `--no-temporal-rescue`: disable the bounded video near-miss retry pass.
- `--rescue-window`: nearby frame window for temporal rescue. Default is `2`.
- `--max-rescue-targets`: maximum candidate tracks rescued per video. Default is `8`.
- `--no-frame-recovery`: disable ranked whole-frame enhancement recovery.
- `--recovery-top-frames`: maximum ranked frames to try with whole-frame recovery. Default is `10`.
- `--recovery-max-variants`: maximum whole-frame recovery variants per ranked frame. Default is `0`, which means try all recovery variants.
- `--recovery-max-side`: resize ranked recovery frames so the longest side is at most this many pixels. Default is `1600`; use `0` for original full resolution.

## Output Folder

By default, results are written under `outputs/`:

- `outputs/annotated/`: annotated image/frame results.
- `outputs/crops/`: QR crops when `--save-crops` is enabled.
- `outputs/frames/`: processed video frames when `--save-frames` is enabled.
- `outputs/debug/`: debug variants when `--debug` is enabled.
- `outputs/results.json`: structured machine-readable results.
- `outputs/report.html`: optional dark report when `--report` is enabled.

## results.json

The JSON output includes run metadata, decoded QR text, filter status, match status, QR corner points, centers, sizes, source decoder, preprocessing variant, merged duplicate sources, selected candidate metadata, raw candidate diagnostics, near-miss metadata, failure reasons, and crop paths when enabled.

Useful failure reasons:

- `decoded`: a QR was decoded.
- `no_candidate`: no QR-like region was found.
- `candidate_no_decode`: one or more QR-like regions were found, but decoding failed.
- `checksum_near_miss`: ZXing saw QR/barcode structure but failed checksum validation, so the symbol is close to readable but damaged, blurred, too small, or distorted.

All detections are saved. Filters do not delete results; they mark each detection with `passes_filter` and `filter_reason`.
Candidate outline legend in annotated report images:

- Faint cyan boxes: raw OpenCV/contour candidates found before merge/capping.
- Orange boxes: selected merged candidates attempted for decoding.
- Green/yellow/red boxes: decoded results depending on filter/match status.

## Detection Pipeline

1. Accept an image, video, or folder.
2. Extract sampled frames for videos.
3. In `fast` mode, run only cheap full-frame ZXing attempts.
4. In `smart` mode:
   - run cheap full-frame decode attempts
   - create downscaled locator variants: original, grayscale, CLAHE, adaptive threshold
   - ask OpenCV to locate QR-like quadrilaterals on locator variants
   - add full-resolution locator variants for large frames so tiny QR candidates are not erased by downscaling
   - add contour-based quadrilateral candidates with lower tiny-candidate thresholds and shape/size scoring
   - crop each candidate with padding so the quiet border is preserved
   - perspective-rectify candidate crops when 4 corners are available
   - create QR-focused crop cleanup variants that remove border-connected dark background, force a white quiet zone, and try inner-square QR crops
   - run ZXing on crop variants with multiple binarizers
   - use OpenCV decode as a crop fallback
   - use coverage-prioritized fallback tiles if no candidate decodes
   - for videos, retry only the best nearby near-miss/candidate tracks
   - if no detection succeeds, rank sampled frames and try whole-frame recovery variants: histogram equalization, CLAHE, thresholds, denoise/deblur/sharpen, rotations, scale-ups, and combo passes such as `grayscale_clahe_sharpen`
5. Merge duplicate detections by normalized text, center distance, and IoU.
6. Normalize/filter decoded text.
7. Save annotations, JSON, crops, frames, candidate metadata, near-miss metadata, and optional report.

This avoids the old brute-force behavior where robust mode created thousands of whole-frame/tile variants for every 4K video frame. Whole-frame recovery only runs on the best ranked frames after the candidate pipeline fails, and it downscales very large frames by default to keep hard videos responsive.

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

- Extremely tiny QR codes may be possible to outline as candidates but still impossible to decode if there are too few pixels.
- Motion blur can destroy QR modules.
- Glare or reflection can break decoding.
- Missing quiet border around the QR reduces reliability; the pipeline now adds synthetic quiet-zone variants, but it cannot recover modules that are blurred or too small.
- Low camera resolution limits recoverability.
- Extreme angles can make decoding fail even after preprocessing.
- `checksum_near_miss` means the detector saw QR/barcode-like structure, but the data was not reliable enough to return as decoded text.
- Some inputs may be non-QR barcodes. `QR-DET5.mp4` is recovered as `DataBar Stacked`, not QR.

## Tips

- Use high-resolution input.
- Avoid blur and camera shake.
- Keep the QR quiet border visible.
- Print larger QR codes when possible.
- Use good, even lighting.
- Avoid extreme viewing angles.
- For hard videos, try `--frame-step 1 --profile` so the pipeline sees every frame.
