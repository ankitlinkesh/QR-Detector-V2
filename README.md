# Live-QR-target-matcher

A Python OpenCV + ZXing-C++ project for detecting multiple QR codes, matching them against a target QR, and calculating target-center alignment for autonomous drone delivery workflows.

This project uses a laptop webcam to detect QR codes, save one QR code as a target, and then find the matching QR code among multiple visible QR codes.

It is meant as a base project for AeroTHON-style drone QR detection. It does not include drone control, MAVLink, Flask, databases, or web apps.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

## How to Run

```powershell
python main.py
```

The app opens your default laptop webcam at camera index `0`.

## Controls

- `s`: Save the first visible QR code with non-empty decoded data as the target.
- `r`: Reset the saved target.
- `d`: Save the current frame to `work/debug_frames` and print ZXing/OpenCV detection counts.
- `q`: Quit the app.

## Project Flow

1. Start the app.
2. Hold one QR code in front of the camera.
3. The app draws a blue box and shows `QR Found`.
4. Press `s` to save that QR code's decoded text as the target.
5. Show a camera view with multiple QR codes.
6. The app compares every decoded QR code with the saved target.
7. The matching QR gets a green box.
8. Non-matching QR codes get red boxes.
9. For the matching QR, the app draws its center point and displays:
   - `error_x`: QR center minus camera frame center on the x-axis.
   - `error_y`: QR center minus camera frame center on the y-axis.

These error values are useful later for target-centering logic in a drone project.

## File Overview

- `main.py`: Runs the webcam loop, keyboard controls, drawing, and terminal logs.
- `qr_detector.py`: Detects and decodes QR codes with ZXing, with OpenCV as a fallback.
- `qr_matcher.py`: Saves the target, compares QR text, and calculates center errors.
- `requirements.txt`: Lists the Python dependency.
- `README.md`: Explains setup and usage.

## Limitations

QR detection depends heavily on camera quality and the environment. Results may be poor when:

- The QR code is blurry.
- Lighting is too dark, too bright, or uneven.
- The QR code is too far from the camera.
- The QR code is too small in the frame.
- The QR code is viewed from a sharp angle.
- The camera is moving quickly.
- Multiple QR codes overlap or are partially hidden.

For best results, use clear printed QR codes, steady camera movement, good lighting, and enough distance for the entire QR code to fit in the frame.

The app uses ZXing as the main QR decoder because OpenCV's built-in QR detector can miss codes when many are visible at once. If all QR codes still are not detected, press `d` to save a debug frame and print counts from both ZXing and OpenCV. That helps separate software detection limits from camera/printing issues.
