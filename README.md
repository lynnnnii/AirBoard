# AirBoard

**Pinch-Gesture Air Writing on PDF Slides**  
Introduction to Computer Vision · Team **Maroon 5**

AirBoard lets presenters annotate PDF slides using only a webcam. Pinching the thumb and index finger writes on the slide; releasing the pinch lifts the pen. OpenCV provides the display, MediaPipe Hand Landmarker tracks hand landmarks, and the project implements gesture recognition, calibration, smoothing, and PDF compositing.

## Features

- **Air writing:** draw on slides using a thumb–index pinch.
- **Pen and highlighter:** switch tools with a double-clap gesture and choose colors or line widths from the on-screen palette.
- **Editing:** erase complete strokes with a fist; use sideways thumb gestures to undo and redo.
- **Laser pointer:** point with the index finger without leaving ink.
- **Custom gesture mapping:** change gesture-to-action assignments in the settings screen.
- **Coordinate calibration:** map a writing region in camera space to slide coordinates with a four-point homography.
- **Smooth handwriting:** One Euro filtering and stroke interpolation, with distinct handling for highlighter strokes.
- **Session persistence:** automatically save annotations and the current page locally and restore them when reopening the same PDF. The original PDF remains unchanged.
- **Alternative inputs:** use a recorded video, mouse, or virtual input for debugging and demonstrations.

## Requirements

- Python **3.10–3.12** (MediaPipe 0.10.21 does not support Python 3.13 or later).
- A webcam for live gesture recognition; alternatively, use mouse, virtual, or video input.
- Windows or macOS. CPU execution is supported.
- The bundled `models/hand_landmarker.task` model and sample PDFs under `data/pdfs/ui-test/`.

## Installation

### Windows (PowerShell)

Run these commands from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.\.venv\Scripts\python.exe -m airboard.ui.demo --pdf data/pdfs/ui-test/segmentation-p01-15.pdf
```

If your Python launcher does not have Python 3.12, replace `py -3.12` with an installed Python 3.10–3.12 executable.

### macOS (Terminal)

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-ui.txt
.venv/bin/python -m airboard.ui.demo --pdf data/pdfs/ui-test/segmentation-p01-15.pdf
```

On macOS, grant camera access to Terminal or your IDE when prompted. If using an external or Continuity Camera, adjust the camera index.

## Running the demo

The supported entry point is **`python -m airboard.ui.demo`**; `main.py` is a placeholder.

```powershell
# Live webcam with the bundled sample PDF
.\.venv\Scripts\python.exe -m airboard.ui.demo --pdf data/pdfs/ui-test/segmentation-p01-15.pdf

# Webcam without a PDF: standalone white-slide preview
.\.venv\Scripts\python.exe -m airboard.ui.demo

# Alternative input modes (append additional flags as needed)
.\.venv\Scripts\python.exe -m airboard.ui.demo --mouse --pdf data/pdfs/ui-test/segmentation-p01-15.pdf
.\.venv\Scripts\python.exe -m airboard.ui.demo --virtual
.\.venv\Scripts\python.exe -m airboard.ui.demo --video path/to/recording.mp4 --pdf data/pdfs/ui-test/segmentation-p01-15.pdf

# Full-screen display
.\.venv\Scripts\python.exe -m airboard.ui.demo --fullscreen --pdf data/pdfs/ui-test/segmentation-p01-15.pdf
```

By default the application uses a webcam and opens an approximately 1800 × 1000 OpenCV window, with the camera preview on the left and slide/annotations on the right.

## Gesture controls

| Gesture | Default action |
| --- | --- |
| Thumb + index pinch | Draw / lift pen |
| Double clap | Toggle pen and highlighter |
| Fist (all fingers folded) | Stroke eraser |
| Index finger only | Laser pointer |
| Two extended fingers | Stop writing |
| Thumb pointing left | Undo |
| Thumb pointing right | Redo |

Swipe-to-turn-page gestures are intentionally **disabled** to prevent accidental navigation. Use the keyboard arrow keys to move between PDF pages.

### Keyboard and interface

| Key | Action |
| --- | --- |
| `C` | Calibrate writing region |
| `R` | Reset to the default writing region (normal view) |
| `S` | Open gesture settings |
| `X` | Clear annotations on the current page |
| Left / right arrow | Change PDF page |
| `Q` or `Esc` | Quit (outside the settings menu) |

During calibration, select the corners in this order: **top-left → top-right → bottom-right → bottom-left**. Hover over the on-screen palette to change pen color or thickness. Settings are stored separately from the source PDF.

## Project structure

```text
airboard/
  calibration/     Four-point homography and writing-region settings
  gestures/        Gesture detection and recognition rules
  input/           Camera capture utilities
  render/          PDF rendering, ink composition, erasing, undo/redo, sessions
  smoothing/       Filters, stroke smoothing, and interpolation
  tracking/        MediaPipe-based hand landmark tracking
  ui/              OpenCV UI, event controller, palette, and demo entry point
config/            Default parameters and gesture-action mappings
data/pdfs/ui-test/ Sample PDF files for demonstrations
models/            Bundled MediaPipe hand landmarker model
tests/             Unit and regression tests
tools/             Rendering setup utility
requirements-ui.txt
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The repository includes regression tests for gesture handling, coordinate mapping, smoothing, PDF rendering, UI behavior, and session persistence. Test counts and results depend on the checked-out revision and local environment.

## Technical approach

Each frame follows the pipeline **camera capture → landmark detection → gesture interpretation → homography → smoothing → PDF compositing → OpenCV display**. The pretrained MediaPipe model detects hand landmarks only. The gesture rules, pen state, coordinate mapping, smoothing, and layer composition are implemented by the team.

## Documentation

- [Coordinate mapping and OpenCV UI](COORDINATE_UI.md)
- [Integration changes and compatibility notes](INTEGRATION_CHANGES.md)
- [Bundled PDF test data](data/pdfs/ui-test/README.md)

## Limitations and future work

Gesture accuracy depends on lighting, camera positioning, hand posture, and occlusion. Double-clap recognition can be inconsistent; long handwriting sessions may be tiring. Planned improvements include shape correction, longer-distance hand detection, and easier note sharing. The current UI does not provide a complete exported-annotated-PDF workflow.

## Team

**Maroon 5** — Hyunbo Kim (hand tracking and gestures), Subin Lee (smoothing), Yelynn Lee (PDF rendering and notes), Junseo Lim (mapping and UI), Hangmin Jo (integration and presentation).
