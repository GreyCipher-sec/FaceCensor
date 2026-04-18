# face_censor.py

A command-line tool to automatically detect and censor faces in images, video files, and live webcam feeds using OpenCV. Supports three censor effects, Gaussian blur, pixelation, and solid black box, all switchable at runtime during webcam mode.

---

## Project Structure

```
face_censor.py       # Main script
models/              # Created automatically on --download-models
  deploy.prototxt
  res10_300x300_ssd_iter_140000.caffemodel
```

---

## Features

- **Three input modes**: static images, video files, and live webcam
- **Three censor effects**: blur, pixelate, blackbox (switchable on the fly in webcam mode)
- **Two face detectors**: OpenCV DNN (ResNet SSD, recommended) with automatic fallback to Haar Cascade
- **Smart output naming**: automatically appends `_censored` to the filename if no output path is given
- **Progress reporting** for video files (every 30 frames)
- **Live HUD overlay** in webcam mode showing current effect and detected face count
- **Padded bounding boxes**: each detected face region is slightly expanded before censoring to avoid edge clipping
- Zero external API calls, everything runs locally

---

## Requirements

- Python 3.7+
- opencv-contrib-python (not `opencv-python`, the contrib build is required for webcam display)
- numpy
- matplotlib (used for the webcam live preview window)

Install with:

```bash
pip install opencv-contrib-python numpy matplotlib
```

---

## Setup

### Step 1 - Download the DNN model (recommended, one-time)

```bash
python face_censor.py --download-models
```

This downloads two files (~10 MB total) into a `models/` subdirectory next to the script:

| File | Size | Purpose |
|---|---|---|
| `deploy.prototxt` | ~27 KB | Network architecture definition |
| `res10_300x300_ssd_iter_140000.caffemodel` | ~10 MB | Pre-trained ResNet SSD weights |

> **No internet?** Skip this step. The script automatically falls back to OpenCV's built-in Haar Cascade detector, which is always available but less accurate on angled or partially occluded faces.

---

## Usage

### Censor an image

```bash
python face_censor.py --input photo.jpg --effect blur
```

Output is saved as `photo_censored.jpg` in the same directory. To specify the output path explicitly:

```bash
python face_censor.py --input photo.jpg --output result.jpg --effect pixelate
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

---

### Censor a video file

```bash
python face_censor.py --input video.mp4 --effect blackbox
```

Output is saved as `video_censored.mp4`. The original FPS and resolution are preserved. Progress is printed every 30 frames:

```
[INFO] Processing 1800 frames at 30.0 fps …
  30/1800 (2%)
  60/1800 (3%)
  ...
[OK] Done → video_censored.mp4
```

Supported video formats: any format your OpenCV build supports (commonly `.mp4`, `.avi`, `.mov`, `.mkv`).

---

### Live webcam

```bash
python face_censor.py --webcam --effect blur
```

Opens your default camera (device index 0) and displays a live preview via matplotlib. Effects are switchable on the fly with keyboard shortcuts.

**Keyboard shortcuts in webcam mode:**

| Key | Action |
|---|---|
| `b` | Switch to Gaussian blur |
| `p` | Switch to pixelation |
| `k` | Switch to black box |
| `q` | Quit |

**Linux / Wayland / USB webcam notes:**

The webcam capture uses the V4L2 backend explicitly with MJPG format at 1280×720, which is required for reliable frame capture on most USB webcams under Linux. If your camera is not at `/dev/video0` or uses a different resolution, you may need to adjust the device index and resolution in `process_webcam`. You can check available devices and formats with:

```bash
v4l2-ctl --list-devices
v4l2-ctl --list-formats-ext
```

The preview window is rendered via matplotlib instead of `cv2.imshow` to avoid Qt font and Wayland compatibility issues that affect OpenCV's built-in display backend on some Linux systems.

---

## All CLI Options

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--input` | `-i` | path | — | Input image or video file |
| `--output` | `-o` | path | auto | Output file path |
| `--webcam` | — | flag | false | Use live webcam feed |
| `--effect` | `-e` | string | `blur` | Censor effect: `blur`, `pixelate`, `blackbox` |
| `--confidence` | `-c` | float | `0.5` | DNN detection confidence threshold (0–1) |
| `--download-models` | — | flag | false | Download DNN model files and exit |

---

## Censor Effects

### `blur` - Gaussian Blur
Applies a strong Gaussian blur (kernel size 99×99, sigma 30) to the face region. The face is still recognisably human-shaped but all identifying detail is smoothed away. Best for a natural, professional look.

### `pixelate` - Pixelation
Shrinks the face region down to a 12×12 grid and scales it back up with nearest-neighbour interpolation, producing a blocky mosaic. Classic news-broadcast style.

### `blackbox` - Solid Black Rectangle
Fills the face bounding box with solid black. Maximum coverage, no information leaks through. Useful when downstream processing (e.g. OCR, ML pipelines) needs a hard mask rather than a perceptual effect.

All effects include a 10% padding margin around the raw detection bounding box so that hairline and chin edges are covered.

---

## Face Detectors

The script picks the best available detector at startup and logs which one it is using.

### OpenCV DNN - ResNet SSD (preferred)
- Loaded when `models/deploy.prototxt` and `models/res10_300x300_ssd_iter_140000.caffemodel` are present
- Input frame is resized to 300×300, normalised, and passed through a single-shot multibox detector
- Returns detections with a confidence score; only those above `--confidence` (default 0.5) are kept
- Handles angled faces, partial occlusion, and varying lighting conditions much better than Haar

### Haar Cascade (fallback)
- Always available: uses `haarcascade_frontalface_default.xml` bundled with OpenCV
- Faster but more prone to false positives and misses on non-frontal faces
- Uses `scaleFactor=1.1`, `minNeighbors=5`, `minSize=(30, 30)`

To force use of the Haar fallback even if models are present, simply rename or remove the `models/` directory.

---

## Tuning Detection

**Too many false positives** (random objects being censored):

```bash
python face_censor.py --input video.mp4 --effect blur --confidence 0.7
```

Raising the threshold above 0.5 requires the detector to be more certain before flagging a face.

**Missing faces** (real faces not being detected):

```bash
python face_censor.py --input video.mp4 --effect blur --confidence 0.3
```

Lowering the threshold catches weaker detections at the cost of more false positives. Also ensure you have the DNN model downloaded, the Haar fallback is significantly less sensitive on angled faces.

---

## Limitations

- The DNN model was trained on frontal and slightly off-axis faces. Extreme profile angles (beyond ~45°) may not be detected reliably.
- Very small faces (below ~30×30 pixels in the source resolution) are generally not detected.
- The Haar fallback is front-face only and performs poorly in low light.
- Video output uses the `mp4v` codec. If your player has trouble, re-encode the output with FFmpeg: `ffmpeg -i video_censored.mp4 -c:v libx264 output_h264.mp4`
- Audio is not copied to the output video. To re-attach the original audio: `ffmpeg -i video_censored.mp4 -i original.mp4 -c copy -map 0:v:0 -map 1:a:0 final.mp4`
- Webcam mode uses matplotlib for display. It is slightly less smooth than a native OpenCV window but avoids Qt/Wayland compatibility issues on Linux.
- The webcam capture is hardcoded to 1280×720 MJPG via V4L2. If your camera does not support this resolution or format, edit the `cap.set()` calls in `process_webcam` to match your camera's supported formats (check with `v4l2-ctl --list-formats-ext`).

---

## Examples

```bash
# Pixelate all faces in a group photo
python face_censor.py -i group_photo.png -e pixelate

# Black-box faces in a dashcam video, high confidence only
python face_censor.py -i dashcam.mp4 -o dashcam_safe.mp4 -e blackbox -c 0.65

# Real-time webcam starting with pixelation (switch effects with B/P/K)
python face_censor.py --webcam -e pixelate

# Re-download models if they get corrupted
rm -rf models/
python face_censor.py --download-models
```

---

## License

This project uses the OpenCV DNN face detector model originally released by OpenCV under the Apache 2.0 licence. The script itself is provided as-is for personal and research use.
