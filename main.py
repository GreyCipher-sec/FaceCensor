#!/usr/bin/env python3

import argparse
import os
import sys
import urllib.request
import numpy as np
import cv2

# ──────────────────────────────────────────────
# DNN model URLs & local paths
# ──────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
CAFFEMODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/"
    "dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)
PROTOTXT_PATH = os.path.join(MODELS_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")


# ──────────────────────────────────────────────
# Model downloader
# ──────────────────────────────────────────────
def download_models():
    """Download DNN face-detector model files into ./models/"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    files = [(PROTOTXT_URL, PROTOTXT_PATH), (CAFFEMODEL_URL, CAFFEMODEL_PATH)]
    for url, dest in files:
        if os.path.exists(dest):
            print(f"  Already exists: {dest}")
            continue
        print(f"  Downloading {os.path.basename(dest)} …")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  Saved → {dest}")
        except Exception as e:
            print(f"  ERROR downloading {url}: {e}")
            sys.exit(1)
    print("Done. You can now run face_censor.py without --download-models.")


# ──────────────────────────────────────────────
# Detector loader
# ──────────────────────────────────────────────
def load_detector():
    """Return (detector_type, detector_object)."""
    if os.path.exists(PROTOTXT_PATH) and os.path.exists(CAFFEMODEL_PATH):
        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
        print("[INFO] Using OpenCV DNN face detector (ResNet SSD)")
        return "dnn", net
    else:
        print("[INFO] DNN model not found — falling back to Haar Cascade.")
        print("       Run with --download-models for better accuracy.")
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        return "haar", cascade


# ──────────────────────────────────────────────
# Face detection
# ──────────────────────────────────────────────
def detect_faces_dnn(net, frame, confidence_threshold=0.5):
    """Return list of (x, y, w, h) using DNN detector."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces


def detect_faces_haar(cascade, frame):
    """Return list of (x, y, w, h) using Haar Cascade."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return list(faces) if len(faces) > 0 else []


def detect_faces(detector_type, detector, frame):
    if detector_type == "dnn":
        return detect_faces_dnn(detector, frame)
    return detect_faces_haar(detector, frame)


# ──────────────────────────────────────────────
# Censor effects
# ──────────────────────────────────────────────
def apply_blur(frame, x, y, w, h, strength=99):
    """Gaussian blur on face region."""
    ksize = strength if strength % 2 == 1 else strength + 1   # must be odd
    roi = frame[y:y+h, x:x+w]
    frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (ksize, ksize), 30)
    return frame


def apply_pixelate(frame, x, y, w, h, blocks=12):
    """Pixelate face region by downscaling then upscaling."""
    roi = frame[y:y+h, x:x+w]
    small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y:y+h, x:x+w] = pixelated
    return frame


def apply_blackbox(frame, x, y, w, h):
    """Solid black rectangle over face."""
    frame[y:y+h, x:x+w] = 0
    return frame


def censor_face(frame, x, y, w, h, effect):
    """Apply chosen censor effect with a small padding margin."""
    pad = int(min(w, h) * 0.1)
    fx = max(0, x - pad)
    fy = max(0, y - pad)
    fw = min(frame.shape[1] - fx, w + 2 * pad)
    fh = min(frame.shape[0] - fy, h + 2 * pad)

    if effect == "blur":
        return apply_blur(frame, fx, fy, fw, fh)
    elif effect == "pixelate":
        return apply_pixelate(frame, fx, fy, fw, fh)
    elif effect == "blackbox":
        return apply_blackbox(frame, fx, fy, fw, fh)
    return frame


# ──────────────────────────────────────────────
# Overlay HUD (webcam / video preview)
# ──────────────────────────────────────────────
EFFECT_KEYS = {"b": "blur", "p": "pixelate", "k": "blackbox"}

def draw_hud(frame, effect, face_count):
    h, w = frame.shape[:2]
    label = f"Effect: {effect.upper()}  |  Faces: {face_count}  |  [B]lur [P]ixelate [K]blackbox [Q]uit"
    cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, label, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ──────────────────────────────────────────────
# Processing modes
# ──────────────────────────────────────────────
def process_image(input_path, output_path, effect, detector_type, detector):
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {input_path}")
        sys.exit(1)

    faces = detect_faces(detector_type, detector, frame)
    for (x, y, w, h) in faces:
        frame = censor_face(frame, x, y, w, h, effect)

    cv2.imwrite(output_path, frame)
    print(f"[OK] {len(faces)} face(s) censored → {output_path}")


def process_video(input_path, output_path, effect, detector_type, detector):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    print(f"[INFO] Processing {total} frames at {fps:.1f} fps …")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces(detector_type, detector, frame)
        for (x, y, w, h) in faces:
            frame = censor_face(frame, x, y, w, h, effect)
        out.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"  {frame_idx}/{total} ({pct:.0f}%)")

    cap.release()
    out.release()
    print(f"[OK] Done → {output_path}")


def process_webcam(effect, detector_type, detector):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(5):
        cap.read()

    current_effect = [effect]

    fig, ax = plt.subplots()
    ax.axis("off")
    ret, first_frame = cap.read()
    im = ax.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))

    def on_key(event):
        if event.key == "q":
            cap.release()
            plt.close()
        elif event.key in EFFECT_KEYS:
            current_effect[0] = EFFECT_KEYS[event.key]
            print(f"[INFO] Effect changed to: {current_effect[0]}")

    def update(_):
        grabbed, frame = cap.read()
        if not grabbed:
            return [im]
        faces = detect_faces(detector_type, detector, frame)
        for (x, y, w, h) in faces:
            frame = censor_face(frame, x, y, w, h, current_effect[0])
        im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        im.axes.figure.canvas.draw_idle()
        return [im]

    fig.canvas.mpl_connect("key_press_event", on_key)
    ani = animation.FuncAnimation(fig, update, interval=30, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    cap.release()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Censor faces in images, videos, or webcam.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--download-models", action="store_true",
                        help="Download DNN model files and exit.")
    parser.add_argument("--input", "-i", help="Input image or video file path.")
    parser.add_argument("--output", "-o", help="Output file path (image or video).")
    parser.add_argument("--webcam", action="store_true",
                        help="Use live webcam instead of a file.")
    parser.add_argument(
        "--effect", "-e",
        choices=["blur", "pixelate", "blackbox"],
        default="blur",
        help="Censor effect to apply (default: blur).",
    )
    parser.add_argument(
        "--confidence", "-c", type=float, default=0.5,
        help="DNN confidence threshold 0–1 (default: 0.5).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.download_models:
        download_models()
        return

    detector_type, detector = load_detector()

    if args.webcam:
        process_webcam(args.effect, detector_type, detector)

    elif args.input:
        if not os.path.exists(args.input):
            print(f"[ERROR] Input file not found: {args.input}")
            sys.exit(1)

        # Auto-detect image vs video
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        ext = os.path.splitext(args.input)[1].lower()

        output = args.output or (
            os.path.splitext(args.input)[0] + "_censored" + ext
        )

        if ext in image_exts:
            process_image(args.input, output, args.effect, detector_type, detector)
        else:
            process_video(args.input, output, args.effect, detector_type, detector)

    else:
        print("[ERROR] Provide --input <file> or --webcam (or --download-models).")
        sys.exit(1)


if __name__ == "__main__":
    main()
