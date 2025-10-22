#!/usr/bin/env python3
"""
Simple YOLO detection without CLIP dependencies
"""

import argparse
import os
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Simple YOLO detection")
    p.add_argument("--image", "-i", required=True, help="Path to input image")
    p.add_argument("--model", "-m", default="models/yolov8s.pt", help="Path to YOLO model")
    p.add_argument("--output", "-o", default="output.jpg", help="Path to save annotated image")
    p.add_argument("--conf", default=0.25, type=float, help="Confidence threshold")
    return p.parse_args()

def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("ERROR: Could not import ultralytics. Install with: pip install ultralytics")
        return 1

    if not os.path.isfile(args.image):
        print(f"Input image not found: {args.image}")
        return 1

    if not os.path.isfile(args.model):
        print(f"Model file not found: {args.model}")
        return 1

    print(f"Loading model from: {args.model}")
    model = YOLO(args.model)

    print(f"Running inference on: {args.image}")
    results = model.predict(source=args.image, conf=args.conf, save=True, imgsz=640)

    # Print detections
    r = results[0]
    print("Detections:")
    if r.boxes is None or len(r.boxes) == 0:
        print("  No detections found")
    else:
        for i, box in enumerate(r.boxes):
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = r.names[cls]
            print(f"  {i}: {name} (class {cls}) conf={conf:.3f} box={xyxy}")

    print(f"Results saved to: runs/detect/")
    print("Done.")

    return 0

if __name__ == '__main__':
    sys.exit(main())
