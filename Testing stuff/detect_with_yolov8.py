"""
Simple YOLOv8 image detector

Usage:
    python detect_with_yolov8.py --image path/to/image.jpg --output output.jpg

This script expects `yolov8s.pt` to be in the same folder ("Testing stuff").
It uses the `ultralytics` package (YOLOv8) and OpenCV to load/save images.
"""
import argparse
import os
import sys

# ...existing code...

def parse_args():
    p = argparse.ArgumentParser(description="Run yolov8s detection on a single image")
    p.add_argument("--image", "-i", required=True, help="Path to input image")
    p.add_argument("--model", "-m", default=os.path.join(os.path.dirname(__file__), "yolov8s.pt"), help="Path to yolov8 model file")
    p.add_argument("--output", "-o", default="output.jpg", help="Path to save annotated image")
    p.add_argument("--conf", default=0.25, type=float, help="Confidence threshold")
    return p.parse_args()


def main():
    args = parse_args()

    # lazy imports with helpful message
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("ERROR: could not import ultralytics. Install with: pip install ultralytics")
        raise

    try:
        import cv2
    except Exception:
        print("ERROR: could not import opencv-python. Install with: pip install opencv-python")
        raise

    if not os.path.isfile(args.image):
        print(f"Input image not found: {args.image}")
        sys.exit(2)

    if not os.path.isfile(args.model):
        print(f"Model file not found: {args.model}")
        sys.exit(2)

    print(f"Loading model from: {args.model}")
    model = YOLO(args.model)

    print(f"Running inference on: {args.image}")
    results = model.predict(source=args.image, conf=args.conf, save=False)

    # results is a list (one per image) - we used a single image
    r = results[0]

    # print detections to console
    print("Detections:")
    if r.boxes is None or len(r.boxes) == 0:
        print("  No detections")
    else:
        # each box has xyxy, conf, cls
        for i, box in enumerate(r.boxes):
            xyxy = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else box.xyxy.tolist()
            conf = float(box.conf[0]) if hasattr(box, 'conf') else float(box.conf)
            cls = int(box.cls[0]) if hasattr(box, 'cls') else int(box.cls)
            name = r.names[cls] if hasattr(r, 'names') and cls in r.names else str(cls)
            print(f"  {i}: {name} (class {cls}) conf={conf:.3f} box={xyxy}")

    # save annotated image using ultralytics rendering helper
    annotated_path = args.output
    print(f"Saving annotated image to: {annotated_path}")
    # ultralytics provides .plot() or .save() methods depending on version
    try:
        # try to use ultralytics built-in save
        model.predict(source=args.image, conf=args.conf, save=True, imgsz=640)
        # the ultralytics save will create a runs/detect/exp... dir; copy the saved image if present
        # attempt to find most recent runs/detect/exp*/{basename}
        import glob
        import shutil
        base = os.path.basename(args.image)
        candidates = sorted(glob.glob(os.path.join(os.getcwd(), 'runs', 'detect', 'exp*', base)), key=os.path.getmtime, reverse=True)
        if candidates:
            shutil.copy(candidates[0], annotated_path)
        else:
            print("Could not find saved annotated image in runs/detect/exp*/; falling back to manual drawing")
            raise RuntimeError("no saved image found")
    except Exception:
        # fallback: load image and draw boxes using cv2
        img = cv2.imread(args.image)
        if img is None:
            print("Failed to read image for annotation fallback")
            sys.exit(1)
        h, w = img.shape[:2]
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else box.xyxy.tolist()
                x1, y1, x2, y2 = map(int, xyxy[:4])
                conf = float(box.conf[0]) if hasattr(box, 'conf') else float(box.conf)
                cls = int(box.cls[0]) if hasattr(box, 'cls') else int(box.cls)
                name = r.names[cls] if hasattr(r, 'names') and cls in r.names else str(cls)
                label = f"{name} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(annotated_path, img)

    print("Done.")

if __name__ == '__main__':
    main()
