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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def parse_args():
    p = argparse.ArgumentParser(description="Run yolov8s detection on a single image")
    p.add_argument("--image", "-i", required=True, help="Path to input image")
    p.add_argument("--model", "-m", default=os.path.join(os.path.dirname(__file__), "yolov8s.pt"), help="Path to yolov8 model file")
    p.add_argument("--output", "-o", default="output.jpg", help="Path to save annotated image")
    p.add_argument("--conf", default=0.25, type=float, help="Confidence threshold")
    p.add_argument("--diagnose", action="store_true", help="Print model class names and a check for 'tree' then exit")
    p.add_argument("--clip", action="store_true", help="Run a CLIP zero-shot image-level label check (no boxes)")
    p.add_argument("--clip-labels", default=",".join(["tree","door","car","phone","laptop","computer","person","bicycle","bench","sign","dog","cat","chair","table","cup","bottle"]), help="Comma-separated labels to test with CLIP (used with --clip)")
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

    # Diagnostic mode: print available class names and exit
    if args.diagnose:
        # model.names might be a dict or list depending on ultralytics version
        names = None
        try:
            names = getattr(model, 'names', None)
        except Exception:
            names = None

        if names is None:
            # try to run a dry prediction to extract names from a Results object
            try:
                tmp = model.predict(source=args.image, conf=args.conf, max_det=1)
                rtmp = tmp[0]
                names = getattr(rtmp, 'names', None)
            except Exception:
                names = None

        if names is None:
            print('Could not determine model class names programmatically.')
        else:
            # normalize into a list
            if isinstance(names, dict):
                items = [v for k, v in sorted(names.items())]
            else:
                items = list(names)
            print('Model class names:')
            for i, n in enumerate(items):
                print(f'  {i}: {n}')

            # check for 'tree' or related names
            lower = [n.lower() for n in items]
            suggestions = [s for s in ['tree', 'plant', 'potted plant', 'vegetation', 'bush', 'flower'] if any(s in x for x in lower)]
            if any('tree' == x for x in lower) or 'tree' in ' '.join(lower):
                print('\nThis model appears to include a "tree" class.')
            else:
                print('\nThis model does NOT include a "tree" class. Closest matches (if any):', suggestions or 'none')
                print('Reason it may not detect trees: the model was likely trained on COCO which does not include "tree" as a detection class.')
                print('Options: train/fine-tune a model with tree labels, use an open-vocabulary detector (Grounding DINO + SAM/CLIP), or use a specialized tree/tree-crown model (e.g., DeepForest).')
        return

    # CLIP zero-shot image-level check (no bounding boxes) -------------------------------------------------
    if args.clip:
        labels = [s.strip() for s in args.clip_labels.split(',') if s.strip()]
        print(f"Running CLIP zero-shot check for labels: {labels}")
        try:
            import torch
            from PIL import Image
            from transformers import CLIPProcessor, CLIPModel
        except Exception as e:
            print("ERROR: CLIP dependencies not available. Install with: pip install transformers torch")
            print("(For CUDA-enabled torch builds see https://pytorch.org/get-started/locally/)")
            raise

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        try:
            processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
            clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
        except Exception as e:
            print('Failed to load CLIP model:', e)
            raise

        # load image
        try:
            image = Image.open(args.image).convert('RGB')
        except Exception as e:
            print('Failed to open image for CLIP check:', e)
            raise

        # prepare text and image inputs
        text_inputs = processor(text=labels, return_tensors='pt', padding=True)
        image_inputs = processor(images=image, return_tensors='pt')
        # move to device
        for k, v in image_inputs.items():
            image_inputs[k] = v.to(device)
        for k, v in text_inputs.items():
            text_inputs[k] = v.to(device)

        with torch.no_grad():
            image_emb = clip_model.get_image_features(**image_inputs)
            text_emb = clip_model.get_text_features(**text_inputs)

            # normalize
            image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)

            # scores (1 x N)
            scores = (100.0 * image_emb @ text_emb.T).softmax(dim=-1)
            scores = scores[0].cpu().tolist()

        ranked = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
        print('\nCLIP zero-shot top matches:')
        for label, score in ranked[:min(10, len(ranked))]:
            print(f'  {label}: {score:.4f}')

        return

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
