import argparse
import time
import os
import sys
import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description='Webcam YOLOv8 detector with optional open-vocab (CLIP)')
    p.add_argument('-m', '--model', default=os.path.join(os.path.dirname(__file__), 'yolov8s.pt'), help='Path to YOLOv8 model')
    p.add_argument('--rate', type=float, default=0.25, help='Seconds between frames (default 0.25)')
    p.add_argument('--conf', type=float, default=0.25, help='YOLO confidence threshold')
    p.add_argument('--show', action='store_true', help='Show detection window')
    p.add_argument('--openvocab', action='store_true', help='Run CLIP open-vocab classification on each box')
    p.add_argument('--labels', default='', help='Comma-separated labels for open-vocab classification (if empty, use preset vocab)')
    p.add_argument('--vocab-preset', choices=['common','extended'], default='common', help='Preset vocabulary to use when --labels is empty')
    p.add_argument('--topk', type=int, default=3, help='Show top-k CLIP matches per box')
    p.add_argument('--clip-threshold', type=float, default=0.12, help='Similarity threshold to accept CLIP match (cosine, 0-1)')
    return p.parse_args()


def crop_box(frame, xyxy, pad=5):
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    return frame[y1:y2, x1:x2]


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    print(f"Loading YOLO model from {args.model}")
    ymodel = YOLO(args.model)

    # prepare CLIP if requested
    clip_processor = None
    clip_model = None
    text_embeds = None
    device = None
    labels = [l.strip() for l in args.labels.split(',') if l.strip()]
    # built-in vocab presets (small sets tuned for everyday objects)
    PRESET_COMMON = [
        'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant',
        'stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
        'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
        'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
        'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
        'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
        'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush','tree','door','window','sidewalk',
        'road','bench','lamp','mailbox','sign','bicycle helmet','backyard','flower','grass','shoe','sock','shirt','pants','hat'
    ]
    PRESET_EXTENDED = PRESET_COMMON + [
        'wallet','watch','ring','glasses','camera','printer','fan','heater','umbrella stand','stool','cabinet','drawer',
        'bottle opener','thermos','water bottle','plate','glass','teapot','mug','carrot','celery','lettuce','cucumber','suit',
        'stroller','wheelchair','purse','briefcase','charger','adapter','headphones','speaker','microphone','guitar','piano',
    ]

    if args.openvocab:
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            from PIL import Image
        except Exception:
            print('Open-vocab requested but required packages are missing. Install with: pip install torch transformers pillow')
            args.openvocab = False
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f'Loading CLIP on device {device}...')
            clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
            clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)

            # choose label list
            if not labels:
                labels = PRESET_COMMON if args.vocab_preset == 'common' else PRESET_EXTENDED
            print(f'Using {len(labels)} open-vocab labels')

            # compute text embeddings in batches to avoid OOM
            batch_size = 128
            text_embeds_list = []
            for i in range(0, len(labels), batch_size):
                batch = labels[i:i+batch_size]
                text_inputs = clip_processor(text=batch, return_tensors='pt', padding=True)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                with torch.no_grad():
                    t_emb = clip_model.get_text_features(**text_inputs)
                    t_emb = t_emb / t_emb.norm(p=2, dim=-1, keepdim=True)
                    text_embeds_list.append(t_emb)
            text_embeds = torch.cat(text_embeds_list, dim=0)

    # open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Failed to open webcam')
        sys.exit(1)

    last_time = 0.0
    try:
        while True:
            now = time.time()
            if now - last_time < args.rate:
                time.sleep(max(0.01, args.rate - (now - last_time)))
            last_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print('Failed to read frame from webcam')
                break

            # run YOLO on the frame
            results = ymodel.predict(source=frame, conf=args.conf, imgsz=640)
            r = results[0]

            # draw detections and optionally run CLIP on each crop
            if r.boxes is not None:
                for box in r.boxes:
                    try:
                        xyxy = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else box.xyxy.tolist()
                        x1, y1, x2, y2 = map(int, xyxy[:4])
                        conf = float(box.conf[0]) if hasattr(box, 'conf') else float(box.conf)
                        cls = int(box.cls[0]) if hasattr(box, 'cls') else int(box.cls)
                        name = ymodel.names[cls] if hasattr(ymodel, 'names') and cls in ymodel.names else str(cls)
                        label = f"{name} {conf:.2f}"

                        # crop and run CLIP if requested
                        extra = ''
                        if args.openvocab and clip_model is not None:
                            crop = crop_box(frame, (x1, y1, x2, y2))
                            if crop.size != 0:
                                # prepare image for CLIP
                                img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                image_inputs = clip_processor(images=img, return_tensors='pt')
                                image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                                with torch.no_grad():
                                    img_feat = clip_model.get_image_features(**image_inputs)
                                    img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)
                                    sims = (img_feat @ text_embeds.T)[0]
                                    best_idx = int(sims.argmax().cpu().numpy())
                                    best_label = labels[best_idx]
                                    score = float(sims[best_idx].cpu().numpy())
                                    extra = f'|{best_label} {score:.2f}'

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label + extra, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception:
                        continue

            if args.show:
                cv2.imshow('Webcam - YOLOv8', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print('Interrupted by user')
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()