"""
Detic webcam open-vocabulary detector (prototype)

Quick notes:
- Detic depends on Detectron2 and the Detic repo. Installing Detectron2 on Windows can be tricky.
- Typical install steps (Linux/NVIDIA GPU):
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
    pip install 'git+https://github.com/facebookresearch/detectron2.git'
    git clone https://github.com/IDEA-Research/Detic.git
    cd Detic
    pip install -r requirements.txt
    python setup.py build develop
  Then download a Detic config and weights (see Detic README).

Usage example:
    python detic_webcam.py --config /path/to/config.yaml --weights /path/to/model_weights.pth --labels "tree,door,phone,car" --show

This script will attempt to run Detic's DefaultPredictor and pass the text labels as open-vocab queries.
It prints detections and draws boxes on the webcam.
"""

import argparse
import time
import os
import sys
import cv2


def parse_args():
    p = argparse.ArgumentParser(description='Detic webcam open-vocab demo')
    p.add_argument('--config', required=False, help='Path to Detic config YAML (optional)')
    p.add_argument('--weights', required=False, help='Path to Detic model weights (.pth)')
    p.add_argument('--labels', default='tree,door,car,phone,laptop,person', help='Comma-separated text queries for Detic')
    p.add_argument('--show', action='store_true', help='Show annotated webcam window')
    p.add_argument('--rate', type=float, default=0.25, help='Seconds between frames')
    p.add_argument('--device', default=None, help='Device to run on (cuda or cpu). If omitted, will use cuda if available')
    return p.parse_args()


def main():
    args = parse_args()
    labels = [l.strip() for l in args.labels.split(',') if l.strip()]

    # Try to import detectron2 and detic
    try:
        import torch
    except Exception as e:
        print('ERROR: torch is required for Detic. Install pytorch first. Message:', e)
        sys.exit(1)

    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog
    except Exception as e:
        print('ERROR: detectron2 not available. Install detectron2. See https://github.com/facebookresearch/detectron2')
        print('Detailed error:', e)
        sys.exit(1)

    # Detic optional helpers
    try:
        # Detic injects its config extension; importing may fail if Detic isn't installed properly
        from detic.config import add_detic_config
    except Exception as e:
        print('WARNING: Detic helpers not available (detic package not importable). You may still run a Detectron2 model if you provide a compatible config.')
        add_detic_config = None

    # Build config
    cfg = get_cfg()
    if add_detic_config is not None:
        try:
            add_detic_config(cfg)
        except Exception:
            # ignore if it fails
            pass

    if args.config:
        if not os.path.isfile(args.config):
            print('Config file not found:', args.config)
            sys.exit(1)
        cfg.merge_from_file(args.config)
    else:
        print('No config provided. Using default COCO R50 config if available. For best open-vocab support provide a Detic config file.')
        # Attempt to use a common detectron2 R50 config
        try:
            # This path depends on detectron2 model zoo availability; user is encouraged to provide a config
            from detectron2.model_zoo import get_config
            cfg = get_config('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml')
        except Exception:
            pass

    # weights
    if args.weights:
        if not os.path.isfile(args.weights):
            print('Weights file not found:', args.weights)
            sys.exit(1)
        cfg.MODEL.WEIGHTS = args.weights
    else:
        print('No weights provided. You must supply a Detic-trained model weights for open-vocab detection.')
        # continue; predictor creation will probably fail

    # device
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.MODEL.DEVICE = device

    print('Using device:', device)

    # Create predictor
    try:
        predictor = DefaultPredictor(cfg)
    except Exception as e:
        print('Failed to create DefaultPredictor; check config/weights. Error:')
        print(e)
        sys.exit(1)

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Failed to open webcam')
        sys.exit(1)

    print('Starting webcam loop. Press Ctrl+C to stop.')
    try:
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                print('Failed to read frame')
                break

            # Detic open-vocab may accept a `text` kwarg (depending on implementation). We'll try both.
            outputs = None
            try:
                outputs = predictor(frame, text=labels)
            except TypeError:
                # predictor may not accept text - fallback to plain predictor
                outputs = predictor(frame)

            # outputs is a dict with 'instances'
            instances = outputs.get('instances', None) if isinstance(outputs, dict) else None

            if instances is not None:
                # get fields safely
                try:
                    boxes = instances.pred_boxes.tensor.cpu().numpy()
                except Exception:
                    boxes = None
                try:
                    scores = instances.scores.cpu().numpy()
                except Exception:
                    scores = None
                try:
                    classes = instances.pred_classes.cpu().numpy()
                except Exception:
                    classes = None

                # metadata may contain thing_classes
                try:
                    meta = MetadataCatalog.get(cfg.DATASETS.TEST[0]) if len(cfg.DATASETS.TEST) else None
                    thing_classes = meta.thing_classes if meta and hasattr(meta, 'thing_classes') else None
                except Exception:
                    thing_classes = None

                # visualize
                if args.show:
                    v = Visualizer(frame[:, :, ::-1])
                    out = v.draw_instance_predictions(instances.to('cpu'))
                    cv2.imshow('Detic - webcam', out.get_image()[:, :, ::-1])
                else:
                    # Print detections
                    if boxes is not None:
                        for i, box in enumerate(boxes):
                            cls_name = None
                            if thing_classes is not None and classes is not None and classes[i] < len(thing_classes):
                                cls_name = thing_classes[int(classes[i])]
                            print(f'Box {i}: {cls_name or classes[i] if classes is not None else "?"} score={scores[i] if scores is not None else "?"} box={box}')

            # rate limiting
            elapsed = time.time() - start
            if elapsed < args.rate:
                time.sleep(args.rate - elapsed)

            if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print('Stopping')
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
