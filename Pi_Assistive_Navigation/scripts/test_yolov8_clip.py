#!/usr/bin/env python3
"""
Test YOLOv8 + CLIP integration
Verifies models load and work correctly before running main system
"""

import os
import sys
import time
from pathlib import Path

print("="*60)
print("YOLOv8 + CLIP Integration Test")
print("="*60)

# Test imports
print("\n1. Testing imports...")
try:
    import cv2
    print("   ✓ OpenCV")
except ImportError as e:
    print(f"   ✗ OpenCV: {e}")
    sys.exit(1)

try:
    from ultralytics import YOLO
    print("   ✓ Ultralytics YOLOv8")
except ImportError as e:
    print(f"   ✗ Ultralytics: {e}")
    print("   Install with: pip install ultralytics")
    sys.exit(1)

try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"   ✗ PyTorch: {e}")
    print("   Install with: pip install torch torchvision")
    sys.exit(1)

try:
    from transformers import CLIPProcessor, CLIPModel
    print("   ✓ HuggingFace Transformers (CLIP)")
except ImportError as e:
    print(f"   ✗ Transformers: {e}")
    print("   Install with: pip install transformers")
    sys.exit(1)

try:
    from PIL import Image
    import numpy as np
    print("   ✓ PIL/Numpy")
except ImportError as e:
    print(f"   ✗ PIL/Numpy: {e}")
    sys.exit(1)

# Test YOLOv8
print("\n2. Testing YOLOv8...")
try:
    model_path = "models/yolov8n.pt"
    if not os.path.exists(model_path):
        print(f"   Model not found at {model_path}")
        print("   Downloading yolov8n.pt...")
    
    model = YOLO('yolov8n.pt')  # Auto-downloads if missing
    print(f"   ✓ YOLOv8 loaded")
    
    # Test inference with dummy image
    print("   Running test inference...")
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    start = time.time()
    results = model.predict(source=dummy_image, conf=0.5, imgsz=320, verbose=False)
    inference_time = (time.time() - start) * 1000
    
    print(f"   ✓ Inference successful: {inference_time:.1f}ms")
    print(f"   Available classes: {len(model.names)}")
    
    # Show some class names
    sample_classes = list(model.names.values())[:10]
    print(f"   Sample classes: {', '.join(sample_classes)}")
    
except Exception as e:
    print(f"   ✗ YOLOv8 error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test CLIP
print("\n3. Testing CLIP...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    print("   Loading CLIP model (this may take a moment)...")
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
    print("   ✓ CLIP loaded")
    
    # Test zero-shot classification
    print("   Running test zero-shot classification...")
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    labels = ["forest", "city", "beach", "indoor"]
    
    text_inputs = processor(text=labels, return_tensors='pt', padding=True)
    image_inputs = processor(images=dummy_image, return_tensors='pt')
    
    # Move to device
    for k, v in image_inputs.items():
        image_inputs[k] = v.to(device)
    for k, v in text_inputs.items():
        text_inputs[k] = v.to(device)
    
    start = time.time()
    with torch.no_grad():
        image_emb = clip_model.get_image_features(**image_inputs)
        text_emb = clip_model.get_text_features(**text_inputs)
        image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
        scores = (100.0 * image_emb @ text_emb.T).softmax(dim=-1)
    
    inference_time = (time.time() - start) * 1000
    print(f"   ✓ Inference successful: {inference_time:.1f}ms")
    
    scores = scores[0].cpu().tolist()
    ranked = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
    print(f"   Top prediction: {ranked[0][0]} ({ranked[0][1]:.3f})")
    
except Exception as e:
    print(f"   ✗ CLIP error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test camera
print("\n4. Testing camera...")
try:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("   ✗ Could not open camera")
        print("   Note: Camera may not be available on this system")
    else:
        ret, frame = camera.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"   ✓ Camera working: {w}x{h}")
        else:
            print("   ✗ Could not read frame")
        camera.release()
except Exception as e:
    print(f"   ⚠ Camera test skipped: {e}")

# Test with real image if available
print("\n5. Testing with real image (optional)...")
test_images = [
    "../Sound_Vision/data/Ash_Tree_-_geograph.org.uk_-_590710.jpg",
    "test_image.jpg",
]

test_image_path = None
for path in test_images:
    if os.path.exists(path):
        test_image_path = path
        break

if test_image_path:
    print(f"   Found test image: {test_image_path}")
    try:
        # Load image
        frame = cv2.imread(test_image_path)
        if frame is None:
            raise ValueError("Could not read image")
        
        print(f"   Image size: {frame.shape[1]}x{frame.shape[0]}")
        
        # YOLOv8 detection
        print("   Running YOLOv8 detection...")
        start = time.time()
        results = model.predict(source=frame, conf=0.5, imgsz=320, verbose=False)
        yolo_time = (time.time() - start) * 1000
        
        r = results[0]
        num_detections = len(r.boxes) if r.boxes is not None else 0
        print(f"   ✓ YOLOv8: {num_detections} objects detected in {yolo_time:.1f}ms")
        
        if num_detections > 0:
            for i, box in enumerate(r.boxes[:3]):  # Show first 3
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = r.names[cls]
                print(f"     - {name} ({conf:.2f})")
        
        # CLIP scene classification
        print("   Running CLIP scene classification...")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        scene_labels = ["forest", "park", "beach", "urban street", "indoor hallway", "museum"]
        
        text_inputs = processor(text=scene_labels, return_tensors='pt', padding=True)
        image_inputs = processor(images=image, return_tensors='pt')
        
        for k, v in image_inputs.items():
            image_inputs[k] = v.to(device)
        for k, v in text_inputs.items():
            text_inputs[k] = v.to(device)
        
        start = time.time()
        with torch.no_grad():
            image_emb = clip_model.get_image_features(**image_inputs)
            text_emb = clip_model.get_text_features(**text_inputs)
            image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
            scores = (100.0 * image_emb @ text_emb.T).softmax(dim=-1)
        
        clip_time = (time.time() - start) * 1000
        scores = scores[0].cpu().tolist()
        ranked = sorted(zip(scene_labels, scores), key=lambda x: x[1], reverse=True)
        
        print(f"   ✓ CLIP: Top scene predictions ({clip_time:.1f}ms)")
        for label, score in ranked[:3]:
            print(f"     - {label}: {score:.3f}")
        
    except Exception as e:
        print(f"   ✗ Real image test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("   ⚠ No test image found, skipping")
    print("   Place a test image at: test_image.jpg")

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("✓ All core components working!")
print()
print("Performance estimates for Pi 4:")
print(f"  YOLOv8-nano @ 320: ~5-8 FPS")
print(f"  CLIP @ 224: ~2-3 FPS")
print(f"  Combined (alternating): ~1-2 FPS effective")
print()
print("You can now run the full system:")
print("  python3 main_yolov8.py")
print("="*60)


