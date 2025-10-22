#!/usr/bin/env python3
"""
Test YOLOv8 object detection with sample images
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def test_yolo_detection():
    """Test YOLOv8 on sample images"""
    print("Testing YOLOv8 Object Detection...")
    
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Test with sample images if available
    sample_images = [
        "data/Ash_Tree_-_geograph.org.uk_-_590710.jpg",
        "Testing stuff/Ash_Tree_-_geograph.org.uk_-_590710.jpg",
        "runs/detect/predict/Ash_Tree_-_geograph.org.uk_-_590710.jpg"
    ]
    
    for img_path in sample_images:
        if Path(img_path).exists():
            print(f"\nTesting with: {img_path}")
            
            # Run inference
            results = model.predict(
                source=img_path,
                conf=0.5,
                imgsz=320,
                verbose=False
            )
            
            # Process results
            if results and len(results) > 0:
                r = results[0]
                print(f"  Image shape: {r.orig_img.shape}")
                
                if r.boxes is not None and len(r.boxes) > 0:
                    print(f"  Detected {len(r.boxes)} objects:")
                    for i, box in enumerate(r.boxes):
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        name = r.names[cls] if cls in r.names else str(cls)
                        
                        print(f"    {i+1}. {name} (confidence: {conf:.2f})")
                        print(f"       Bbox: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
                else:
                    print("  No objects detected")
            else:
                print("  No results returned")
            
            break
    else:
        print("No sample images found, creating a test image...")
        
        # Create a test image with some objects
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some colored rectangles to simulate objects
        cv2.rectangle(test_img, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
        cv2.rectangle(test_img, (300, 150), (400, 250), (0, 255, 0), -1)  # Green rectangle
        cv2.rectangle(test_img, (500, 200), (600, 300), (0, 0, 255), -1)  # Red rectangle
        
        # Run inference on test image
        results = model.predict(
            source=test_img,
            conf=0.3,
            imgsz=320,
            verbose=False
        )
        
        print(f"Test image shape: {test_img.shape}")
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                print(f"Detected {len(r.boxes)} objects in test image")
            else:
                print("No objects detected in test image")

if __name__ == "__main__":
    test_yolo_detection()

