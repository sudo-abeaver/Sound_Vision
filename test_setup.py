#!/usr/bin/env python3
"""
Test script to verify Sound_Vision setup
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import ultralytics
        print("✓ ultralytics imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ultralytics: {e}")
        return False
    
    try:
        import cv2
        print("✓ opencv-python imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import cv2: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PIL: {e}")
        return False
    
    return True

def test_models():
    """Test that model files exist"""
    print("\nTesting model files...")
    
    model_files = [
        "models/yolov8n.pt",
        "models/yolov8s.pt"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"✓ {model_file} found ({size:.1f} MB)")
        else:
            print(f"✗ {model_file} not found")
            return False
    
    return True

def test_data():
    """Test that test data exists"""
    print("\nTesting data files...")
    
    data_files = [
        "data/Ash_Tree_-_geograph.org.uk_-_590710.jpg"
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            size = os.path.getsize(data_file) / 1024  # KB
            print(f"✓ {data_file} found ({size:.1f} KB)")
        else:
            print(f"✗ {data_file} not found")
            return False
    
    return True

def test_yolo_functionality():
    """Test basic YOLO functionality"""
    print("\nTesting YOLO functionality...")
    
    try:
        from ultralytics import YOLO
        import os
        
        model_path = "models/yolov8s.pt"
        if not os.path.exists(model_path):
            print(f"✗ Model file not found: {model_path}")
            return False
        
        # Load model
        model = YOLO(model_path)
        print("✓ YOLO model loaded successfully")
        
        # Test with a simple prediction
        test_image = "data/Ash_Tree_-_geograph.org.uk_-_590710.jpg"
        if os.path.exists(test_image):
            results = model.predict(source=test_image, conf=0.25, save=False, verbose=False)
            print("✓ YOLO prediction completed successfully")
            print(f"  Found {len(results[0].boxes) if results[0].boxes is not None else 0} detections")
        else:
            print("✗ Test image not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ YOLO functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Sound_Vision Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_models,
        test_data,
        test_yolo_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Sound_Vision is ready to use.")
        return 0
    else:
        print("✗ Some tests failed. Please check the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
