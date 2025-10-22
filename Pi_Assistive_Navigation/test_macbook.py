#!/usr/bin/env python3
"""
Quick test script for MacBook development
Tests all components without requiring hardware
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import cv2
        print("   ✓ OpenCV")
    except ImportError as e:
        print(f"   ✗ OpenCV: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("   ✓ Ultralytics YOLOv8")
    except ImportError as e:
        print(f"   ✗ Ultralytics: {e}")
        return False
    
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        print("   ✓ PyTorch + Transformers (CLIP)")
    except ImportError as e:
        print(f"   ✗ PyTorch/Transformers: {e}")
        return False
    
    try:
        import pygame
        print("   ✓ Pygame")
    except ImportError as e:
        print(f"   ✗ Pygame: {e}")
        return False
    
    try:
        import yaml
        print("   ✓ PyYAML")
    except ImportError as e:
        print(f"   ✗ PyYAML: {e}")
        return False
    
    return True

def test_camera():
    """Test camera access"""
    print("\n📷 Testing camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("   ⚠️  Camera not available (may be in use by another app)")
            return False
        
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"   ✓ Camera working: {w}x{h}")
            cap.release()
            return True
        else:
            print("   ✗ Could not read frame")
            cap.release()
            return False
            
    except Exception as e:
        print(f"   ✗ Camera error: {e}")
        return False

def test_audio():
    """Test audio system"""
    print("\n🔊 Testing audio...")
    
    try:
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        
        # Test beep generation
        sample_rate = 44100
        duration = 0.1
        frequency = 440
        samples = int(sample_rate * duration)
        wave = np.sin(2 * np.pi * frequency * np.arange(samples) / sample_rate)
        wave = (wave * 32767 * 0.1).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        
        sound = pygame.sndarray.make_sound(stereo_wave)
        sound.play()
        time.sleep(0.2)
        
        print("   ✓ Audio system working")
        pygame.mixer.quit()
        return True
        
    except Exception as e:
        print(f"   ✗ Audio error: {e}")
        return False

def test_models():
    """Test YOLOv8 and CLIP models"""
    print("\n🤖 Testing models...")
    
    try:
        from ultralytics import YOLO
        import torch
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
        
        # Test YOLOv8
        print("   Loading YOLOv8...")
        model_path = "yolov8n.pt"
        if not os.path.exists(model_path):
            print("   Downloading YOLOv8 model...")
        
        yolo = YOLO('yolov8n.pt')  # Auto-downloads
        print("   ✓ YOLOv8 loaded")
        
        # Test CLIP
        print("   Loading CLIP...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
        print(f"   ✓ CLIP loaded on {device}")
        
        # Quick inference test
        print("   Running inference test...")
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # YOLOv8 test
        start = time.time()
        results = yolo.predict(source=dummy_image, conf=0.5, imgsz=320, verbose=False)
        yolo_time = (time.time() - start) * 1000
        print(f"   ✓ YOLOv8: {yolo_time:.1f}ms")
        
        # CLIP test
        image = Image.fromarray(dummy_image)
        labels = ["forest", "city", "indoor"]
        text_inputs = processor(text=labels, return_tensors='pt', padding=True)
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
        print(f"   ✓ CLIP: {clip_time:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Model error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_soundscapes():
    """Test soundscape directories"""
    print("\n🎵 Testing soundscape directories...")
    
    soundscape_dirs = [
        "soundscapes/forest",
        "soundscapes/park", 
        "soundscapes/beach",
        "soundscapes/city",
        "soundscapes/residential",
        "soundscapes/indoor",
        "soundscapes/museum",
        "soundscapes/plaza",
        "soundscapes/parking"
    ]
    
    all_exist = True
    for dir_path in soundscape_dirs:
        if os.path.exists(dir_path):
            files = list(Path(dir_path).glob("*.mp3")) + list(Path(dir_path).glob("*.wav"))
            if files:
                print(f"   ✓ {dir_path} ({len(files)} files)")
            else:
                print(f"   ⚠️  {dir_path} (empty)")
                all_exist = False
        else:
            print(f"   ✗ {dir_path} (missing)")
            all_exist = False
    
    if not all_exist:
        print("   💡 Add audio files (.mp3 or .wav) to soundscape directories")
        print("   💡 See soundscapes/README.md for free sources")
    
    return all_exist

def main():
    """Run all tests"""
    print("="*60)
    print("🧪 ASSISTIVE NAVIGATION - MACBOOK TEST")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Camera", test_camera),
        ("Audio", test_audio),
        ("Models", test_models),
        ("Soundscapes", test_soundscapes)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ✗ {name} test failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:15s}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 All tests passed! You can run:")
        print("   python3 main_yolov8.py")
    else:
        print("\n⚠️  Some tests failed. Install missing dependencies:")
        print("   pip install ultralytics torch transformers pygame opencv-python pyyaml")
    
    print("="*60)

if __name__ == "__main__":
    main()
