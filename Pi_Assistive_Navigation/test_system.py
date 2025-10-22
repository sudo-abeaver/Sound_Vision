#!/usr/bin/env python3
"""
Test script for the YOLOv8 Assistive Navigation System
Tests all components without requiring camera access
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO")
    except ImportError as e:
        print(f"✗ Ultralytics: {e}")
        return False
    
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        print(f"✓ PyTorch {torch.__version__}")
        print("✓ Transformers (CLIP)")
    except ImportError as e:
        print(f"✗ PyTorch/Transformers: {e}")
        return False
    
    try:
        import pygame
        print("✓ Pygame")
    except ImportError as e:
        print(f"✗ Pygame: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML")
    except ImportError as e:
        print(f"✗ PyYAML: {e}")
        return False
    
    return True

def test_yolo():
    """Test YOLOv8 model loading"""
    print("\nTesting YOLO...")
    
    try:
        from ultralytics import YOLO
        
        # Test loading YOLOv8n (will auto-download if needed)
        print("Loading YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        print("✓ YOLOv8n model loaded successfully")
        
        # Test inference on a dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = model.predict(dummy_image, verbose=False)
        print("✓ YOLOv8 inference test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ YOLO test failed: {e}")
        return False

def test_clip():
    """Test CLIP model loading"""
    print("\nTesting CLIP...")
    
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
        
        # Test loading CLIP model
        print("Loading CLIP model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
        print("✓ CLIP model loaded successfully")
        
        # Test inference on a dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        text_inputs = processor(text=["forest", "park", "beach"], return_tensors='pt', padding=True)
        image_inputs = processor(images=dummy_image, return_tensors='pt')
        
        # Move to device
        for k, v in image_inputs.items():
            image_inputs[k] = v.to(device)
        for k, v in text_inputs.items():
            text_inputs[k] = v.to(device)
        
        with torch.no_grad():
            image_emb = model.get_image_features(**image_inputs)
            text_emb = model.get_text_features(**text_inputs)
            print("✓ CLIP inference test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ CLIP test failed: {e}")
        return False

def test_audio():
    """Test audio system"""
    print("\nTesting Audio...")
    
    try:
        import pygame
        import numpy as np
        
        # Initialize pygame mixer
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        print("✓ Pygame mixer initialized")
        
        # Test generating a beep
        sample_rate = 44100
        duration = 0.1
        frequency = 440
        samples = int(sample_rate * duration)
        wave = np.sin(2 * np.pi * frequency * np.arange(samples) / sample_rate)
        wave = (wave * 32767 * 0.5).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        
        sound = pygame.sndarray.make_sound(stereo_wave)
        print("✓ Audio generation test passed")
        
        pygame.mixer.quit()
        return True
        
    except Exception as e:
        print(f"✗ Audio test failed: {e}")
        return False

def test_soundscapes():
    """Test soundscape files"""
    print("\nTesting Soundscapes...")
    
    soundscape_path = Path("soundscapes")
    if not soundscape_path.exists():
        print("✗ Soundscapes directory not found")
        return False
    
    print(f"✓ Soundscapes directory found: {soundscape_path}")
    
    # Check each soundscape folder
    folders = ['forest', 'park', 'beach', 'city', 'indoor', 'museum', 'plaza', 'parking', 'residential']
    for folder in folders:
        folder_path = soundscape_path / folder
        if folder_path.exists():
            audio_files = list(folder_path.glob("*.wav")) + list(folder_path.glob("*.mp3"))
            print(f"✓ {folder}: {len(audio_files)} audio files")
        else:
            print(f"✗ {folder}: folder not found")
    
    return True

def test_camera():
    """Test camera access (may require permission)"""
    print("\nTesting Camera...")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✓ Camera access successful")
                print(f"  Frame shape: {frame.shape}")
                cap.release()
                return True
            else:
                print("✗ Camera opened but failed to read frame")
                cap.release()
                return False
        else:
            print("✗ Camera not accessible (may need permission)")
            return False
            
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("YOLOv8 Assistive Navigation System - Component Tests")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("YOLO", test_yolo),
        ("CLIP", test_clip),
        ("Audio", test_audio),
        ("Soundscapes", test_soundscapes),
        ("Camera", test_camera),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:15} {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    if not all_passed:
        print("\nNote: Camera access may require permission in System Preferences > Security & Privacy > Camera")
        print("The system can still run in simulation mode without camera access.")

if __name__ == "__main__":
    main()

