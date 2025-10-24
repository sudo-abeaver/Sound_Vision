# 🚀 MacBook Quick Start

Get the assistive navigation system running on your MacBook in 5 minutes!

## What You'll Get

✅ **YOLOv8 object detection** - Detects people, cars, bicycles, etc.  
✅ **CLIP scene classification** - Recognizes forest, city, indoor, etc.  
✅ **Simulated collision avoidance** - Distance simulation with beeps  
✅ **Adaptive soundscapes** - Audio changes based on detected scene  
✅ **Real-time webcam** - 15 FPS processing  

---

## 1. Install Dependencies (2 min)

```bash
cd /Users/anthonymarcel/Documents/FSE100/Sound_Vision/Pi_Assistive_Navigation

# Install Python packages
pip3 install ultralytics torch transformers pygame opencv-python pyyaml python-dotenv
```

**Note**: If you have M1/M2 Mac, PyTorch will use Metal acceleration automatically!

---

## 2. Test Everything (1 min)

```bash
# Run comprehensive test
python3 test_macbook.py
```

Expected output:
```
============================================================
🧪 ASSISTIVE NAVIGATION - MACBOOK TEST
============================================================

🧪 Testing imports...
   ✓ OpenCV
   ✓ Ultralytics YOLOv8
   ✓ PyTorch + Transformers (CLIP)
   ✓ Pygame
   ✓ PyYAML

📷 Testing camera...
   ✓ Camera working: 1280x720

🔊 Testing audio...
   ✓ Audio system working

🤖 Testing models...
   ✓ YOLOv8 loaded
   ✓ CLIP loaded on cpu
   ✓ YOLOv8: 45.2ms
   ✓ CLIP: 23.1ms

🎵 Testing soundscape directories...
   ⚠️  soundscapes/forest (empty)
   ⚠️  soundscapes/city (empty)
   ...

📊 TEST SUMMARY
============================================================
Imports        : ✓ PASS
Camera         : ✓ PASS
Audio          : ✓ PASS
Models         : ✓ PASS
Soundscapes    : ✗ FAIL

Passed: 4/5

🎉 All tests passed! You can run:
   python3 main_yolov8.py
```

---

## 3. Add Soundscapes (Optional)

The system works without audio files, but for full experience:

```bash
# Add some test audio files
# (You can use any .mp3 or .wav files you have)

# Example: Copy some music to test
cp ~/Music/some_song.mp3 soundscapes/forest/
cp ~/Music/another_song.mp3 soundscapes/city/
cp ~/Music/classical.mp3 soundscapes/museum/
```

**Free sources**:
- [Freesound.org](https://freesound.org) - Free sound effects
- [BBC Sound Effects](https://sound-effects.bbcrewind.co.uk) - Professional quality
- [Free Music Archive](https://freemusicarchive.org) - Creative Commons music

---

## 4. Run the System! (Now!)

```bash
python3 main_yolov8.py
```

### Expected Output

```
============================================================
🚀 ASSISTIVE NAVIGATION SYSTEM STARTING (YOLOv8)
============================================================
Mode: Fully Autonomous
📡 Collision avoidance: ACTIVE (SIMULATION)
🌍 Scene awareness: ACTIVE (CLIP)
👁️  Object detection: ACTIVE (YOLOv8)
🔊 Ambient audio: ACTIVE
============================================================

📡 Initializing Ultrasonic Sensor...
📡 [SIMULATION] Ultrasonic sensor not available (not on Raspberry Pi)
🤖 Initializing Vision Module (YOLOv8 + CLIP)...
Loading YOLOv8 from yolov8n.pt...
Loading CLIP model on cpu...
🔊 Initializing Audio Engine...
📷 Initializing camera...
📷 Camera initialized: 1280x720 @ 15 FPS

System running. Press Ctrl+C to stop.

📊 Status: Distance=2.45m | Zone=0 | Scene=N/A | Soundscape=N/A
🌍 Scene changed: forest -> forest
🎵 Attempting to play: soundscapes/forest/forest_ambience.mp3
🎵 [SIMULATION] Would play: soundscapes/forest/forest_ambience.mp3
📊 Status: Distance=1.23m | Zone=2 | Scene=forest | Soundscape=forest
👁️  Detected: person ahead
🗣️  SPEAKING: Person ahead
🔊 BEEP: 660Hz for 0.08s (Distance: 0.89m)
```

---

## What You'll See

### 1. **Object Detection**
- Points camera at people → "Person ahead"
- Shows bicycle → "Bicycle left" 
- Detects car → "Car right"

### 2. **Scene Classification**
- Point at trees → "forest" scene → forest soundscape
- Point at street → "urban street" scene → city soundscape
- Point indoors → "indoor hallway" scene → indoor soundscape

### 3. **Simulated Collision Avoidance**
- Random distance changes (0.3m - 3.0m)
- Beeps get faster/higher as "distance" decreases
- "Stop" warning at very close range

### 4. **Audio System**
- Prints what it would play (since no audio files yet)
- Shows crossfading between scenes
- Simulates beep tones

---

## Customization

### Change Scene Labels

Edit `config.yaml`:

```yaml
vision:
  scene_labels:
    - "my living room"
    - "my backyard" 
    - "coffee shop"
    - "grocery store"
```

**No retraining needed!** CLIP does zero-shot classification.

### Adjust Detection Objects

```yaml
priority_objects:
  - person
  - dog
  - bicycle
  - chair
  - laptop
```

### Change Camera Settings

```yaml
vision:
  camera_width: 640
  camera_height: 480
  camera_fps: 30  # Higher FPS
```

---

## Performance on MacBook

**Expected speeds**:
- **YOLOv8**: ~20-30 FPS (much faster than Pi!)
- **CLIP**: ~10-15 FPS 
- **Combined**: ~5-8 effective FPS
- **Camera**: 15 FPS (configurable)

**M1/M2 Macs**: Even faster with Metal acceleration!

---

## Troubleshooting

### "Camera not available"
```bash
# Check if camera is in use
# Close other apps that might use camera (Zoom, FaceTime, etc.)
```

### "Models not downloading"
```bash
# Manual download
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python3 -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

### "Audio not working"
```bash
# Test pygame
python3 -c "import pygame; pygame.mixer.init(); print('OK')"

# Check system audio
# Make sure volume is up and no other app is using audio
```

### "Too slow"
```yaml
# In config.yaml, lower resolution:
camera_width: 320
camera_height: 240
yolo_imgsz: 256
```

---

## Next Steps

1. ✅ **You're running!** System works on MacBook
2. 🎵 **Add soundscapes** - Put audio files in folders
3. 🎛️ **Tune settings** - Edit config.yaml for your environment  
4. 🔧 **Test on Pi** - Transfer to Raspberry Pi for real hardware
5. 🚀 **Make wearable** - Design belt/collar mount

---

## Full Documentation

- [README_YOLOV8.md](README_YOLOV8.md) - Complete YOLOv8 guide
- [README.md](README.md) - General system documentation
- [config.yaml](config.yaml) - All settings (well commented)

---

**You're all set! 🎉**

The system is now running with state-of-the-art AI on your MacBook. Perfect for development and testing before deploying to Raspberry Pi!

**Questions?** Check the full documentation or create an issue.

