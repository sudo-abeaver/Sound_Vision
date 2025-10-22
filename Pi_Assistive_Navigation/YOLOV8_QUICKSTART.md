# YOLOv8 Quick Start 🚀

Get the assistive navigation system running with **YOLOv8 + CLIP** in 10 minutes!

## Why This Version?

✅ **No model training** - Works out of the box  
✅ **Auto-downloads** - Models download automatically  
✅ **Zero-shot scenes** - Change scene labels without retraining  
✅ **Your existing setup** - Uses your HuggingFace API key  

---

## 1. Install Dependencies (5 min)

```bash
cd Pi_Assistive_Navigation

# Install Python packages
pip3 install -r requirements_yolov8.txt

# PyTorch for CPU (if on Raspberry Pi)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**On Mac/Linux development machine:**
```bash
pip3 install torch torchvision  # Will use accelerated version
```

---

## 2. Setup Environment (1 min)

```bash
# Copy .env template
cp .env.example .env

# Add your HuggingFace token (optional but recommended)
echo "HUGGINGFACE_TOKEN=your_token_here" >> .env
```

Get your token at: https://huggingface.co/settings/tokens

**Note**: Token is optional. Without it, models still download but you may hit rate limits.

---

## 3. Add Soundscapes (2 min)

Add at least one audio file to test:

```bash
# Example: Add a forest soundscape
cp ~/Music/forest_ambience.mp3 soundscapes/forest/

# Or download free one
wget https://freesound.org/...  # (Find from Freesound.org)
```

See [soundscapes/README.md](soundscapes/README.md) for free sources.

---

## 4. Test Installation (2 min)

```bash
# Verify YOLOv8 + CLIP work
python3 scripts/test_yolov8_clip.py
```

Expected output:
```
============================================================
YOLOv8 + CLIP Integration Test
============================================================

1. Testing imports...
   ✓ OpenCV
   ✓ Ultralytics YOLOv8
   ✓ PyTorch
   ✓ HuggingFace Transformers (CLIP)

2. Testing YOLOv8...
   ✓ YOLOv8 loaded
   ✓ Inference successful: 125.3ms

3. Testing CLIP...
   ✓ CLIP loaded
   ✓ Inference successful: 89.2ms

✓ All core components working!
```

If this passes, you're ready! 🎉

---

## 5. Run System (Now!)

### Without Hardware (Testing)

```bash
# Run without ultrasonic sensor (simulation mode)
python3 main_yolov8.py
```

The system will:
- ✓ Load YOLOv8 and CLIP models (first run downloads ~500MB)
- ✓ Open webcam (if available)
- ✓ Play soundscapes
- ✓ Simulate distance readings

### With Hardware (Raspberry Pi)

Wire up your hardware first (see [docs/WIRING.md](docs/WIRING.md)):
- HC-SR04 ultrasonic sensor
- USB webcam
- Headphones

Then:
```bash
python3 main_yolov8.py
```

---

## Expected Behavior

### On Start

```
============================================================
ASSISTIVE NAVIGATION SYSTEM STARTING (YOLOv8)
============================================================
Mode: Fully Autonomous
- Collision avoidance: ACTIVE
- Scene awareness: ACTIVE (CLIP)
- Object detection: ACTIVE (YOLOv8)
- Ambient audio: ACTIVE
============================================================

Distance: 2.45m | Zone: 0 | Scene: indoor hallway | Soundscape: indoor
```

### During Operation

1. **Move your hand near sensor** → Should beep (faster as you get closer)
2. **Point camera at different scenes** → Soundscape changes automatically
3. **Show objects to camera** → Announces "Person ahead", "Car left", etc.

### Status Updates

Every 1-2 seconds you'll see:
```
Distance: 1.23m | Zone: 2 | Scene: forest | Soundscape: forest
Distance: 0.89m | Zone: 3 | Scene: forest | Soundscape: forest
```

Press `Ctrl+C` to stop.

---

## What Gets Downloaded (First Run)

| Model | Size | Purpose |
|-------|------|---------|
| YOLOv8-nano | ~6 MB | Object detection |
| CLIP ViT-B/32 | ~350 MB | Scene classification |
| **Total** | **~356 MB** | |

Downloads to `~/.cache/` - only happens once.

---

## Customization

### Change Scene Labels

Edit `config.yaml`:

```yaml
vision:
  scene_labels:
    - "my living room"
    - "my backyard"
    - "my office"
```

No retraining needed! CLIP does zero-shot classification.

### Change Detection Objects

```yaml
priority_objects:
  - person
  - dog
  - bicycle
  - chair  # Add any COCO class
```

See [COCO classes](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/).

### Adjust Speed vs Accuracy

**Faster** (for Pi):
```yaml
vision:
  yolo_imgsz: 256              # Smaller
  scene_inference_interval: 1.5
  detect_inference_interval: 1.2
```

**More Accurate**:
```yaml
vision:
  yolo_model: "models/yolov8s.pt"  # Larger model
  yolo_imgsz: 416                  # Higher resolution
```

---

## Troubleshooting

### Models Not Downloading

```bash
# Manual download YOLOv8
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Manual download CLIP
python3 -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

### "CUDA out of memory"

You're on CPU, this shouldn't happen. If it does:
```yaml
# In config, lower resolution
camera_width: 320
camera_height: 240
```

### No Audio

```bash
# Check pygame
python3 -c "import pygame; pygame.mixer.init(); print('OK')"

# Check files exist
ls -lh soundscapes/forest/
```

### Camera Not Found

```bash
# Check camera
ls /dev/video*

# Test with OpenCV
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"
```

---

## Using Your Existing Models

You mentioned you have models in `Sound_Vision/`:

```bash
# Copy your YOLOv8 model
cp ../Sound_Vision/models/yolov8s.pt models/

# Update config
nano config.yaml
# Change: yolo_model: "models/yolov8s.pt"
```

---

## Next Steps

1. ✅ **You're running!** System is working
2. 📁 **Add more soundscapes** to all folders
3. 🎛️ **Tune config.yaml** for your environment
4. 🔧 **Wire Pi hardware** if testing on development machine
5. 🚀 **Deploy to Pi** and make wearable

---

## Full Documentation

- [README_YOLOV8.md](README_YOLOV8.md) - Complete YOLOv8 guide
- [README.md](README.md) - General documentation
- [docs/WIRING.md](docs/WIRING.md) - Hardware setup
- [config.yaml](config.yaml) - All settings (well commented)

---

**You're all set! 🎉**

The system is running with state-of-the-art YOLOv8 object detection and CLIP scene classification, no training required!

**Questions?** Check [README_YOLOV8.md](README_YOLOV8.md) for detailed info.

