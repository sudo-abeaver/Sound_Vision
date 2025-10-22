# Assistive Navigation - YOLOv8 Version 🚀

This is the **YOLOv8 + CLIP** version of the assistive navigation system, using:

- **Ultralytics YOLOv8-nano** for object detection
- **HuggingFace CLIP** for zero-shot scene classification
- **Same architecture** as the TFLite version

## Why YOLOv8?

✅ **No model training required** - YOLOv8 comes pre-trained on COCO  
✅ **Auto-downloads** - First run downloads yolov8n.pt automatically  
✅ **Zero-shot scenes** - CLIP classifies any scene without training  
✅ **Better accuracy** - State-of-the-art detection and classification  
✅ **Easy integration** - Uses your existing HuggingFace API key  

## Quick Start

### 1. Install Dependencies

```bash
# System packages
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y libatlas-base-dev libopenblas-dev
sudo apt-get install -y espeak portaudio19-dev
sudo apt-get install -y libsdl2-dev libsdl2-mixer-dev

# Python packages
pip3 install -r requirements_yolov8.txt

# PyTorch for Pi 4 (CPU)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Setup Environment

```bash
# Create .env file with your HuggingFace token (optional)
cp .env.example .env
nano .env
# Add: HUGGINGFACE_TOKEN=your_token_here
```

**Note**: The token is optional. Models will download without it, but you might hit rate limits.

### 3. Add Soundscapes

Add audio files (.mp3 or .wav) to the `soundscapes/` folders:

```
soundscapes/
  forest/
  park/
  beach/
  city/
  residential/
  indoor/
  museum/
  plaza/
  parking/
```

See [soundscapes/README.md](soundscapes/README.md) for sources.

### 4. Wire Hardware

Follow [docs/WIRING.md](docs/WIRING.md):
- HC-SR04 ultrasonic sensor (GPIO 23/24 with voltage divider!)
- USB webcam
- Headphones or Bluetooth speaker

### 5. Run!

```bash
python3 main_yolov8.py
```

**First run**: YOLOv8 and CLIP models will download automatically (~500MB total).

Expected output:
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
```

## How It Works

### Object Detection (YOLOv8-nano)

- **Model**: `yolov8n.pt` (auto-downloads, ~6MB)
- **Classes**: 80 COCO objects (person, car, bicycle, dog, etc.)
- **Speed**: ~5-8 FPS @ 320×320 on Pi 4
- **Accuracy**: Excellent for common objects

**What it detects**:
```python
priority_objects = [
    'person', 'bicycle', 'car', 'motorcycle', 
    'bus', 'truck', 'dog', 'cat', 'bench'
]
```

Only announces priority objects to avoid spam.

### Scene Classification (CLIP)

- **Model**: `openai/clip-vit-base-patch32` (HuggingFace)
- **Method**: Zero-shot (no training needed!)
- **Speed**: ~2-3 FPS @ 224×224 on Pi 4
- **Accuracy**: Good for general scenes

**Scene labels** (defined in config.yaml):
```yaml
scene_labels:
  - "forest"
  - "park"
  - "beach"
  - "urban street"
  - "residential area"
  - "indoor hallway"
  - "museum gallery"
  - "store"
  - "plaza"
  - "parking lot"
```

You can add/change these without retraining!

### Alternating Inference

To manage CPU load, scene and object detection alternate:

```
Frame 0: Scene classification (CLIP)
Frame 1: Skip
Frame 2: Object detection (YOLOv8)
Frame 3: Skip
Frame 4: Scene classification (CLIP)
...
```

This achieves ~1-2 effective FPS with both models running.

## Configuration

Edit `config.yaml`:

### Adjust YOLOv8 Settings

```yaml
vision:
  yolo_model: "models/yolov8n.pt"  # or yolov8s.pt for better accuracy
  yolo_conf: 0.5                   # Detection threshold (0.0-1.0)
  yolo_imgsz: 320                  # Lower = faster (256, 320, 416, 640)
```

### Change Scene Labels

```yaml
vision:
  scene_labels:
    - "coffee shop"      # Add new scenes
    - "library"
    - "gym"
    - "classroom"
```

CLIP will classify them with **no training required**!

### Priority Objects

```yaml
priority_objects:
  - person
  - dog
  - bicycle
  - stairs  # Note: COCO doesn't have "stairs" - use scene classification
```

## Performance Tuning

### For Faster Speed

```yaml
vision:
  yolo_imgsz: 256                  # Smaller = faster
  scene_inference_interval: 1.5   # Less frequent CLIP
  detect_inference_interval: 1.2  # Less frequent YOLOv8
  camera_width: 320                # Lower resolution
  camera_height: 240
```

### For Better Accuracy

```yaml
vision:
  yolo_model: "models/yolov8s.pt"  # Larger model
  yolo_imgsz: 416                  # Larger inference size
  yolo_conf: 0.4                   # Lower threshold
  camera_width: 640                # Higher resolution
```

**Trade-off**: Better accuracy = slower FPS

## Comparison: TFLite vs YOLOv8

| Feature | TFLite (main.py) | YOLOv8 (main_yolov8.py) |
|---------|------------------|-------------------------|
| **Setup** | Requires model conversion | Auto-downloads |
| **Training** | Must train custom models | Pre-trained on COCO |
| **Speed** | Faster (~8 FPS) | Slower (~5 FPS) |
| **Accuracy** | Depends on training | Excellent out-of-box |
| **Scene Classes** | Fixed at training | Change anytime (CLIP) |
| **Model Size** | Smaller (~5MB) | Larger (~50MB) |
| **Best For** | Custom domains, production | Quick start, prototyping |

**Recommendation**: Start with YOLOv8, then optimize with TFLite if needed.

## Troubleshooting

### YOLOv8 Not Downloading

```bash
# Manual download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/

# Or use Python
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### CLIP Model Not Loading

```bash
# Check HuggingFace connection
python3 -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"

# If rate-limited, set your token in .env:
export HUGGINGFACE_TOKEN=your_token_here
```

### Too Slow on Pi 4

1. **Lower inference size**:
   ```yaml
   yolo_imgsz: 256  # Instead of 320
   ```

2. **Increase intervals**:
   ```yaml
   scene_inference_interval: 2.0
   detect_inference_interval: 1.5
   ```

3. **Use yolov8n** (not yolov8s/m/l):
   ```yaml
   yolo_model: "models/yolov8n.pt"  # Nano is fastest
   ```

4. **Disable scene classification** (object detection only):
   ```python
   # In main_yolov8.py, comment out:
   # scene = self.vision.classify_scene(frame)
   ```

### Out of Memory

Pi 4 with 2GB RAM might struggle. Solutions:

```bash
# Increase swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set: CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Or use 4GB/8GB Pi 4
```

## Using Your Existing YOLOv8 Setup

I see you already have YOLOv8 in `Sound_Vision/`. You can reuse your models:

```bash
# Copy your trained model
cp ../Sound_Vision/models/yolov8s.pt models/

# Update config
nano config.yaml
# Change: yolo_model: "models/yolov8s.pt"

# Run with your model
python3 main_yolov8.py
```

## Advanced: Custom YOLOv8 Training

If you want to detect custom objects (e.g., specific obstacles):

```bash
# Train on custom dataset
yolo train model=yolov8n.pt data=your_dataset.yaml epochs=50 imgsz=320

# Export for Pi
yolo export model=runs/detect/train/weights/best.pt format=torchscript

# Use in config
# yolo_model: "models/custom_yolov8n.pt"
```

See [Ultralytics docs](https://docs.ultralytics.com) for training guide.

## Next Steps

1. **Test system**: `python3 main_yolov8.py`
2. **Add soundscapes**: Populate `soundscapes/` folders
3. **Tune config**: Adjust for your environment
4. **Mount hardware**: Integrate into wearable setup
5. **Auto-start**: Setup systemd service (see main README)

## Resources

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [HuggingFace CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
- [COCO Dataset Classes](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/)
- [Main README](README.md) - Full documentation

## Support

Having issues? Check:

1. [Main README](README.md) - General troubleshooting
2. [WIRING guide](docs/WIRING.md) - Hardware help
3. [GitHub Issues](https://github.com/your-repo/issues)

---

**Built with ❤️ using YOLOv8 + CLIP**

*"Bringing state-of-the-art AI to assistive technology"*

