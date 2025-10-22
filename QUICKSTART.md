# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Quick Start (One Command!)
```bash
cd /Users/anthonymarcel/Documents/FSE100/Sound_Vision
./start.sh
```

### Alternative: Manual Activation
```bash
cd /Users/anthonymarcel/Documents/FSE100/Sound_Vision
source venv/bin/activate
```

### 2. Test Your Setup
```bash
python test_setup.py
```

### 3. Run Object Detection
```bash
# Basic detection on the test image
python scripts/detect_with_yolov8.py --image data/Ash_Tree_-_geograph.org.uk_-_590710.jpg --output output.jpg

# See what classes the model can detect
python scripts/detect_with_yolov8.py --image data/Ash_Tree_-_geograph.org.uk_-_590710.jpg --diagnose
```

### 4. Try Different Models
```bash
# Use the nano model (faster, less accurate)
python scripts/detect_with_yolov8.py --image data/Ash_Tree_-_geograph.org.uk_-_590710.jpg --model models/yolov8n.pt --output output_nano.jpg

# Use the small model (slower, more accurate)
python scripts/detect_with_yolov8.py --image data/Ash_Tree_-_geograph.org.uk_-_590710.jpg --model models/yolov8s.pt --output output_small.jpg
```

## 📁 Project Structure
```
Sound_Vision/
├── scripts/              # Python scripts
│   ├── detect_with_yolov8.py
│   ├── detic_webcam.py
│   └── stationary image processor.py
├── models/               # YOLO model files
│   ├── yolov8n.pt
│   └── yolov8s.pt
├── data/                 # Test images
│   └── Ash_Tree_-_geograph.org.uk_-_590710.jpg
├── runs/                 # Output directory
├── venv/                 # Virtual environment
├── requirements.txt      # Dependencies
├── setup.py             # Package setup
├── test_setup.py        # Setup verification
└── README.md            # Full documentation
```

## 🔧 Common Commands

### Object Detection
```bash
# Detect objects in an image
python scripts/detect_with_yolov8.py --image path/to/your/image.jpg --output result.jpg

# Adjust confidence threshold
python scripts/detect_with_yolov8.py --image path/to/your/image.jpg --conf 0.5 --output result.jpg
```

### Model Information
```bash
# See all available classes
python scripts/detect_with_yolov8.py --image data/Ash_Tree_-_geograph.org.uk_-_590710.jpg --diagnose
```

## 🐛 Troubleshooting

### Virtual Environment Issues
```bash
# If activation fails, recreate the environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Model Download Issues
The YOLO models should already be included. If missing:
```bash
# Download models manually
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -O models/yolov8s.pt
```

### Permission Issues
```bash
# Make scripts executable
chmod +x scripts/*.py
```

## 📚 Next Steps

1. **Read the full documentation**: Check `README.md` for detailed information
2. **Try different images**: Use your own images for detection
3. **Experiment with parameters**: Adjust confidence thresholds and models
4. **Explore advanced features**: Try CLIP integration or Detic webcam detection

## 🆘 Need Help?

- Check the `README.md` for detailed documentation
- Run `python test_setup.py` to verify your setup
- Look at the script help: `python scripts/detect_with_yolov8.py --help`
