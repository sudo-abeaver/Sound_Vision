# Sound_Vision

A computer vision project focused on object detection using YOLOv8 and advanced vision models.

## Project Overview

This project contains several computer vision tools and scripts for object detection:

- **YOLOv8 Detection**: Standard object detection using pre-trained YOLOv8 models
- **CLIP Integration**: Zero-shot image classification using CLIP models
- **Detic Integration**: Open-vocabulary object detection (experimental)
- **Webcam Support**: Real-time detection from webcam feeds

## Project Structure

```
Sound_Vision/
├── Testing stuff/           # Main scripts and models
│   ├── detect_with_yolov8.py    # YOLOv8 detection script
│   ├── detic_webcam.py          # Detic webcam detection
│   ├── stationary image processor.py  # Basic image processing
│   ├── Image Processing.py      # Advanced image processing
│   ├── requirements.txt        # Python dependencies
│   ├── yolov8n.pt             # YOLOv8 nano model
│   ├── yolov8s.pt             # YOLOv8 small model
│   └── Ash_Tree_-_geograph.org.uk_-_590710.jpg  # Test image
├── runs/                     # YOLOv8 output directory
│   └── detect/
├── venv/                     # Python virtual environment
├── LICENSE                   # MIT License
└── README.md                 # This file
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/sudo-abeaver/Sound_Vision.git
cd Sound_Vision
```

### 2. Quick Start (Recommended)
```bash
# One-command setup
./start.sh
```

### 3. Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up Hugging Face token (for CLIP features)
echo "HUGGINGFACE_HUB_TOKEN=your_token_here" > .env
```

### 4. Install Additional Dependencies (Optional)
For CLIP functionality:
```bash
pip install transformers torch
```

For Detic functionality:
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Usage

### YOLOv8 Object Detection

Basic detection on an image:
```bash
python "Testing stuff/detect_with_yolov8.py" --image path/to/image.jpg --output output.jpg
```

Diagnose model classes:
```bash
python "Testing stuff/detect_with_yolov8.py" --image path/to/image.jpg --diagnose
```

CLIP zero-shot classification (requires transformers):
```bash
python "Testing stuff/detect_with_yolov8.py" --image path/to/image.jpg --clip --clip-labels "tree,plant,vegetation"
```

### Detic Webcam Detection

Real-time open-vocabulary detection:
```bash
python "Testing stuff/detic_webcam.py" --config path/to/config.yaml --weights path/to/model.pth --labels "tree,door,car" --show
```

## Model Information

### YOLOv8 Models
- **yolov8n.pt**: Nano model (6.5MB) - Fastest, lower accuracy
- **yolov8s.pt**: Small model (22.6MB) - Balanced speed/accuracy

### Supported Classes
The YOLOv8 models are trained on the COCO dataset and support 80 object classes including:
- Person, vehicles (car, truck, bus, etc.)
- Animals (cat, dog, horse, etc.)
- Common objects (chair, table, laptop, etc.)

**Note**: Trees are not included in the COCO dataset. For tree detection, consider:
- Using CLIP for zero-shot classification
- Training a custom model with tree labels
- Using specialized tree detection models

## Dependencies

### Core Dependencies
- `ultralytics>=8.0.0` - YOLOv8 framework
- `opencv-python` - Computer vision operations
- `Pillow` - Image processing

### Optional Dependencies
- `transformers` - For CLIP models
- `torch` - PyTorch framework
- `detectron2` - For Detic models

## Troubleshooting

### CLIP Authentication Error
If you encounter authentication errors with CLIP models:

**Option 1: Set Hugging Face Token**
```bash
# Get token from: https://huggingface.co/settings/tokens
export HUGGINGFACE_HUB_TOKEN=your_token_here
```

**Option 2: Use Alternative Scripts**
```bash
# Use simple YOLO detection (no CLIP needed)
python scripts/simple_detect.py --image data/Ash_Tree_-_geograph.org.uk_-_590710.jpg

# Try alternative CLIP models
python scripts/detect_with_clip_alternative.py --image data/Ash_Tree_-_geograph.org.uk_-_590710.jpg
```

**Option 3: Skip CLIP Features**
The CLIP functionality is optional. You can use YOLO detection without CLIP by using the `simple_detect.py` script.

### CUDA Support
For GPU acceleration, install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [OpenAI](https://openai.com/) for CLIP models
- [Facebook Research](https://github.com/facebookresearch/detectron2) for Detectron2
- [IDEA-Research](https://github.com/IDEA-Research/Detic) for Detic