# TFLite Model Guide

How to obtain, convert, and optimize models for Raspberry Pi 4.

---

## Overview

The system requires two models:

1. **Scene Classifier** (required) - Determines current environment
2. **Object Detector** (optional) - Identifies specific objects

Both must be **TensorFlow Lite** format with **INT8 quantization** for optimal Pi 4 performance.

---

## Scene Classifier

### Requirements

- **Input**: RGB image (typically 224×224 or 320×320)
- **Output**: Class probabilities (12-20 scene categories)
- **Format**: `.tflite` (INT8 quantized)
- **Target speed**: 5-10 FPS on Pi 4

### Recommended Architectures

| Model | Input Size | Params | Speed | Accuracy |
|-------|------------|--------|-------|----------|
| **MobileNetV2** | 224×224 | 3.5M | ~8 FPS | Good |
| **EfficientNet-Lite0** | 224×224 | 4.7M | ~6 FPS | Better |
| **MobileNetV3-Small** | 224×224 | 2.5M | ~10 FPS | Good |

### Scene Classes (Suggested)

Based on Places365, but reduced to useful categories:

```
forest          # Woods, trees, nature
park            # Open grass, playgrounds
beach           # Sand, coast, waterfront
coast           # Cliff, seaside
urban_street    # City street, downtown
street          # Any street
residential     # Suburbs, houses
indoor_hall     # Corridor, hallway
corridor        # Passage
staircase       # Stairs
museum_gallery  # Museum, art gallery
gallery         # Art space
store           # Shop, retail
plaza           # Public square
parking_lot     # Parking area
```

Save to `labels/scene_labels.txt` (one per line).

---

## Object Detector

### Requirements

- **Input**: RGB image (typically 300×300 or 320×320)
- **Output**: Bounding boxes, classes, scores
- **Format**: `.tflite` (INT8 quantized)
- **Target speed**: 3-8 FPS on Pi 4

### Recommended Architectures

| Model | Input Size | Params | Speed | Accuracy |
|-------|------------|--------|-------|----------|
| **MobileNet-SSD** | 300×300 | 5.5M | ~7 FPS | Good |
| **YOLOv5-Nano** | 320×320 | 1.9M | ~5 FPS | Better |
| **EfficientDet-Lite0** | 320×320 | 3.2M | ~4 FPS | Best |

### Important Classes

Focus on navigation-relevant objects:

```
person, bicycle, car, motorcycle, bus, truck
traffic light, stop sign, bench, dog
stairs, staircase, door, doorway
chair, dining table
```

Standard COCO 80-class models work well. Save labels to `labels/detect_labels.txt`.

---

## Option 1: Pre-trained Models

### TensorFlow Hub

**Scene Classification**:
```bash
# Download MobileNetV2 (Places365)
wget https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4?tf-hub-format=compressed

# (Note: This is an example URL, actual links may vary)
```

**Object Detection**:
```bash
# COCO-SSD (pre-converted TFLite)
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
mv detect.tflite models/detect_int8.tflite
```

### Google Coral Models

Pre-optimized for Edge TPU (also work on CPU):

```bash
# Download EfficientDet-Lite
wget https://github.com/google-coral/test_data/raw/master/efficientdet_lite0_320_ptq.tflite -O models/detect_int8.tflite
```

---

## Option 2: Convert Existing Models

### Convert PyTorch → TFLite

If you have a PyTorch model:

```python
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

# 1. Load your PyTorch model
model = torch.load('scene_model.pth')
model.eval()

# 2. Convert to TorchScript
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model = optimize_for_mobile(traced_model)

# 3. Convert to ONNX
torch.onnx.export(model, example_input, "scene.onnx")

# 4. Convert ONNX → TFLite (use separate tool)
# See: https://github.com/onnx/onnx-tensorflow
```

### Convert Keras → TFLite

If you have a Keras/TF model:

```python
import tensorflow as tf

# 1. Load model
model = tf.keras.models.load_model('scene_model.h5')

# 2. Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset for INT8 quantization
def representative_dataset():
    for _ in range(100):
        # Use real images from your domain
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# 3. Convert
tflite_model = converter.convert()

# 4. Save
with open('models/scene_int8.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## Option 3: Train Your Own

### Scene Classifier (Transfer Learning)

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# 1. Load pre-trained base
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# 2. Add classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(15, activation='softmax')(x)  # 15 scene classes

model = Model(inputs=base_model.input, outputs=predictions)

# 3. Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train on your dataset
# (Load images organized in folders by class)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'datasets/scenes/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

model.fit(train_generator, epochs=10)

# 5. Fine-tune
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_generator, epochs=5)

# 6. Save
model.save('scene_model.h5')

# 7. Convert to TFLite (see above)
```

### Training Data Sources

**Scene Classification**:
- [Places365](http://places2.csail.mit.edu/) - 10M images, 365 categories
- [SUN Database](https://vision.princeton.edu/projects/2010/SUN/) - 130k images
- Your own photos (use smartphone to capture 100+ per class)

**Object Detection**:
- [COCO Dataset](https://cocodataset.org/) - 330k images, 80 classes
- [Open Images](https://storage.googleapis.com/openimages/web/index.html) - 9M images
- Fine-tune on domain-specific data

---

## Quantization (Critical for Pi Performance)

### Why Quantize?

- **Speed**: 2-3× faster inference
- **Size**: 4× smaller models
- **Power**: Lower energy consumption

### INT8 Post-Training Quantization

```python
import tensorflow as tf
import numpy as np

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset (100-500 samples)
def representative_dataset():
    dataset = load_your_calibration_data()  # Real images!
    for image in dataset:
        image = tf.image.resize(image, (224, 224))
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, 0)
        yield [image]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.float32
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
with open('model_int8.tflite', 'wb') as f:
    f.write(tflite_model)
```

**Important**: Use real images from your domain for calibration!

---

## Testing & Benchmarking

### Test Model

```python
import numpy as np
import tflite_runtime.interpreter as tflite

# Load model
interpreter = tflite.Interpreter(model_path='models/scene_int8.tflite')
interpreter.allocate_tensors()

# Get I/O details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Input dtype: {input_details[0]['dtype']}")

# Run inference
dummy_input = np.random.rand(*input_details[0]['shape']).astype(input_details[0]['dtype'])
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
print(f"Output shape: {output.shape}")
print(f"Output: {output}")
```

### Benchmark Speed

```bash
# Use provided script
python3 scripts/benchmark_models.py

# Expected output:
# Scene Classifier:   120.50 ms (8.3 FPS)
# Object Detector:    180.25 ms (5.5 FPS)
```

**Target Performance** (Pi 4):
- Scene: <150ms (>6 FPS)
- Detection: <200ms (>5 FPS)

If slower, try:
- Lower input resolution
- Smaller architecture (MobileNet vs EfficientNet)
- Verify INT8 quantization

---

## Model Optimization Tips

### 1. Reduce Input Size

```python
# 224×224 → 192×192 saves ~25% compute
# May reduce accuracy slightly
```

### 2. Prune Weights

```python
import tensorflow_model_optimization as tfmot

model = tfmot.sparsity.keras.prune_low_magnitude(model)
# Removes least important weights
```

### 3. Knowledge Distillation

Train smaller "student" model to mimic larger "teacher":

```python
# Teacher: EfficientNet-B3 (accurate but slow)
# Student: MobileNetV2 (fast but less accurate)
# Student learns from teacher's outputs
```

### 4. Use Coral Edge TPU (Optional)

For even faster inference (~20× speedup):

```bash
# Compile for Edge TPU
edgetpu_compiler models/scene_int8.tflite

# Requires Coral USB Accelerator ($60)
```

---

## Debugging

### Model won't load

```bash
# Check file integrity
file models/scene_int8.tflite
# Should output: "TensorFlow Lite model"

# Check model structure
python3 -c "import tflite_runtime.interpreter as tflite; i = tflite.Interpreter('models/scene_int8.tflite'); print('OK')"
```

### Low accuracy

- **Wrong input preprocessing**: Check normalization (0-1 vs 0-255)
- **Poor calibration dataset**: Use diverse, real images
- **Over-quantization**: Try float16 instead of INT8

### Slow inference

- **Check quantization**: INT8 models should be 4× smaller than float32
- **Lower resolution**: 192×192 instead of 320×320
- **Simpler architecture**: MobileNetV3-Small instead of V2

---

## Resources

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)
- [Coral Edge TPU](https://coral.ai)
- [YOLOv5 Export Guide](https://github.com/ultralytics/yolov5/issues/251)
- [Places365 Dataset](http://places2.csail.mit.edu/)

---

**Next**: [Training Custom Models](TRAINING.md)

