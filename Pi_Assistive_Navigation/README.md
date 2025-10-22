# Assistive Navigation System for Raspberry Pi 4

<div align="center">

**A fully autonomous sensory navigation aid combining ultrasonic collision avoidance, AI-powered scene awareness, and adaptive ambient soundscapes.**

</div>

---

## 🎯 Overview

This system runs continuously on a Raspberry Pi 4 mounted on a belt, providing real-time assistance through:

- **🚨 Collision Avoidance**: Ultrasonic sensor with distance-based audio beeps (rate + pitch increase as obstacles get closer)
- **👁️ Visual Awareness**: Tiny TFLite models for scene classification and object detection
- **🎵 Ambient Audio**: Automatic soundscape selection based on environment (forest, city, beach, museum/classical)
- **🔊 Spoken Cues**: Minimal, rate-limited voice announcements for important objects (stairs, doorways, etc.)

**Zero user input required** – the system self-manages everything from collision warnings to music selection.

---

## 📋 Hardware Requirements

### Required Components

| Component | Specification | Notes |
|-----------|--------------|-------|
| **Raspberry Pi 4** | 4GB+ RAM recommended | 2GB minimum |
| **Ultrasonic Sensor** | HC-SR04 | Front-facing, belt-mounted |
| **USB Webcam** | 720p or better | Collar-mounted or chest-mounted |
| **Audio Output** | 3.5mm headphones or Bluetooth | Speakers or bone-conduction headphones |
| **Power Bank** | 10,000+ mAh | Must support Pi 4 power draw |
| **MicroSD Card** | 32GB+ Class 10 | For OS + models + soundscapes |

### Optional Enhancements

- **USB Audio Interface**: Better audio quality than built-in 3.5mm jack
- **Bone Conduction Headphones**: Keeps ears free for ambient sound
- **GPIO Voltage Divider**: Required for HC-SR04 echo pin (5V → 3.3V)

---

## ⚡ Quick Start

### 1. Raspberry Pi Setup

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y python3-pip python3-dev git
sudo apt-get install -y libatlas-base-dev libhdf5-dev
sudo apt-get install -y python3-opencv
sudo apt-get install -y espeak  # Text-to-speech
sudo apt-get install -y portaudio19-dev
sudo apt-get install -y libsdl2-dev libsdl2-mixer-dev

# Clone or copy this project
cd ~
git clone <your-repo-url> assistive
cd assistive

# Install Python dependencies
pip3 install -r requirements.txt

# Special: Install TFLite Runtime (lighter than full TensorFlow)
pip3 install --index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

### 2. Hardware Wiring

#### HC-SR04 Ultrasonic Sensor

**⚠️ IMPORTANT**: The Echo pin outputs 5V, but Pi GPIO pins are 3.3V max. Use a voltage divider!

```
HC-SR04 → Raspberry Pi
  VCC   → Pin 2 (5V)
  GND   → Pin 6 (GND)
  TRIG  → Pin 16 (GPIO 23)
  ECHO  → Pin 18 (GPIO 24) ← USE VOLTAGE DIVIDER!

Voltage Divider for Echo:
  Echo pin → 1kΩ resistor → GPIO 24
                         ↓
                      2kΩ resistor → GND
```

#### USB Webcam

Simply plug into any USB 3.0 port (blue port for best performance).

#### Audio

- **3.5mm**: Plug into headphone jack
- **Bluetooth**: See Bluetooth setup section below

### 3. Download Models

You need to provide your own TFLite models (INT8 quantized for best Pi 4 performance):

```bash
mkdir -p models
cd models

# Example: Download a MobileNetV2 scene classifier
# You can train your own or convert from existing models
# Place your models here:
#   - scene_int8.tflite  (scene classification)
#   - detect_int8.tflite (object detection, optional)

# For training/converting models, see docs/MODEL_GUIDE.md
```

**Recommended Model Sources:**
- **Scene Classifier**: MobileNetV2 or EfficientNet-Lite0 trained on Places365 subset
- **Object Detector**: YOLOv5-nano, MobileNet-SSD, or EfficientDet-Lite0

See [Model Guide](docs/MODEL_GUIDE.md) for conversion instructions.

### 4. Add Soundscapes

Populate the `soundscapes/` directory with your audio files:

```bash
# Structure (place .wav or .mp3 files in each folder):
soundscapes/
  forest/
    forest_ambience_1.wav
    forest_birds.wav
  park/
    park_ambience.wav
  beach/
    ocean_waves.mp3
  city/
    city_street.wav
    traffic_ambient.mp3
  residential/
    suburb_quiet.wav
  indoor/
    indoor_ambient.wav
  museum/
    classical_1.mp3
    classical_2.mp3
  plaza/
    plaza_ambience.wav
  parking/
    parking_lot.wav
```

**Sources for Free Soundscapes:**
- [Freesound.org](https://freesound.org)
- [Free Music Archive](https://freemusicarchive.org)
- [BBC Sound Effects](https://sound-effects.bbcrewind.co.uk)

### 5. Configure System

Edit `config.yaml` to tune parameters:

```yaml
# Adjust ultrasonic zones for your walking speed
ultrasonic:
  zones:
    - min: 1.2
      max: 2.0
      beep_interval: 0.65
      pitch: 440

# Adjust audio volumes
audio:
  soundscape_volume: 0.7
  beep_volume: 0.6
  speech_rate_limit: 5.0  # Seconds between announcements
```

### 6. Test Run

```bash
# Make executable
chmod +x main.py

# Test run (without hardware)
python3 main.py

# On Pi with hardware
python3 main.py
```

Expected output:
```
============================================================
ASSISTIVE NAVIGATION SYSTEM STARTING
============================================================
Mode: Fully Autonomous
- Collision avoidance: ACTIVE
- Scene awareness: ACTIVE
- Ambient audio: ACTIVE
============================================================

Distance: 2.45m | Zone: 0 | Scene: indoor_hall | Soundscape: indoor
```

Press `Ctrl+C` to stop.

---

## 🎛️ System Behavior

### Collision Avoidance (Always-On)

The ultrasonic sensor runs at **50Hz** with median filtering:

| Distance | Behavior | Beep Rate | Pitch |
|----------|----------|-----------|-------|
| > 2.0m | Silent | — | — |
| 2.0 – 1.2m | Low alert | 650ms intervals | 440 Hz (A4) |
| 1.2 – 0.7m | Medium alert | 320ms intervals | 660 Hz (E5) |
| 0.7 – 0.45m | High alert | 150ms intervals | 880 Hz (A5) |
| < 0.45m | **DANGER** | Continuous tone | 1100 Hz |

When in danger zone, also speaks **"Stop"** immediately.

### Visual Awareness (Low-Rate)

Scene classification runs every **~500ms**:
- Detects current environment (forest, city, museum, etc.)
- Requires 8 consistent readings before changing soundscape
- Triggers ambient music crossfade (3.5s smooth transition)

Object detection runs every **~800ms**:
- Announces important objects: stairs, people, vehicles
- Rate-limited: minimum 5 seconds between announcements
- Provides direction: "left", "ahead", or "right"

### Audio Engine

**Three simultaneous layers:**
1. **Soundscape Bed**: Loops ambient audio at 70% volume
2. **Collision Beeps**: Short sine-wave beeps, volume adjusts automatically
3. **Voice Cues**: Text-to-speech for important events

**Auto-ducking**: Music drops to 30% when speaking or beeping, then restores.

### Special: Museum/Art Mode

If scene is detected as `museum_gallery` or `indoor_hall` **AND** a frame-like rectangle is found in the image (simple contour detection), the system switches to classical music.

---

## 🔧 Customization

### Adjust Distance Zones

Edit `config.yaml`:

```yaml
ultrasonic:
  zones:
    - min: 1.5  # Change warning distance
      max: 2.5
      beep_interval: 0.5  # Faster beeps
      pitch: 500  # Higher pitch
```

### Add New Scenes

1. Add scene class to `labels/scene_labels.txt`
2. Create soundscape folder in `soundscapes/<scene_name>/`
3. Map in `config.yaml`:

```yaml
scene_mapping:
  coffee_shop: "indoor"
  library: "museum"  # Use classical
```

### Change Speech Rate

```yaml
audio:
  speech_rate_limit: 3.0  # Speak more frequently (every 3s)
```

### Disable Object Detection

Set in `config.yaml`:

```yaml
vision:
  detect_model: ""  # Empty = disabled
```

---

## 🚀 Auto-Start on Boot

To run automatically when Pi boots:

### Method 1: systemd Service (Recommended)

```bash
sudo nano /etc/systemd/system/assistive-nav.service
```

Add:
```ini
[Unit]
Description=Assistive Navigation System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/assistive
ExecStart=/usr/bin/python3 /home/pi/assistive/main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable assistive-nav.service
sudo systemctl start assistive-nav.service
sudo systemctl status assistive-nav.service
```

### Method 2: rc.local (Simple)

```bash
sudo nano /etc/rc.local
```

Add before `exit 0`:
```bash
cd /home/pi/assistive && python3 main.py &
```

---

## 🎧 Bluetooth Audio Setup

For wireless headphones:

```bash
# Install Bluetooth audio
sudo apt-get install -y pulseaudio pulseaudio-module-bluetooth
sudo usermod -a -G bluetooth pi

# Pair headphones
bluetoothctl
> power on
> agent on
> scan on
> pair <MAC_ADDRESS>
> trust <MAC_ADDRESS>
> connect <MAC_ADDRESS>
> quit

# Set as default output
pacmd set-default-sink <bluetooth_sink>
```

---

## 📊 Performance Tips

### Optimize for Speed

1. **Lower camera resolution** in `config.yaml`:
   ```yaml
   vision:
     camera_width: 256
     camera_height: 192
   ```

2. **Increase inference intervals**:
   ```yaml
   vision:
     scene_inference_interval: 0.8  # Slower = less CPU
     detect_inference_interval: 1.2
   ```

3. **Disable object detection** if not needed:
   ```yaml
   vision:
     detect_model: ""
   ```

4. **Use INT8 quantized models** (already recommended)

5. **Overclock Pi 4** (careful with cooling):
   ```bash
   sudo nano /boot/config.txt
   # Add:
   over_voltage=6
   arm_freq=2000
   ```

### Monitor Performance

```bash
# CPU temperature
vcgencmd measure_temp

# System load
htop

# Test inference speed
python3 scripts/benchmark_models.py
```

---

## 🛠️ Troubleshooting

### Camera Not Detected

```bash
# Check camera
ls /dev/video*

# Test with OpenCV
python3 -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# Check permissions
sudo usermod -a -G video pi
```

### Ultrasonic Sensor Not Working

- **Check wiring**: Especially voltage divider on Echo pin
- **Test GPIO**:
  ```python
  import RPi.GPIO as GPIO
  GPIO.setmode(GPIO.BCM)
  GPIO.setup(23, GPIO.OUT)
  GPIO.setup(24, GPIO.IN)
  ```
- **Verify voltage**: Echo → 3.3V max (use multimeter)

### No Audio Output

```bash
# Test speaker
speaker-test -t wav -c 2

# Check ALSA config
aplay -l

# Set default device
sudo nano /usr/share/alsa/alsa.conf
# Change defaults.ctl.card and defaults.pcm.card to your device
```

### Model Not Loading

- **Check file exists**: `ls -lh models/scene_int8.tflite`
- **Verify INT8 quantization**: Model should be small (<5MB for MobileNet)
- **Try tensorflow.lite** instead of tflite_runtime if issues persist

### High CPU Usage

- Lower camera FPS: `camera_fps: 10`
- Increase inference intervals: `scene_inference_interval: 1.0`
- Use smaller models: EfficientNet-Lite0 instead of MobileNetV2

---

## 📚 Project Structure

```
Pi_Assistive_Navigation/
├── main.py                 # Main application
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── models/                # TFLite models (you provide)
│   ├── scene_int8.tflite
│   └── detect_int8.tflite
│
├── labels/                # Model labels
│   ├── scene_labels.txt
│   └── detect_labels.txt
│
├── soundscapes/           # Audio files (you provide)
│   ├── forest/
│   ├── park/
│   ├── beach/
│   ├── city/
│   ├── residential/
│   ├── indoor/
│   ├── museum/
│   ├── plaza/
│   └── parking/
│
├── scripts/               # Utility scripts
│   ├── benchmark_models.py
│   ├── test_ultrasonic.py
│   ├── test_camera.py
│   └── convert_model.py
│
└── docs/                  # Additional documentation
    ├── MODEL_GUIDE.md
    ├── WIRING.md
    └── TRAINING.md
```

---

## 🔬 Advanced Usage

### Custom Model Training

See [docs/TRAINING.md](docs/TRAINING.md) for:
- Training scene classifiers on Places365
- Fine-tuning object detectors for specific environments
- Converting models to TFLite INT8

### Multi-Sensor Fusion

Extend with additional sensors:
```python
# In main.py, add to __init__:
from sensors import IMUSensor, GPSSensor

self.imu = IMUSensor()
self.gps = GPSSensor()
```

### Remote Monitoring

Add network logging:
```python
import socket
# Send status to remote server
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(status_json.encode(), ('192.168.1.100', 5000))
```

---

## 📝 Configuration Reference

### Ultrasonic Zones

```yaml
zones:
  - min: <float>          # Minimum distance (meters)
    max: <float>          # Maximum distance (meters)
    beep_interval: <float> # Seconds between beeps (0 = silent)
    pitch: <int>          # Frequency in Hz
    continuous: <bool>    # Optional: continuous tone
```

### Scene Mapping

```yaml
scene_mapping:
  <detected_class>: <soundscape_folder>
```

### Audio Volumes

All volumes are 0.0 – 1.0:
- `soundscape_volume`: Normal background music
- `ducked_volume`: Music during speech/beeps
- `beep_volume`: Collision warning beeps
- `tts_volume`: Speech announcements

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Better art detection (trained model vs. heuristic)
- [ ] GPS integration for location-based soundscapes
- [ ] IMU for stair detection enhancement
- [ ] Multi-language TTS support
- [ ] Battery monitoring and low-power mode
- [ ] Companion mobile app for settings

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- **TensorFlow Lite**: Efficient ML inference
- **OpenCV**: Computer vision processing
- **Pygame**: Audio mixing and playback
- **Raspberry Pi Foundation**: Affordable hardware platform

---

## 📞 Support

Issues? Questions? Create an issue on GitHub or contact the team.

**Project by FSE100 - Making technology accessible to everyone.**

---

## 🎯 Future Roadmap

### Phase 2
- [ ] Depth camera support (Intel RealSense, Oak-D)
- [ ] Stereo spatial audio (direction-aware beeps)
- [ ] Cloud model updates (download new scenes)
- [ ] Activity recognition (walking, running, sitting)

### Phase 3
- [ ] Multi-user profiles (personalized soundscapes)
- [ ] Emergency alert system (fall detection)
- [ ] Integration with smart home devices
- [ ] Voice commands (optional override mode)

---

<div align="center">

**Built with ❤️ for accessibility**

[Documentation](docs/) | [Issues](https://github.com/your-repo/issues) | [Discussions](https://github.com/your-repo/discussions)

</div>

