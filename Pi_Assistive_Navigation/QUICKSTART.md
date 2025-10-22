# 🚀 Quick Start Guide

Get your Assistive Navigation System running in 15 minutes!

## Prerequisites

- Raspberry Pi 4 (2GB+ RAM)
- MicroSD card with Raspberry Pi OS installed
- HC-SR04 ultrasonic sensor + wiring
- USB webcam
- Headphones or speaker
- Internet connection (for setup)

---

## Step 1: Hardware Setup ⚡

### Ultrasonic Sensor Wiring

**⚠️ CRITICAL: Use voltage divider on Echo pin!**

```
HC-SR04          Raspberry Pi
  VCC    →       Pin 2 (5V)
  GND    →       Pin 6 (GND)
  TRIG   →       Pin 16 (GPIO 23)
  ECHO   →       Pin 18 (GPIO 24) ← via voltage divider!

Voltage Divider:
  Echo pin → 1kΩ → GPIO 24
                ↓
             2kΩ → GND
```

### Connect Camera & Audio

1. Plug USB webcam into any USB port
2. Connect headphones to 3.5mm jack (or use Bluetooth)

---

## Step 2: Software Setup 💻

```bash
# SSH into your Pi or open terminal

# Download project
cd ~
git clone <your-repo-url> assistive
cd assistive

# Run automated setup
bash scripts/setup_pi.sh

# This installs all dependencies - takes 5-10 minutes
```

---

## Step 3: Add Models & Audio 🎵

### Get TFLite Models

You need two models (quantized INT8 for Pi 4):

1. **Scene Classifier**: `models/scene_int8.tflite`
   - MobileNetV2 or EfficientNet-Lite0
   - Trained on Places365 or similar
   
2. **Object Detector**: `models/detect_int8.tflite` (optional)
   - YOLOv5-nano or MobileNet-SSD
   - Standard COCO classes

**Option A**: Use pre-trained models from TensorFlow Hub  
**Option B**: Train your own (see [docs/TRAINING.md](docs/TRAINING.md))

### Add Soundscapes

Place audio files (MP3 or WAV) in folders:

```bash
soundscapes/
  forest/       # Forest ambience, birds
  park/         # Park sounds, light activity
  beach/        # Ocean waves, seagulls
  city/         # Urban traffic, city buzz
  residential/  # Quiet neighborhood
  indoor/       # Indoor ambient
  museum/       # Classical music
  plaza/        # Public square sounds
  parking/      # Parking lot ambient
```

**Free sources**:
- [Freesound.org](https://freesound.org)
- [BBC Sound Effects](https://sound-effects.bbcrewind.co.uk)
- YouTube (with proper licensing)

---

## Step 4: Test Hardware 🧪

### Test Ultrasonic Sensor

```bash
python3 scripts/test_ultrasonic.py
```

Expected output:
```
Distance:  123.4 cm  [████████████                    ]
```

Move your hand in front - distance should change.

### Test Camera

```bash
python3 scripts/test_camera.py
```

A window should open showing live video. Press 'q' to quit.

### Benchmark Models (optional)

```bash
python3 scripts/benchmark_models.py
```

This shows inference speed. Target: >5 FPS for scene, >3 FPS for detection.

---

## Step 5: Configure ⚙️

Edit `config.yaml` if needed:

```yaml
# Adjust collision distances
ultrasonic:
  zones:
    - min: 1.2
      max: 2.0
      beep_interval: 0.65  # Seconds between beeps

# Adjust volumes
audio:
  soundscape_volume: 0.7  # Background music
  beep_volume: 0.6       # Warning beeps
```

---

## Step 6: Run! 🎉

```bash
python3 main.py
```

You should see:

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

**Test it:**
1. Move your hand near the sensor → should beep
2. Walk around → soundscape should play
3. Point camera at different scenes → soundscape should change

Press `Ctrl+C` to stop.

---

## Step 7: Auto-Start (Optional) 🔄

Make it run on boot:

```bash
sudo nano /etc/systemd/system/assistive-nav.service
```

Paste:

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

[Install]
WantedBy=multi-user.target
```

Enable:

```bash
sudo systemctl enable assistive-nav.service
sudo systemctl start assistive-nav.service
```

Check status:

```bash
sudo systemctl status assistive-nav.service
```

---

## Troubleshooting 🔧

### No beeps?

- Check wiring (especially voltage divider!)
- Run: `python3 scripts/test_ultrasonic.py`
- Verify GPIO permissions: `groups` should show `gpio`

### No camera?

- Run: `ls /dev/video*` (should show `/dev/video0`)
- Try: `python3 scripts/test_camera.py`
- Check permissions: `groups` should show `video`

### No audio?

- Test speaker: `speaker-test -t wav`
- Check volume: `alsamixer`
- Verify pygame: `python3 -c "import pygame; print('OK')"`

### Models not loading?

- Verify files exist: `ls -lh models/`
- Check format: must be `.tflite` (not `.pb` or `.h5`)
- Run benchmark: `python3 scripts/benchmark_models.py`

### High CPU / slow?

Edit `config.yaml`:

```yaml
vision:
  camera_width: 256          # Lower resolution
  camera_height: 192
  scene_inference_interval: 1.0  # Slower inference
```

---

## Next Steps 📚

- **Tune zones**: Adjust `config.yaml` for your walking speed
- **Add more scenes**: Train custom scene classifier
- **Mount hardware**: Design belt + collar mount
- **Extend features**: Add GPS, IMU, or other sensors

Full documentation: [README.md](README.md)

---

## Support 💬

Having issues? Check:

1. [README.md](README.md) - Full documentation
2. [GitHub Issues](https://github.com/your-repo/issues)
3. Community forums

---

**You're all set! Enjoy your assistive navigation system.** 🎉

