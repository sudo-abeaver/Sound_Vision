# Hardware Wiring Guide

Detailed instructions for connecting sensors and peripherals to Raspberry Pi 4.

---

## ⚠️ Safety First

- **Power off** Pi before making connections
- **Never apply >3.3V** to GPIO pins (use voltage dividers!)
- **Double-check polarity** on power connections
- Use **anti-static precautions** when handling Pi

---

## HC-SR04 Ultrasonic Sensor

### Pinout

| HC-SR04 Pin | Function | Raspberry Pi Pin |
|-------------|----------|------------------|
| VCC | Power (5V) | Pin 2 (5V) |
| TRIG | Trigger | Pin 16 (GPIO 23) |
| ECHO | Echo | Pin 18 (GPIO 24) **via voltage divider** |
| GND | Ground | Pin 6 (GND) |

### Critical: Voltage Divider Circuit

The Echo pin outputs **5V**, but Pi GPIO pins are rated for **3.3V max**. You MUST use a voltage divider!

```
    HC-SR04 Echo Pin (5V)
            |
           [R1 = 1kΩ]
            |
            +--------> to GPIO 24 (Pin 18)
            |
           [R2 = 2kΩ]
            |
           GND

Output voltage = 5V × (R2 / (R1 + R2))
               = 5V × (2kΩ / 3kΩ)
               = 3.33V ✓
```

### Wiring Diagram

```
Raspberry Pi 4               HC-SR04
  (Top View)              (Front View)

    ┌─────────┐              ┌──┐
    │ USB USB │              │  │
    │ USB USB │              │  │
    ├─────────┤              └──┘
  2 ●5V  5V  ●  4          VCC TRIG ECHO GND
  6 ●GND      ●  8           │   │    │   │
    │         │              │   │    │   └─> GND (Pin 6)
    │         │              │   │    │
 16 ●GPIO23   │              │   │    └────> Voltage divider -> GPIO24
 18 ●GPIO24   │              │   │
    │         │              │   └─────────> GPIO23 (Pin 16)
    └─────────┘              └─────────────> 5V (Pin 2)
```

### Physical Mounting

**Belt Mount (Front-Facing)**:
- Mount sensor pointing forward
- Height: waist level (approx. 1m from ground)
- Angle: slightly downward (10-15°) to catch low obstacles
- Use velcro or 3D-printed bracket
- Keep wires secure (zip ties)

---

## USB Webcam

### Connection

Simply plug into any **USB 3.0 port** (blue port) for best performance.

### Mounting

**Collar/Chest Mount**:
- Attach to shirt collar or chest strap
- Point forward at eye level
- Adjustable angle (tilt down 10-15°)
- Use clip or lanyard
- Keep USB cable slack for movement

### Recommended Specs

- **Resolution**: 720p (1280×720) minimum
- **Frame rate**: 30 FPS
- **Low-light**: Good performance for indoor scenes
- **Wide angle**: 70-90° field of view

### Compatible Models

- Logitech C270 / C310 / C920
- Microsoft LifeCam
- Generic USB webcams (most work)

---

## Audio Output

### Option 1: 3.5mm Headphones

**Pros**: Simple, no pairing needed  
**Cons**: Cable, lower quality

Connect to 3.5mm jack on Pi.

**Enable audio output**:
```bash
sudo raspi-config
# System Options -> Audio -> Headphones
```

### Option 2: Bluetooth Headphones

**Pros**: Wireless, better for mobility  
**Cons**: Battery, latency, pairing

**Setup**:
```bash
# Install Bluetooth audio
sudo apt-get install pulseaudio pulseaudio-module-bluetooth

# Pair headphones
bluetoothctl
> power on
> agent on
> scan on
# (note MAC address of your headphones)
> pair XX:XX:XX:XX:XX:XX
> trust XX:XX:XX:XX:XX:XX
> connect XX:XX:XX:XX:XX:XX
> quit
```

**Auto-connect on boot**:
```bash
# Edit bluetooth config
sudo nano /etc/bluetooth/main.conf
# Add/uncomment:
AutoEnable=true
```

### Option 3: Bone Conduction Headphones

**Pros**: Ears free for ambient sound, safest for navigation  
**Cons**: More expensive

Recommended: AfterShokz/Shokz, Vidonn

Connect via Bluetooth (see above).

---

## Power Supply

### Requirements

- **Output**: 5V @ 3A minimum (15W)
- **Connector**: USB-C
- **Capacity**: 10,000+ mAh for 6-8 hours runtime

### Recommended Power Banks

- Anker PowerCore 10000
- RAVPower 20000mAh
- Goal Zero 22000mAh (ruggedized)

### Power Consumption

Typical draw:
- Pi 4: 3-5W
- Webcam: 1-2W
- Ultrasonic: <0.5W
- Total: **~5-7W** (Pi can spike to 8W under load)

**Runtime estimate**: 10,000mAh bank ≈ 6-8 hours

### Belt Mount

- Use phone armband or belt pouch
- Short USB-C cable to Pi
- Velcro or carabiner attachment

---

## Complete System Wiring

```
┌──────────────────────────────────────────────┐
│            RASPBERRY PI 4                     │
│                                               │
│  ┌────────┐                                  │
│  │USB USB │ ← Camera (USB 3.0)               │
│  │USB USB │ ← (open)                         │
│  └────────┘                                   │
│                                               │
│  ○ ○ ○ ○ ... GPIO Pins                       │
│  ○ ○ ○ ○                                     │
│    ↑ ↑                                        │
│    │ └──────────→ Ultrasonic TRIG            │
│    └────────────→ Ultrasonic ECHO (via div)  │
│                                               │
│  ┌─────────┐                                 │
│  │ USBC ○──┼────→ Power bank                 │
│  └─────────┘                                  │
│                                               │
│  ●─────────────→ 3.5mm headphones            │
│                                               │
└───────────────────────────────────────────────┘

        Ultrasonic Sensor
             ┌───┐
             │▓▓▓│ ← Mounted on front of belt
             └───┘

        Camera
           ┌─┐
           │●│ ← Mounted on chest/collar
           └─┘
```

---

## Troubleshooting

### Ultrasonic gives random readings

- **Loose connection**: Check wiring
- **Missing voltage divider**: Echo pin will damage GPIO!
- **Interference**: Move away from other sensors
- **Angle**: Ensure sensor points forward (not tilted up/down too much)

### Camera not detected

```bash
# Check if detected
ls /dev/video*

# Check permissions
groups
# Should include 'video'

# Add to group if missing
sudo usermod -a -G video $USER
```

### No audio output

```bash
# Test audio
speaker-test -t wav

# Check device
aplay -l

# Select correct output
sudo raspi-config
# System Options -> Audio -> Select device
```

### GPIO permission denied

```bash
# Add user to gpio group
sudo usermod -a -G gpio $USER

# Reboot
sudo reboot
```

---

## Advanced: Custom GPIO Pins

To use different GPIO pins, edit `config.yaml`:

```yaml
ultrasonic:
  trig_pin: 23  # Change to your pin (BCM numbering)
  echo_pin: 24  # Change to your pin (BCM numbering)
```

**Available GPIO pins** (BCM):
- Safe to use: 2, 3, 4, 17, 27, 22, 10, 9, 11, 5, 6, 13, 19, 26, 14, 15, 18, 23, 24, 25, 8, 7
- Avoid: 0, 1 (reserved for I2C)

---

## Bill of Materials

| Item | Qty | Est. Cost |
|------|-----|-----------|
| Raspberry Pi 4 (4GB) | 1 | $55 |
| HC-SR04 Ultrasonic | 1 | $3 |
| USB Webcam | 1 | $20-50 |
| Resistors (1kΩ, 2kΩ) | 2 | $1 |
| MicroSD Card (32GB) | 1 | $10 |
| USB-C Power Bank | 1 | $25 |
| Jumper Wires | 10 | $5 |
| Headphones | 1 | $20-100 |
| Mounting Hardware | - | $10 |
| **Total** | | **$150-260** |

---

**Next**: [Software Setup](../QUICKSTART.md)

