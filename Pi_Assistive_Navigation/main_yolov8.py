#!/usr/bin/env python3
"""
Assistive Navigation System for Raspberry Pi 4 - YOLOv8 Version
-----------------------------------------------------------------
Uses Ultralytics YOLOv8 + HuggingFace CLIP for optimal performance

Features:
- YOLOv8-nano for object detection (optimized for Pi)
- CLIP for zero-shot scene classification
- Ultrasonic collision avoidance
- Adaptive soundscapes
- Zero user input required

Author: FSE100 Project
License: MIT
"""

import time
import threading
import numpy as np
import cv2
import yaml
import random
import os
import subprocess
from pathlib import Path
from collections import deque, Counter
from dataclasses import dataclass
from typing import Optional, Tuple, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import RPi.GPIO as GPIO
    RASPBERRY_PI = True
except ImportError:
    print("WARNING: RPi.GPIO not found. Running in simulation mode.")
    RASPBERRY_PI = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("WARNING: ultralytics not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    print("WARNING: CLIP not available. Install with: pip install transformers torch")
    CLIP_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    print("WARNING: pygame not available for audio")
    PYGAME_AVAILABLE = False


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    default_config = {
        'ultrasonic': {
            'trig_pin': 23,
            'echo_pin': 24,
            'zones': [
                {'min': 2.0, 'max': 999, 'beep_interval': 0, 'pitch': 0},
                {'min': 1.2, 'max': 2.0, 'beep_interval': 0.65, 'pitch': 440},
                {'min': 0.7, 'max': 1.2, 'beep_interval': 0.32, 'pitch': 660},
                {'min': 0.45, 'max': 0.7, 'beep_interval': 0.15, 'pitch': 880},
                {'min': 0.0, 'max': 0.45, 'beep_interval': 0, 'pitch': 1100, 'continuous': True}
            ],
            'hysteresis': 0.1,
            'median_window': 5
        },
        'vision': {
            'yolo_model': 'models/yolov8n.pt',  # YOLOv8-nano for speed
            'yolo_conf': 0.5,
            'yolo_imgsz': 320,
            'clip_model': 'openai/clip-vit-base-patch32',
            'scene_labels': [
                'forest', 'park', 'beach', 'urban street', 'residential area',
                'indoor hallway', 'corridor', 'staircase', 'museum gallery',
                'store', 'plaza', 'parking lot'
            ],
            'scene_inference_interval': 0.8,
            'detect_inference_interval': 1.0,
            'scene_stability_window': 6,
            'camera_width': 640,
            'camera_height': 480,
            'camera_fps': 15
        },
        'audio': {
            'soundscape_volume': 0.7,
            'ducked_volume': 0.3,
            'beep_volume': 0.6,
            'tts_volume': 0.8,
            'crossfade_duration': 3.5,
            'speech_rate_limit': 5.0,
            'beep_duration': 0.08
        },
        'scene_mapping': {
            'forest': 'forest',
            'park': 'park',
            'beach': 'beach',
            'urban street': 'city',
            'residential area': 'residential',
            'indoor hallway': 'indoor',
            'corridor': 'indoor',
            'staircase': 'indoor',
            'museum gallery': 'museum',
            'store': 'plaza',
            'plaza': 'plaza',
            'parking lot': 'parking'
        },
        'priority_objects': ['person', 'bicycle', 'car', 'bus', 'truck', 'dog']
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            for key, value in user_config.items():
                if key in default_config and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
    
    return default_config


# ============================================================================
# Ultrasonic Sensor Module (Same as before)
# ============================================================================

class UltrasonicSensor:
    """HC-SR04 ultrasonic sensor manager with median filtering"""
    
    def __init__(self, trig_pin: int, echo_pin: int, median_window: int = 5):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.median_window = median_window
        self.readings = deque(maxlen=median_window)
        
        if RASPBERRY_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.trig_pin, GPIO.OUT)
            GPIO.setup(self.echo_pin, GPIO.IN)
            GPIO.output(self.trig_pin, False)
            time.sleep(0.1)
    
    def get_distance(self) -> Optional[float]:
        """Get distance in meters (median filtered)"""
        if not RASPBERRY_PI:
            return random.uniform(0.3, 3.0)
        
        try:
            GPIO.output(self.trig_pin, True)
            time.sleep(0.00001)
            GPIO.output(self.trig_pin, False)
            
            timeout = time.time() + 0.1
            pulse_start = time.time()
            while GPIO.input(self.echo_pin) == 0:
                pulse_start = time.time()
                if pulse_start > timeout:
                    return None
            
            pulse_end = time.time()
            while GPIO.input(self.echo_pin) == 1:
                pulse_end = time.time()
                if pulse_end > timeout:
                    return None
            
            pulse_duration = pulse_end - pulse_start
            distance = pulse_duration * 17150 / 100.0
            
            if 0.02 <= distance <= 4.0:
                self.readings.append(distance)
                if len(self.readings) >= 3:
                    return float(np.median(list(self.readings)))
            
            return None
            
        except Exception as e:
            print(f"Ultrasonic error: {e}")
            return None
    
    def cleanup(self):
        if RASPBERRY_PI:
            GPIO.cleanup()


# ============================================================================
# Vision Module (YOLOv8 + CLIP)
# ============================================================================

class VisionModuleYOLO:
    """YOLOv8 for object detection + CLIP for scene classification"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Initialize YOLOv8
        self.yolo_model = None
        if YOLO_AVAILABLE:
            yolo_path = config['vision']['yolo_model']
            if os.path.exists(yolo_path):
                print(f"Loading YOLOv8 from {yolo_path}...")
                self.yolo_model = YOLO(yolo_path)
            else:
                print(f"YOLOv8 model not found at {yolo_path}, downloading yolov8n.pt...")
                self.yolo_model = YOLO('yolov8n.pt')  # Auto-downloads
        
        # Initialize CLIP
        self.clip_model = None
        self.clip_processor = None
        if CLIP_AVAILABLE:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Loading CLIP model on {device}...")
            try:
                self.clip_processor = CLIPProcessor.from_pretrained(
                    config['vision']['clip_model']
                )
                self.clip_model = CLIPModel.from_pretrained(
                    config['vision']['clip_model']
                ).to(device)
                self.device = device
            except Exception as e:
                print(f"CLIP loading error: {e}")
        
        # Scene labels for CLIP
        self.scene_labels = config['vision']['scene_labels']
        self.scene_votes = deque(maxlen=config['vision']['scene_stability_window'])
    
    def classify_scene(self, frame: np.ndarray) -> Optional[str]:
        """Classify scene using CLIP zero-shot"""
        if not CLIP_AVAILABLE or self.clip_model is None:
            return None
        
        try:
            # Convert to PIL
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Prepare inputs
            text_inputs = self.clip_processor(
                text=self.scene_labels,
                return_tensors='pt',
                padding=True
            )
            image_inputs = self.clip_processor(
                images=image,
                return_tensors='pt'
            )
            
            # Move to device
            for k, v in image_inputs.items():
                image_inputs[k] = v.to(self.device)
            for k, v in text_inputs.items():
                text_inputs[k] = v.to(self.device)
            
            # Inference
            with torch.no_grad():
                image_emb = self.clip_model.get_image_features(**image_inputs)
                text_emb = self.clip_model.get_text_features(**text_inputs)
                
                # Normalize
                image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
                text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
                
                # Scores
                scores = (100.0 * image_emb @ text_emb.T).softmax(dim=-1)
                scores = scores[0].cpu().tolist()
            
            # Get top prediction
            max_idx = scores.index(max(scores))
            confidence = scores[max_idx]
            
            if confidence > 0.15:  # Low threshold for CLIP
                scene = self.scene_labels[max_idx]
                self.scene_votes.append(scene)
                
                # Return stable scene (majority vote)
                if len(self.scene_votes) >= self.config['vision']['scene_stability_window']:
                    most_common = Counter(self.scene_votes).most_common(1)[0]
                    if most_common[1] >= self.config['vision']['scene_stability_window'] // 2:
                        return most_common[0]
            
            return None
            
        except Exception as e:
            print(f"Scene classification error: {e}")
            return None
    
    def detect_objects(self, frame: np.ndarray) -> List[dict]:
        """Detect objects using YOLOv8"""
        if not YOLO_AVAILABLE or self.yolo_model is None:
            return []
        
        try:
            # Run inference
            results = self.yolo_model.predict(
                source=frame,
                conf=self.config['vision']['yolo_conf'],
                imgsz=self.config['vision']['yolo_imgsz'],
                verbose=False
            )
            
            detections = []
            if results and len(results) > 0:
                r = results[0]
                
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        name = r.names[cls] if cls in r.names else str(cls)
                        
                        # Normalize bbox coordinates
                        h, w = frame.shape[:2]
                        x1, y1, x2, y2 = xyxy
                        center_x = (x1 + x2) / 2 / w
                        
                        detections.append({
                            'class': name,
                            'confidence': conf,
                            'bbox': [y1/h, x1/w, y2/h, x2/w],
                            'center_x': center_x
                        })
            
            return detections
            
        except Exception as e:
            print(f"Object detection error: {e}")
            return []


# ============================================================================
# Audio Engine (Same as before)
# ============================================================================

class AudioEngine:
    """Manages ambient soundscapes, collision beeps, and speech"""
    
    def __init__(self, config: dict):
        self.config = config
        self.soundscape_base = Path("soundscapes")
        self.current_soundscape = None
        self.next_soundscape = None
        self.crossfade_progress = 1.0
        
        if PYGAME_AVAILABLE:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(8)
        
        self.beep_channel = None
        self.voice_channel = None
        self.music_channel_current = None
        self.music_channel_next = None
        
        if PYGAME_AVAILABLE:
            self.beep_channel = pygame.mixer.Channel(0)
            self.voice_channel = pygame.mixer.Channel(1)
            self.music_channel_current = pygame.mixer.Channel(2)
            self.music_channel_next = pygame.mixer.Channel(3)
        
        self.last_speech_time = 0
        self.ducking = False
        
    def load_soundscape(self, scene_folder: str):
        """Load and start playing soundscape for scene"""
        folder_path = self.soundscape_base / scene_folder
        if not folder_path.exists():
            print(f"🎵 Soundscape folder not found: {folder_path}")
            print(f"   Please add audio files (.mp3 or .wav) to: {folder_path}")
            return
        
        audio_files = list(folder_path.glob("*.wav")) + list(folder_path.glob("*.mp3"))
        if not audio_files:
            print(f"🎵 No audio files in {folder_path}")
            print(f"   Please add audio files (.mp3 or .wav) to: {folder_path}")
            return
        
        new_file = random.choice(audio_files)
        print(f"🎵 Attempting to play: {new_file}")
        
        if not PYGAME_AVAILABLE:
            print(f"🎵 [SIMULATION] Would play: {new_file}")
            return
        
        try:
            sound = pygame.mixer.Sound(str(new_file))
            
            if self.music_channel_current and self.music_channel_current.get_busy():
                self.next_soundscape = sound
                self.crossfade_progress = 0.0
                self.music_channel_next.play(sound, loops=-1)
                self.music_channel_next.set_volume(0.0)
                print(f"🎵 Crossfading to: {new_file}")
            else:
                self.current_soundscape = sound
                self.music_channel_current.play(sound, loops=-1)
                self.music_channel_current.set_volume(self.config['audio']['soundscape_volume'])
                print(f"🎵 Now playing: {new_file}")
            
        except Exception as e:
            print(f"🎵 Error loading soundscape: {e}")
            print(f"   Make sure the audio file is valid: {new_file}")
    
    def update_crossfade(self, dt: float):
        """Update crossfade between soundscapes"""
        if not PYGAME_AVAILABLE or self.next_soundscape is None:
            return
        
        duration = self.config['audio']['crossfade_duration']
        self.crossfade_progress += dt / duration
        
        if self.crossfade_progress >= 1.0:
            self.music_channel_current.stop()
            self.current_soundscape = self.next_soundscape
            self.next_soundscape = None
            self.music_channel_current, self.music_channel_next = self.music_channel_next, self.music_channel_current
            self.crossfade_progress = 1.0
        else:
            vol_base = self.config['audio']['ducked_volume'] if self.ducking else self.config['audio']['soundscape_volume']
            self.music_channel_current.set_volume(vol_base * (1.0 - self.crossfade_progress))
            self.music_channel_next.set_volume(vol_base * self.crossfade_progress)
    
    def play_beep(self, frequency: int, duration: float):
        """Generate and play a beep tone"""
        print(f"🔊 BEEP: {frequency}Hz for {duration:.2f}s")
        
        if not PYGAME_AVAILABLE:
            print(f"🔊 [SIMULATION] Beep would play: {frequency}Hz for {duration:.2f}s")
            return
        
        try:
            sample_rate = 44100
            samples = int(sample_rate * duration)
            wave = np.sin(2 * np.pi * frequency * np.arange(samples) / sample_rate)
            
            envelope = np.ones(samples)
            fade_samples = int(sample_rate * 0.01)
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            wave = wave * envelope
            
            wave = (wave * 32767 * self.config['audio']['beep_volume']).astype(np.int16)
            stereo_wave = np.column_stack((wave, wave))
            
            sound = pygame.sndarray.make_sound(stereo_wave)
            self.beep_channel.play(sound)
            
            self.set_ducking(True)
            
        except Exception as e:
            print(f"🔊 Beep error: {e}")
    
    def speak(self, text: str, force: bool = False):
        """Text-to-speech using system TTS"""
        current_time = time.time()
        if not force and (current_time - self.last_speech_time) < self.config['audio']['speech_rate_limit']:
            return
        
        self.last_speech_time = current_time
        self.set_ducking(True)
        
        print(f"🗣️  SPEAKING: {text}")
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()
    
    def _speak_thread(self, text: str):
        """Background thread for TTS"""
        try:
            result = subprocess.run(['which', 'say'], capture_output=True)
            if result.returncode == 0:
                subprocess.run(['say', text], check=True)
            else:
                result = subprocess.run(['which', 'espeak'], capture_output=True)
                if result.returncode == 0:
                    subprocess.run(['espeak', text], check=True)
                else:
                    print(f"TTS: {text}")
            
            time.sleep(0.5)
            self.set_ducking(False)
            
        except Exception as e:
            print(f"TTS error: {e}")
            self.set_ducking(False)
    
    def set_ducking(self, enabled: bool):
        """Duck music volume when speaking/beeping"""
        if not PYGAME_AVAILABLE:
            return
        
        self.ducking = enabled
        target_vol = self.config['audio']['ducked_volume'] if enabled else self.config['audio']['soundscape_volume']
        
        if self.music_channel_current:
            self.music_channel_current.set_volume(target_vol * (1.0 - self.crossfade_progress) if self.next_soundscape else target_vol)
        if self.music_channel_next and self.next_soundscape:
            self.music_channel_next.set_volume(target_vol * self.crossfade_progress)


# ============================================================================
# Main Coordinator (Updated for YOLOv8)
# ============================================================================

@dataclass
class SystemState:
    """Current state of the assistive system"""
    distance: float = 999.0
    current_zone: int = 0
    current_scene: Optional[str] = None
    soundscape_folder: Optional[str] = None
    last_detection_announcement: float = 0
    announced_objects: set = None
    
    def __post_init__(self):
        if self.announced_objects is None:
            self.announced_objects = set()


class AssistiveNavigationSystem:
    """Main coordinator for the entire system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.state = SystemState()
        
        print("📡 Initializing Ultrasonic Sensor...")
        if RASPBERRY_PI:
            self.ultrasonic = UltrasonicSensor(
                self.config['ultrasonic']['trig_pin'],
                self.config['ultrasonic']['echo_pin'],
                self.config['ultrasonic']['median_window']
            )
        else:
            print("📡 [SIMULATION] Ultrasonic sensor not available (not on Raspberry Pi)")
            self.ultrasonic = UltrasonicSensor(23, 24, 5)  # Will use simulation mode
        
        print("🤖 Initializing Vision Module (YOLOv8 + CLIP)...")
        self.vision = VisionModuleYOLO(self.config)
        
        print("🔊 Initializing Audio Engine...")
        self.audio = AudioEngine(self.config)
        
        self.camera = None
        self.init_camera()
        
        self.running = False
        self.threads = []
        
        self.last_scene_inference = 0
        self.last_detect_inference = 0
        self.last_beep = 0
        
    def init_camera(self):
        """Initialize camera"""
        try:
            print("📷 Initializing camera...")
            self.camera = cv2.VideoCapture(0)
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['vision']['camera_width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['vision']['camera_height'])
            self.camera.set(cv2.CAP_PROP_FPS, self.config['vision']['camera_fps'])
            
            if not self.camera.isOpened():
                print("⚠️  WARNING: Could not open camera")
                print("   Make sure your webcam is connected and not in use by another app")
                self.camera = None
            else:
                # Test camera
                ret, frame = self.camera.read()
                if ret:
                    h, w = frame.shape[:2]
                    print(f"📷 Camera initialized: {w}x{h} @ {self.config['vision']['camera_fps']} FPS")
                else:
                    print("⚠️  WARNING: Camera opened but cannot read frames")
                    self.camera = None
        except Exception as e:
            print(f"📷 Camera init error: {e}")
            self.camera = None
    
    def ultrasonic_loop(self):
        """Fast loop for ultrasonic collision avoidance"""
        while self.running:
            distance = self.ultrasonic.get_distance()
            if distance is None:
                time.sleep(0.02)
                continue
            
            self.state.distance = distance
            
            zones = self.config['ultrasonic']['zones']
            hysteresis = self.config['ultrasonic']['hysteresis']
            
            for i, zone in enumerate(zones):
                min_dist = zone['min'] - hysteresis if i > self.state.current_zone else zone['min']
                max_dist = zone['max'] + hysteresis if i < self.state.current_zone else zone['max']
                
                if min_dist <= distance < max_dist:
                    self.state.current_zone = i
                    
                    # Only play beeps if we have a real ultrasonic sensor
                    if RASPBERRY_PI:
                        if 'continuous' in zone and zone['continuous']:
                            if time.time() - self.last_beep > 0.5:
                                self.audio.play_beep(zone['pitch'], 0.5)
                                self.audio.speak("Stop", force=True)
                                self.last_beep = time.time()
                        elif zone['beep_interval'] > 0:
                            if time.time() - self.last_beep > zone['beep_interval']:
                                self.audio.play_beep(zone['pitch'], self.config['audio']['beep_duration'])
                                self.last_beep = time.time()
                    
                    break
            
            time.sleep(0.02)
    
    def vision_loop(self):
        """Slow loop for vision processing"""
        while self.running:
            if self.camera is None or not self.camera.isOpened():
                time.sleep(0.1)
                continue
            
            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            
            # Scene classification with CLIP
            if current_time - self.last_scene_inference > self.config['vision']['scene_inference_interval']:
                scene = self.vision.classify_scene(frame)
                if scene and scene != self.state.current_scene:
                    self.state.current_scene = scene
                    self.handle_scene_change(scene)
                
                self.last_scene_inference = current_time
            
            # Object detection with YOLOv8
            if current_time - self.last_detect_inference > self.config['vision']['detect_inference_interval']:
                detections = self.vision.detect_objects(frame)
                self.handle_detections(detections)
                
                self.last_detect_inference = current_time
            
            time.sleep(0.05)
    
    def handle_scene_change(self, scene: str):
        """Handle scene classification change"""
        scene_mapping = self.config['scene_mapping']
        soundscape_folder = scene_mapping.get(scene, 'indoor')
        
        if soundscape_folder != self.state.soundscape_folder:
            print(f"🌍 Scene changed: {scene} -> {soundscape_folder}")
            self.state.soundscape_folder = soundscape_folder
            self.audio.load_soundscape(soundscape_folder)
    
    def handle_detections(self, detections: List[dict]):
        """Handle object detections (minimal spoken cues)"""
        priority_objects = self.config['priority_objects']
        current_time = time.time()
        
        for det in detections:
            obj_class = det['class'].lower()
            
            if obj_class not in priority_objects:
                continue
            
            if obj_class in self.state.announced_objects:
                continue
            
            if current_time - self.state.last_detection_announcement < self.config['audio']['speech_rate_limit']:
                continue
            
            # Determine position
            center_x = det.get('center_x', 0.5)
            if center_x < 0.33:
                position = "left"
            elif center_x < 0.67:
                position = "ahead"
            else:
                position = "right"
            
            # Speak cue
            self.audio.speak(f"{obj_class.title()} {position}")
            
            self.state.announced_objects.add(obj_class)
            self.state.last_detection_announcement = current_time
            
            # Clear after 10 seconds
            threading.Timer(10.0, lambda: self.state.announced_objects.discard(obj_class)).start()
            
            break
    
    def audio_update_loop(self):
        """Update audio engine"""
        while self.running:
            self.audio.update_crossfade(0.1)
            time.sleep(0.1)
    
    def start(self):
        """Start all system threads"""
        print("\n" + "="*60)
        print("🚀 ASSISTIVE NAVIGATION SYSTEM STARTING (YOLOv8)")
        print("="*60)
        print("Mode: Fully Autonomous")
        print("📡 Collision avoidance: ACTIVE" + (" (SIMULATION)" if not RASPBERRY_PI else ""))
        print("🌍 Scene awareness: ACTIVE (CLIP)")
        print("👁️  Object detection: ACTIVE (YOLOv8)")
        print("🔊 Ambient audio: ACTIVE")
        print("="*60 + "\n")
        
        self.running = True
        
        t1 = threading.Thread(target=self.ultrasonic_loop, daemon=True, name="Ultrasonic")
        t2 = threading.Thread(target=self.vision_loop, daemon=True, name="Vision")
        t3 = threading.Thread(target=self.audio_update_loop, daemon=True, name="Audio")
        
        t1.start()
        t2.start()
        t3.start()
        
        self.threads = [t1, t2, t3]
        
        print("System running. Press Ctrl+C to stop.\n")
    
    def stop(self):
        """Stop all threads and cleanup"""
        print("\nShutting down...")
        self.running = False
        
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        if self.camera:
            self.camera.release()
        
        self.ultrasonic.cleanup()
        
        if PYGAME_AVAILABLE:
            pygame.mixer.quit()
        
        print("System stopped.")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    import sys
    
    config_path = "config.yaml" if len(sys.argv) < 2 else sys.argv[1]
    
    system = AssistiveNavigationSystem(config_path)
    
    try:
        system.start()
        
        while True:
            time.sleep(2)  # Print status every 2 seconds
            if system.running:
                print(f"📊 Status: Distance={system.state.distance:.2f}m | "
                      f"Zone={system.state.current_zone} | "
                      f"Scene={system.state.current_scene or 'N/A'} | "
                      f"Soundscape={system.state.soundscape_folder or 'N/A'}")
    
    except KeyboardInterrupt:
        print("\n\nReceived interrupt signal...")
    
    finally:
        system.stop()


if __name__ == "__main__":
    main()

