#!/usr/bin/env python3
"""
Assistive Navigation System for Raspberry Pi 4
-----------------------------------------------
Fully autonomous system combining:
- Ultrasonic collision avoidance (always-on beeping)
- Scene-aware ambient soundscapes
- Minimal spoken cues from vision
- Zero user input required

Author: FSE100 Project
License: MIT
"""

import time
import threading
import queue
import numpy as np
import cv2
import yaml
import random
import os
import subprocess
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List

try:
    import RPi.GPIO as GPIO
    RASPBERRY_PI = True
except ImportError:
    print("WARNING: RPi.GPIO not found. Running in simulation mode.")
    RASPBERRY_PI = False

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    print("WARNING: tflite_runtime not found. Falling back to tensorflow.lite")
    import tensorflow.lite as tflite

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
            'scene_model': 'models/scene_int8.tflite',
            'scene_labels': 'labels/scene_labels.txt',
            'detect_model': 'models/detect_int8.tflite',
            'detect_labels': 'labels/detect_labels.txt',
            'scene_inference_interval': 0.5,
            'detect_inference_interval': 0.8,
            'scene_stability_window': 8,
            'camera_width': 320,
            'camera_height': 240,
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
            'coast': 'beach',
            'urban_street': 'city',
            'street': 'city',
            'residential': 'residential',
            'indoor_hall': 'indoor',
            'corridor': 'indoor',
            'staircase': 'indoor',
            'museum_gallery': 'museum',
            'gallery': 'museum',
            'store': 'plaza',
            'plaza': 'plaza',
            'parking_lot': 'parking'
        },
        'art_detection': {
            'enabled': True,
            'min_area_percent': 12,
            'max_area_percent': 60,
            'edge_threshold': 100
        }
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            # Deep merge
            for key, value in user_config.items():
                if key in default_config and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
    
    return default_config


# ============================================================================
# Ultrasonic Sensor Module
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
            time.sleep(0.1)  # Let sensor settle
    
    def get_distance(self) -> Optional[float]:
        """Get distance in meters (median filtered)"""
        if not RASPBERRY_PI:
            # Simulation: random distance
            return random.uniform(0.3, 3.0)
        
        try:
            # Trigger pulse
            GPIO.output(self.trig_pin, True)
            time.sleep(0.00001)  # 10µs pulse
            GPIO.output(self.trig_pin, False)
            
            # Wait for echo
            timeout = time.time() + 0.1  # 100ms timeout
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
            distance = pulse_duration * 17150  # Speed of sound / 2, in cm
            distance_m = distance / 100.0
            
            # Validate range (HC-SR04: 2cm - 400cm)
            if 0.02 <= distance_m <= 4.0:
                self.readings.append(distance_m)
                if len(self.readings) >= 3:
                    return float(np.median(list(self.readings)))
            
            return None
            
        except Exception as e:
            print(f"Ultrasonic error: {e}")
            return None
    
    def cleanup(self):
        """Clean up GPIO"""
        if RASPBERRY_PI:
            GPIO.cleanup()


# ============================================================================
# Vision Module (TFLite Scene + Object Detection)
# ============================================================================

class VisionModule:
    """Handles scene classification and object detection using TFLite"""
    
    def __init__(self, config: dict):
        self.config = config
        self.scene_labels = self._load_labels(config['vision']['scene_labels'])
        self.detect_labels = self._load_labels(config['vision']['detect_labels'])
        
        # Load models
        self.scene_interpreter = None
        self.detect_interpreter = None
        
        scene_model_path = config['vision']['scene_model']
        if os.path.exists(scene_model_path):
            self.scene_interpreter = tflite.Interpreter(model_path=scene_model_path)
            self.scene_interpreter.allocate_tensors()
            self.scene_input_details = self.scene_interpreter.get_input_details()
            self.scene_output_details = self.scene_interpreter.get_output_details()
        else:
            print(f"WARNING: Scene model not found at {scene_model_path}")
        
        detect_model_path = config['vision']['detect_model']
        if os.path.exists(detect_model_path):
            self.detect_interpreter = tflite.Interpreter(model_path=detect_model_path)
            self.detect_interpreter.allocate_tensors()
            self.detect_input_details = self.detect_interpreter.get_input_details()
            self.detect_output_details = self.detect_interpreter.get_output_details()
        else:
            print(f"INFO: Detect model not found at {detect_model_path} (optional)")
        
        # Scene stability tracking
        self.scene_votes = deque(maxlen=config['vision']['scene_stability_window'])
    
    def _load_labels(self, label_path: str) -> List[str]:
        """Load labels from text file"""
        if not os.path.exists(label_path):
            return []
        with open(label_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def classify_scene(self, frame: np.ndarray) -> Optional[str]:
        """Classify scene from camera frame"""
        if self.scene_interpreter is None or len(self.scene_labels) == 0:
            return None
        
        try:
            # Preprocess
            input_shape = self.scene_input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]
            
            img = cv2.resize(frame, (width, height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Inference
            self.scene_interpreter.set_tensor(self.scene_input_details[0]['index'], img)
            self.scene_interpreter.invoke()
            output = self.scene_interpreter.get_tensor(self.scene_output_details[0]['index'])
            
            # Get top prediction
            class_id = np.argmax(output[0])
            confidence = output[0][class_id]
            
            if confidence > 0.3 and class_id < len(self.scene_labels):
                scene = self.scene_labels[class_id]
                self.scene_votes.append(scene)
                
                # Return stable scene (majority vote)
                if len(self.scene_votes) >= self.config['vision']['scene_stability_window']:
                    from collections import Counter
                    most_common = Counter(self.scene_votes).most_common(1)[0]
                    if most_common[1] >= self.config['vision']['scene_stability_window'] // 2:
                        return most_common[0]
            
            return None
            
        except Exception as e:
            print(f"Scene classification error: {e}")
            return None
    
    def detect_objects(self, frame: np.ndarray) -> List[dict]:
        """Detect objects in frame"""
        if self.detect_interpreter is None:
            return []
        
        try:
            # Preprocess
            input_shape = self.detect_input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]
            
            img = cv2.resize(frame, (width, height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, axis=0)
            
            # Inference
            self.detect_interpreter.set_tensor(self.detect_input_details[0]['index'], img)
            self.detect_interpreter.invoke()
            
            # Parse outputs (assuming standard TFLite object detection format)
            boxes = self.detect_interpreter.get_tensor(self.detect_output_details[0]['index'])[0]
            classes = self.detect_interpreter.get_tensor(self.detect_output_details[1]['index'])[0]
            scores = self.detect_interpreter.get_tensor(self.detect_output_details[2]['index'])[0]
            
            detections = []
            for i in range(len(scores)):
                if scores[i] > 0.5:  # Confidence threshold
                    class_id = int(classes[i])
                    if class_id < len(self.detect_labels):
                        detections.append({
                            'class': self.detect_labels[class_id],
                            'confidence': float(scores[i]),
                            'bbox': boxes[i].tolist()
                        })
            
            return detections
            
        except Exception as e:
            print(f"Object detection error: {e}")
            return []
    
    def detect_art_heuristic(self, frame: np.ndarray) -> bool:
        """Simple heuristic to detect art/paintings (rectangular frames)"""
        if not self.config['art_detection']['enabled']:
            return False
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, self.config['art_detection']['edge_threshold'], 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            frame_area = frame.shape[0] * frame.shape[1]
            min_area = frame_area * self.config['art_detection']['min_area_percent'] / 100
            max_area = frame_area * self.config['art_detection']['max_area_percent'] / 100
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                    if len(approx) == 4:  # Rectangle-like
                        return True
            
            return False
            
        except Exception as e:
            print(f"Art detection error: {e}")
            return False


# ============================================================================
# Audio Engine (Soundscapes + Beeps + TTS)
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
            pygame.mixer.set_num_channels(8)  # Multiple channels for layering
        
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
            print(f"Soundscape folder not found: {folder_path}")
            return
        
        # Get random audio file from folder
        audio_files = list(folder_path.glob("*.wav")) + list(folder_path.glob("*.mp3"))
        if not audio_files:
            print(f"No audio files in {folder_path}")
            return
        
        new_file = random.choice(audio_files)
        
        if not PYGAME_AVAILABLE:
            print(f"Would play: {new_file}")
            return
        
        try:
            sound = pygame.mixer.Sound(str(new_file))
            
            # Start crossfade if currently playing
            if self.music_channel_current and self.music_channel_current.get_busy():
                self.next_soundscape = sound
                self.crossfade_progress = 0.0
                self.music_channel_next.play(sound, loops=-1)
                self.music_channel_next.set_volume(0.0)
            else:
                self.current_soundscape = sound
                self.music_channel_current.play(sound, loops=-1)
                self.music_channel_current.set_volume(self.config['audio']['soundscape_volume'])
            
        except Exception as e:
            print(f"Error loading soundscape: {e}")
    
    def update_crossfade(self, dt: float):
        """Update crossfade between soundscapes"""
        if not PYGAME_AVAILABLE or self.next_soundscape is None:
            return
        
        duration = self.config['audio']['crossfade_duration']
        self.crossfade_progress += dt / duration
        
        if self.crossfade_progress >= 1.0:
            # Crossfade complete
            self.music_channel_current.stop()
            self.current_soundscape = self.next_soundscape
            self.next_soundscape = None
            self.music_channel_current, self.music_channel_next = self.music_channel_next, self.music_channel_current
            self.crossfade_progress = 1.0
        else:
            # Interpolate volumes
            vol_base = self.config['audio']['ducked_volume'] if self.ducking else self.config['audio']['soundscape_volume']
            self.music_channel_current.set_volume(vol_base * (1.0 - self.crossfade_progress))
            self.music_channel_next.set_volume(vol_base * self.crossfade_progress)
    
    def play_beep(self, frequency: int, duration: float):
        """Generate and play a beep tone"""
        if not PYGAME_AVAILABLE:
            return
        
        try:
            sample_rate = 44100
            samples = int(sample_rate * duration)
            wave = np.sin(2 * np.pi * frequency * np.arange(samples) / sample_rate)
            
            # Apply envelope to avoid clicks
            envelope = np.ones(samples)
            fade_samples = int(sample_rate * 0.01)
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            wave = wave * envelope
            
            # Convert to 16-bit stereo
            wave = (wave * 32767 * self.config['audio']['beep_volume']).astype(np.int16)
            stereo_wave = np.column_stack((wave, wave))
            
            sound = pygame.sndarray.make_sound(stereo_wave)
            self.beep_channel.play(sound)
            
            # Duck music briefly
            self.set_ducking(True)
            
        except Exception as e:
            print(f"Beep error: {e}")
    
    def speak(self, text: str, force: bool = False):
        """Text-to-speech using system TTS"""
        current_time = time.time()
        if not force and (current_time - self.last_speech_time) < self.config['audio']['speech_rate_limit']:
            return
        
        self.last_speech_time = current_time
        self.set_ducking(True)
        
        # Use system TTS (macOS: say, Linux: espeak or pico2wave)
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()
    
    def _speak_thread(self, text: str):
        """Background thread for TTS"""
        try:
            # Try macOS 'say' command first
            result = subprocess.run(['which', 'say'], capture_output=True)
            if result.returncode == 0:
                subprocess.run(['say', text], check=True)
            else:
                # Try espeak on Linux
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
# Main Coordinator
# ============================================================================

@dataclass
class SystemState:
    """Current state of the assistive system"""
    distance: float = 999.0
    current_zone: int = 0
    current_scene: Optional[str] = None
    soundscape_folder: Optional[str] = None
    art_detected: bool = False
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
        
        # Initialize modules
        print("Initializing Ultrasonic Sensor...")
        self.ultrasonic = UltrasonicSensor(
            self.config['ultrasonic']['trig_pin'],
            self.config['ultrasonic']['echo_pin'],
            self.config['ultrasonic']['median_window']
        )
        
        print("Initializing Vision Module...")
        self.vision = VisionModule(self.config)
        
        print("Initializing Audio Engine...")
        self.audio = AudioEngine(self.config)
        
        # Camera setup
        self.camera = None
        self.init_camera()
        
        # Control flags
        self.running = False
        self.threads = []
        
        # Timing
        self.last_scene_inference = 0
        self.last_detect_inference = 0
        self.last_beep = 0
        
    def init_camera(self):
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['vision']['camera_width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['vision']['camera_height'])
            self.camera.set(cv2.CAP_PROP_FPS, self.config['vision']['camera_fps'])
            
            if not self.camera.isOpened():
                print("WARNING: Could not open camera")
                self.camera = None
        except Exception as e:
            print(f"Camera init error: {e}")
            self.camera = None
    
    def ultrasonic_loop(self):
        """Fast loop for ultrasonic collision avoidance"""
        while self.running:
            distance = self.ultrasonic.get_distance()
            if distance is None:
                time.sleep(0.02)
                continue
            
            self.state.distance = distance
            
            # Find current zone with hysteresis
            zones = self.config['ultrasonic']['zones']
            hysteresis = self.config['ultrasonic']['hysteresis']
            
            for i, zone in enumerate(zones):
                # Apply hysteresis at boundaries
                min_dist = zone['min'] - hysteresis if i > self.state.current_zone else zone['min']
                max_dist = zone['max'] + hysteresis if i < self.state.current_zone else zone['max']
                
                if min_dist <= distance < max_dist:
                    self.state.current_zone = i
                    
                    # Handle beeping for this zone
                    if 'continuous' in zone and zone['continuous']:
                        # Continuous tone + "Stop"
                        if time.time() - self.last_beep > 0.5:
                            self.audio.play_beep(zone['pitch'], 0.5)
                            self.audio.speak("Stop", force=True)
                            self.last_beep = time.time()
                    elif zone['beep_interval'] > 0:
                        if time.time() - self.last_beep > zone['beep_interval']:
                            self.audio.play_beep(zone['pitch'], self.config['audio']['beep_duration'])
                            self.last_beep = time.time()
                    
                    break
            
            time.sleep(0.02)  # 50Hz
    
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
            
            # Scene classification
            if current_time - self.last_scene_inference > self.config['vision']['scene_inference_interval']:
                scene = self.vision.classify_scene(frame)
                if scene and scene != self.state.current_scene:
                    self.state.current_scene = scene
                    self.handle_scene_change(scene, frame)
                
                self.last_scene_inference = current_time
            
            # Object detection (lower frequency)
            if current_time - self.last_detect_inference > self.config['vision']['detect_inference_interval']:
                detections = self.vision.detect_objects(frame)
                self.handle_detections(detections)
                
                self.last_detect_inference = current_time
            
            time.sleep(0.05)  # ~20Hz frame grab
    
    def handle_scene_change(self, scene: str, frame: np.ndarray):
        """Handle scene classification change"""
        # Map to soundscape folder
        scene_mapping = self.config['scene_mapping']
        soundscape_folder = scene_mapping.get(scene, 'indoor')
        
        # Check for art (museum override)
        if scene in ['museum_gallery', 'indoor_hall']:
            if self.vision.detect_art_heuristic(frame):
                soundscape_folder = 'museum'
                self.state.art_detected = True
        
        # Load new soundscape if changed
        if soundscape_folder != self.state.soundscape_folder:
            print(f"Scene changed: {scene} -> {soundscape_folder}")
            self.state.soundscape_folder = soundscape_folder
            self.audio.load_soundscape(soundscape_folder)
    
    def handle_detections(self, detections: List[dict]):
        """Handle object detections (minimal spoken cues)"""
        important_classes = {'stairs', 'staircase', 'person', 'bicycle', 'car'}
        current_time = time.time()
        
        for det in detections:
            obj_class = det['class'].lower()
            
            # Only announce important objects
            if obj_class not in important_classes:
                continue
            
            # Rate limiting and deduplication
            if obj_class in self.state.announced_objects:
                continue
            
            if current_time - self.state.last_detection_announcement < self.config['audio']['speech_rate_limit']:
                continue
            
            # Determine position (left/center/right)
            bbox = det['bbox']
            center_x = (bbox[1] + bbox[3]) / 2
            
            if center_x < 0.33:
                position = "left"
            elif center_x < 0.67:
                position = "ahead"
            else:
                position = "right"
            
            # Speak cue
            if obj_class in ['stairs', 'staircase']:
                self.audio.speak(f"Stairs {position}")
            elif obj_class == 'person':
                self.audio.speak(f"Person {position}")
            else:
                self.audio.speak(f"{obj_class.title()} {position}")
            
            self.state.announced_objects.add(obj_class)
            self.state.last_detection_announcement = current_time
            
            # Clear announced objects after 10 seconds
            threading.Timer(10.0, lambda: self.state.announced_objects.discard(obj_class)).start()
            
            break  # Only announce one thing at a time
    
    def audio_update_loop(self):
        """Update audio engine (crossfades, ducking restore)"""
        while self.running:
            self.audio.update_crossfade(0.1)
            time.sleep(0.1)
    
    def start(self):
        """Start all system threads"""
        print("\n" + "="*60)
        print("ASSISTIVE NAVIGATION SYSTEM STARTING")
        print("="*60)
        print("Mode: Fully Autonomous")
        print("- Collision avoidance: ACTIVE")
        print("- Scene awareness: ACTIVE")
        print("- Ambient audio: ACTIVE")
        print("="*60 + "\n")
        
        self.running = True
        
        # Start threads
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
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            # Optional: print status
            if system.running:
                print(f"Distance: {system.state.distance:.2f}m | "
                      f"Zone: {system.state.current_zone} | "
                      f"Scene: {system.state.current_scene or 'N/A'} | "
                      f"Soundscape: {system.state.soundscape_folder or 'N/A'}")
    
    except KeyboardInterrupt:
        print("\n\nReceived interrupt signal...")
    
    finally:
        system.stop()


if __name__ == "__main__":
    main()

