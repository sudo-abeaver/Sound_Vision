#!/usr/bin/env python3
"""
Test audio system and soundscape playback
"""

import pygame
import numpy as np
import time
from pathlib import Path

def test_audio_system():
    """Test the audio system components"""
    print("Testing Audio System...")
    
    # Initialize pygame mixer
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    pygame.mixer.set_num_channels(8)
    print("✓ Pygame mixer initialized")
    
    # Test beep generation
    print("\nTesting beep generation...")
    try:
        sample_rate = 44100
        duration = 0.2
        frequency = 440  # A4 note
        
        samples = int(sample_rate * duration)
        wave = np.sin(2 * np.pi * frequency * np.arange(samples) / sample_rate)
        
        # Add envelope to prevent clicks
        envelope = np.ones(samples)
        fade_samples = int(sample_rate * 0.01)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        wave = wave * envelope
        
        wave = (wave * 32767 * 0.5).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        
        sound = pygame.sndarray.make_sound(stereo_wave)
        print("✓ Beep sound generated")
        
        # Play the beep
        channel = pygame.mixer.Channel(0)
        channel.play(sound)
        print("✓ Beep played")
        
        # Wait for beep to finish
        time.sleep(duration + 0.1)
        
    except Exception as e:
        print(f"✗ Beep test failed: {e}")
    
    # Test soundscape loading
    print("\nTesting soundscape loading...")
    soundscape_path = Path("soundscapes")
    
    if soundscape_path.exists():
        print(f"✓ Soundscapes directory found: {soundscape_path}")
        
        # Check each soundscape folder
        folders = ['forest', 'park', 'beach', 'city', 'indoor', 'museum', 'plaza', 'parking', 'residential']
        for folder in folders:
            folder_path = soundscape_path / folder
            if folder_path.exists():
                audio_files = list(folder_path.glob("*.wav")) + list(folder_path.glob("*.mp3"))
                print(f"  {folder}: {len(audio_files)} audio files")
                
                if audio_files:
                    # Try to load the first audio file
                    try:
                        sound = pygame.mixer.Sound(str(audio_files[0]))
                        print(f"    ✓ Loaded: {audio_files[0].name}")
                    except Exception as e:
                        print(f"    ✗ Failed to load {audio_files[0].name}: {e}")
            else:
                print(f"  {folder}: folder not found")
    else:
        print("✗ Soundscapes directory not found")
    
    # Test TTS (text-to-speech)
    print("\nTesting Text-to-Speech...")
    try:
        import subprocess
        
        # Test with 'say' command (macOS)
        result = subprocess.run(['which', 'say'], capture_output=True)
        if result.returncode == 0:
            print("✓ 'say' command available")
            # Test a simple TTS
            subprocess.run(['say', 'Audio system test'], check=True)
            print("✓ TTS test completed")
        else:
            print("✗ 'say' command not available")
            
    except Exception as e:
        print(f"✗ TTS test failed: {e}")
    
    # Cleanup
    pygame.mixer.quit()
    print("\n✓ Audio system test completed")

if __name__ == "__main__":
    test_audio_system()

