#!/usr/bin/env python3
"""
Test script for HC-SR04 ultrasonic sensor
Prints distance readings in real-time
"""

import time
import sys

try:
    import RPi.GPIO as GPIO
    RASPBERRY_PI = True
except ImportError:
    print("ERROR: RPi.GPIO not found. This script must run on a Raspberry Pi.")
    sys.exit(1)

# GPIO pins (change if needed)
TRIG_PIN = 23
ECHO_PIN = 24

def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.output(TRIG_PIN, False)
    print("Waiting for sensor to settle...")
    time.sleep(2)
    print("Sensor ready!\n")

def get_distance():
    """Get distance in centimeters"""
    # Trigger pulse
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)
    
    # Wait for echo
    timeout = time.time() + 0.1
    pulse_start = time.time()
    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()
        if pulse_start > timeout:
            return None
    
    pulse_end = time.time()
    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()
        if pulse_end > timeout:
            return None
    
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # Speed of sound / 2
    
    return distance if 2 <= distance <= 400 else None

def main():
    print("="*50)
    print("HC-SR04 Ultrasonic Sensor Test")
    print("="*50)
    print(f"TRIG Pin: GPIO {TRIG_PIN}")
    print(f"ECHO Pin: GPIO {ECHO_PIN}")
    print("="*50)
    print("\nPress Ctrl+C to stop\n")
    
    setup()
    
    try:
        while True:
            distance = get_distance()
            
            if distance is not None:
                # Visual bar graph
                bars = int(distance / 10)
                bar_graph = "█" * min(bars, 40)
                
                print(f"\rDistance: {distance:6.1f} cm  [{bar_graph:<40}]", end="", flush=True)
            else:
                print("\rOut of range or error" + " "*40, end="", flush=True)
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        GPIO.cleanup()
        print("GPIO cleaned up. Goodbye!")

if __name__ == "__main__":
    main()

