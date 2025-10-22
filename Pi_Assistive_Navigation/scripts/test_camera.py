#!/usr/bin/env python3
"""
Test script for USB camera
Displays live feed with FPS counter
Press 'q' to quit, 's' to save snapshot
"""

import cv2
import time

def main():
    print("="*50)
    print("USB Camera Test")
    print("="*50)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save snapshot")
    print("  'i' - Show camera info")
    print("="*50 + "\n")
    
    # Try to open camera
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("ERROR: Could not open camera!")
        print("\nTroubleshooting:")
        print("1. Check USB connection")
        print("2. Run: ls /dev/video*")
        print("3. Try: sudo usermod -a -G video $USER")
        print("4. Reboot and try again")
        return
    
    # Set camera properties
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual properties
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = camera.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera opened successfully!")
    print(f"Resolution: {int(width)}x{int(height)}")
    print(f"FPS: {int(fps)}")
    print()
    
    frame_count = 0
    start_time = time.time()
    snapshot_count = 0
    
    try:
        while True:
            ret, frame = camera.read()
            
            if not ret:
                print("ERROR: Failed to grab frame")
                break
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 0:
                actual_fps = frame_count / elapsed
            else:
                actual_fps = 0
            
            # Add FPS overlay
            cv2.putText(frame, f"FPS: {actual_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display
            cv2.imshow("Camera Test", frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"snapshot_{snapshot_count:03d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
                snapshot_count += 1
            elif key == ord('i'):
                print(f"\nCamera Info:")
                print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
                print(f"  Actual FPS: {actual_fps:.2f}")
                print(f"  Backend: {camera.getBackendName()}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"\nTotal frames captured: {frame_count}")
        print(f"Average FPS: {actual_fps:.2f}")
        print("Camera released. Goodbye!")

if __name__ == "__main__":
    main()

