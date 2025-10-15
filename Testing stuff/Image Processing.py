import cv2
from ultralytics import YOLO

def real_time_object_detection():
    # Load a pre-trained YOLOv8 model
    # You can choose different models like 'yolov8n.pt' (nano), 'yolov8s.pt' (small), etc.
    model = YOLO('yolov8s.pt')

    # Open the webcam (0 for default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform object detection on the frame
        results = model(frame, stream=True) # stream=True for generator output

        # Process the detection results
        for r in results:
            boxes = r.boxes  # Bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # Coordinates
                conf = float(box.conf[0]) # Confidence score
                cls = int(box.cls[0]) # Class ID
                label = model.names[cls] # Class name

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow('Real-time Object Detection', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    real_time_object_detection()