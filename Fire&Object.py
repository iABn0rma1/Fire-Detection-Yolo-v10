from ultralytics import YOLO
import cv2
import numpy as np
import time
from screeninfo import get_monitors

def process_frame(frame, fire_model, object_model, fire_conf_threshold, object_conf_threshold, scale_factor):
    original_height, original_width = frame.shape[:2]
    
    # Fire detection
    fire_results = fire_model(frame)
    fire_detected = False
    for r in fire_results:
        boxes = r.boxes
        for box in boxes:
            if box.cls == 0 and box.conf > fire_conf_threshold:
                fire_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                text = f'Fire: {box.conf[0]:.2f}'
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.putText(frame, text, (x1, max(text_height, y1 - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
    
    # Object detection
    object_results = object_model(frame)
    for r in object_results:
        boxes = r.boxes
        for box in boxes:
            if box.conf > object_conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = object_model.names[cls]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                text = f'{label}: {box.conf[0]:.2f}'
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.putText(frame, text, (x1, max(text_height, y1 - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    if fire_detected:
        text = 'FIRE DETECTED!'
        font_scale = 0.75
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.putText(frame, text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
    
    return frame, fire_detected

def fit_frame_to_screen(frame, screen_width, screen_height):
    frame_height, frame_width = frame.shape[:2]
    
    # Calculate the scaling factor to fit the frame within the screen
    scale_width = screen_width / frame_width
    scale_height = screen_height / frame_height
    scale = min(scale_width, scale_height)
    
    # Calculate new dimensions
    new_width = int(frame_width * scale)
    new_height = int(frame_height * scale)
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    # Create a canvas of screen size
    canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    # Calculate position to paste the resized frame
    x_offset = (screen_width - new_width) // 2
    y_offset = (screen_height - new_height) // 2
    
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
    
    return canvas, scale

def main():
    fire_model = YOLO('assets/weights/fire.pt')
    object_model = YOLO('assets/weights/yolov10n.pt')
    video_path = 'assets/videos/fire3.mp4'
    fire_conf_threshold = 0.5
    object_conf_threshold = 0.7
    frame_skip = 2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get the dimensions of the primary monitor
    monitor = get_monitors()[0]
    screen_width, screen_height = monitor.width, monitor.height

    # Adjust screen size to leave some margin
    screen_width = int(screen_width * 0.7)
    screen_height = int(screen_height * 0.7)

    frame_count = 0
    processing_times = []

    cv2.namedWindow("Fire and Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fire and Object Detection", screen_width, screen_height)

    # Get original video dimensions
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        start_time = time.time()

        # Calculate scale factor
        scale_factor = min(screen_width / original_width, screen_height / original_height)

        frame, fire_detected = process_frame(frame, fire_model, object_model, 
                                             fire_conf_threshold, object_conf_threshold, scale_factor)

        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)

        fps = 1 / processing_time
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Fit the frame to the screen
        display_frame, _ = fit_frame_to_screen(frame, screen_width, screen_height)
        
        cv2.imshow("Fire and Object Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    avg_processing_time = sum(processing_times) / len(processing_times)
    print(f"Average processing time: {avg_processing_time:.4f} seconds")
    print(f"Average FPS: {1/avg_processing_time:.2f}")

if __name__ == "__main__":
    main()