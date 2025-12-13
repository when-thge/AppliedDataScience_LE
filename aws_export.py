from gpiozero import Device
from gpiozero.pins.lgpio import LGPIOFactory
import sys
import termios
import argparse
from time import sleep
from ultralytics import YOLO
import cv2
import json
from datetime import datetime
import requests

Device.pin_factory = LGPIOFactory()
from gpiozero import LED, Buzzer

green_led = LED(18)
red_led = LED(17)
buzzer = Buzzer(15)

def reset_leds():
    """Turn off both LEDs"""
    green_led.off()
    red_led.off()

def process_detections(results, confidence_threshold=0.70, lambda_url=None):
    """Process YOLO results and control LEDs based on confidence threshold"""
    detected_classes = []
    filtered_detections = []
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            # Get classes and confidences
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            
            # Filter by confidence threshold
            for cls, conf in zip(classes, confidences):
                if conf >= confidence_threshold:
                    detected_classes.append(cls)
                    filtered_detections.append((cls, conf))
    
    # Control LEDs based on detected classes
    if detected_classes:
        # Save detection to JSON file or send to Lambda
        save_detection_json(results, filtered_detections, confidence_threshold, lambda_url)
        
        if any(cls in [14, 15] for cls in detected_classes):
            green_led.on()
            red_led.off()
            buzzer.off()
            print(f"Green LED ON - Detected classes: {filtered_detections}")
        else:
            green_led.off()
            red_led.on()
            buzzer.on()
            sleep(5)
            buzzer.off()
            print(f"Red LED ON - Detected classes: {filtered_detections}")
    else:
        reset_leds()
        print(f"No objects detected above {confidence_threshold} confidence - Both LEDs OFF")
    
    return detected_classes

def save_detection_json(results, filtered_detections, confidence_threshold, lambda_url=None):
    """Save detection results to AWS Lambda"""
    # Get class names from the model
    model_names = results[0].names
    
    # Prepare detection data
    detections = []
    for cls_id, conf in filtered_detections:
        class_name = model_names.get(cls_id, f"Unknown_{cls_id}")
        detections.append({
            "class_id": int(cls_id),
            "class_name": class_name,
            "confidence": float(conf)
        })
    
    # Create JSON structure
    detection_data = {
        "timestamp": datetime.now().isoformat(),
        "confidence_threshold": confidence_threshold,
        "detections": detections
    }
    
    # Send to AWS Lambda if URL is provided
    if lambda_url:
        try:
            response = requests.post(
                lambda_url,
                json=detection_data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"✓ Detection sent to Lambda: {len(detections)} object(s)")
            else:
                print(f"✗ Lambda returned status {response.status_code}: {response.text}")
                # Fallback: save locally if Lambda fails
                save_local_json(detection_data)
                
        except requests.exceptions.Timeout:
            print("✗ Lambda request timed out")
            save_local_json(detection_data)
        except requests.exceptions.RequestException as e:
            print(f"✗ Error sending to Lambda: {e}")
            save_local_json(detection_data)
    else:
        # No Lambda URL provided, save locally
        save_local_json(detection_data)

def save_local_json(detection_data):
    """Save detection data to local JSON file as backup"""
    filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(filename, 'w') as f:
            json.dump(detection_data, f, indent=4)
        print(f"Detection saved locally: {filename}")
    except Exception as e:
        print(f"Error saving detection JSON: {e}")

def livefeed_mode(model, confidence_threshold, lambda_url=None):
    """Live camera feed mode"""
    cap = cv2.VideoCapture(0)
    
    # Optimize camera settings for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag
    
    frame_skip = 2  # Process every Nth frame
    frame_count = 0
    last_results = None
    
    try:
        print("Live feed mode - Press 'q' in the video window to quit")
        print("Classes 14-15: Green LED | Classes 0-13 & 16-22: Red LED")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Optimized: Processing every {frame_skip} frames at 640x480")
        if lambda_url:
            print(f"AWS Lambda URL: {lambda_url}")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            flipped = cv2.flip(frame, 1)
            frame_count += 1
            
            # Only run detection on every Nth frame
            if frame_count % frame_skip == 0:
                # Run YOLO detection with optimized settings
                results = model(flipped, verbose=False, imgsz=640, half=False)
                last_results = results
                
                # Process detections and control LEDs
                process_detections(results, confidence_threshold, lambda_url)
            
            # Use last results for annotation to maintain smooth display
            if last_results is not None:
                annotated_frame = last_results[0].plot()
            else:
                annotated_frame = flipped
            
            # Display the frame
            cv2.imshow('YOLO Detection - Live Feed', annotated_frame)
            
            # Check for 'q' key in OpenCV window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

def loadimg_mode(model, image_path, confidence_threshold, lambda_url=None):
    """Load and process a single image"""
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    try:
        print(f"Image mode - Loading: {image_path}")
        print("Classes 14-15: Green LED | Classes 0-13 & 16-22: Red LED")
        print(f"Confidence threshold: {confidence_threshold}")
        if lambda_url:
            print(f"AWS Lambda URL: {lambda_url}")
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        
        # Process detections and control LEDs
        process_detections(results, confidence_threshold, lambda_url)
        
        # Annotate frame with detections
        annotated_frame = results[0].plot()
        
        # Display the frame
        cv2.imshow('YOLO Detection - Static Image', annotated_frame)
        
        print("\nPress 'q' in the window to quit")
        
        # Wait for 'q' key
        while True:
            if cv2.waitKey(100) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
                
    finally:
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection with LED indicators (NCNN)')
    parser.add_argument('--mode', type=str, required=True, choices=['livefeed', 'loadimg'],
                        help='Detection mode: livefeed or loadimg')
    parser.add_argument('--path', type=str, default=None,
                        help='Path to image file (required when mode=loadimg)')
    parser.add_argument('--model', type=str, default='best_yolo_ncnn_model',
                        help='Path to YOLO NCNN model folder (default: best_yolo_ncnn_model)')
    parser.add_argument('--confidence', type=float, default=0.70,
                        help='Confidence threshold for detections (default: 0.70)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'loadimg' and args.path is None:
        parser.error("--path is required when --mode=loadimg")
    
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0.0 and 1.0")
    
    # Load YOLO NCNN model
    print(f"Loading YOLO NCNN model from: {args.model}")
    print(f"Confidence threshold: {args.confidence}")
    
    # For NCNN models, Ultralytics expects the folder path
    # The folder should contain the .param and .bin files
    model = YOLO(args.model, task='detect')
    
    # Set terminal settings
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    try:
        if args.mode == 'livefeed':
            livefeed_mode(model, args.confidence)
        elif args.mode == 'loadimg':
            loadimg_mode(model, args.path, args.confidence)
            
    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        reset_leds()
        print("LEDs turned off, exiting")

if __name__ == "__main__":
    main()
