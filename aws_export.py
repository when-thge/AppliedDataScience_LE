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
from gpiozero import LED, Buzzer, OutputDevice

green_led = LED(27)
red_led = LED(17)
buzzer = Buzzer(22)
solenoid = OutputDevice(14, active_high=True, initial_value=False)  # Relay for solenoid

def reset_leds():
    """Turn off both LEDs and deactivate solenoid"""
    green_led.off()
    red_led.off()
    solenoid.off()

def activate_solenoid(duration=3):
    """Activate solenoid for a given duration (seconds)"""
    print(f"Solenoid ACTIVATED - spraying for {duration}s")
    solenoid.on()
    sleep(duration)
    solenoid.off()
    print("Solenoid DEACTIVATED")

def process_detections(results, confidence_threshold=0.70, lambda_url=None):
    """Process YOLO results and control LEDs based on confidence threshold"""
    detected_classes = []
    filtered_detections = []
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            
            for cls, conf in zip(classes, confidences):
                if conf >= confidence_threshold:
                    detected_classes.append(cls)
                    filtered_detections.append((cls, conf))
    
    if detected_classes:
        save_detection_json(results, filtered_detections, confidence_threshold, lambda_url)
        
        if any(cls in [14, 15] for cls in detected_classes):
            # Safe plant detected
            green_led.on()
            red_led.off()
            buzzer.off()
            solenoid.off()
            print(f"Green LED ON - Safe plant detected: {filtered_detections}")
        else:
            # Weed detected
            green_led.off()
            red_led.on()
            buzzer.on()
            print(f"Red LED ON - WEED detected: {filtered_detections}")
            activate_solenoid(duration=3)  # Spray for 3 seconds
            buzzer.off()
    else:
        reset_leds()
        print(f"No objects detected above {confidence_threshold} confidence - Both LEDs OFF")
    
    return detected_classes

def save_detection_json(results, filtered_detections, confidence_threshold, lambda_url=None):
    """Save detection results to AWS Lambda"""
    model_names = results[0].names
    
    detections = []
    for cls_id, conf in filtered_detections:
        class_name = model_names.get(cls_id, f"Unknown_{cls_id}")
        detections.append({
            "class_id": int(cls_id),
            "class_name": class_name,
            "confidence": float(conf)
        })
    
    detection_data = {
        "timestamp": datetime.now().isoformat(),
        "confidence_threshold": confidence_threshold,
        "detections": detections
    }
    
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
                save_local_json(detection_data)
                
        except requests.exceptions.Timeout:
            print("✗ Lambda request timed out")
            save_local_json(detection_data)
        except requests.exceptions.RequestException as e:
            print(f"✗ Error sending to Lambda: {e}")
            save_local_json(detection_data)
    else:
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
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_skip = 2
    frame_count = 0
    last_results = None
    
    try:
        print("Live feed mode - Press 'q' in the video window to quit")
        print("Classes 14-15: Green LED | Others: Red LED + Solenoid spray")
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
            
            if frame_count % frame_skip == 0:
                results = model(flipped, verbose=False, imgsz=640, half=False)
                last_results = results
                process_detections(results, confidence_threshold, lambda_url)
            
            if last_results is not None:
                annotated_frame = last_results[0].plot()
            else:
                annotated_frame = flipped
            
            cv2.imshow('YOLO Detection - Live Feed', annotated_frame)
            
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
        print("Classes 14-15: Green LED | Others: Red LED + Solenoid spray")
        print(f"Confidence threshold: {confidence_threshold}")
        if lambda_url:
            print(f"AWS Lambda URL: {lambda_url}")
        
        results = model(frame, verbose=False)
        process_detections(results, confidence_threshold, lambda_url)
        
        annotated_frame = results[0].plot()
        cv2.imshow('YOLO Detection - Static Image', annotated_frame)
        
        print("\nPress 'q' in the window to quit")
        
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
    parser.add_argument('--lambda-url', type=str, default=None, dest='lambda_url',
                        help='AWS Lambda URL to send detection data (optional)')
    parser.add_argument('--spray-duration', type=float, default=3.0, dest='spray_duration',
                        help='Solenoid spray duration in seconds when weed detected (default: 3.0)')
    
    args = parser.parse_args()
    
    if args.mode == 'loadimg' and args.path is None:
        parser.error("--path is required when --mode=loadimg")
    
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0.0 and 1.0")

    if args.spray_duration <= 0:
        parser.error("--spray-duration must be greater than 0")
    
    print(f"Loading YOLO NCNN model from: {args.model}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Solenoid spray duration: {args.spray_duration}s")
    
    model = YOLO(args.model, task='detect')
    
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    try:
        if args.mode == 'livefeed':
            livefeed_mode(model, args.confidence, args.lambda_url)
        elif args.mode == 'loadimg':
            loadimg_mode(model, args.path, args.confidence, args.lambda_url)
            
    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        reset_leds()
        solenoid.off()
        print("LEDs and solenoid turned off, exiting")

if __name__ == "__main__":
    main()
