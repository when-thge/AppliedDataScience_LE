from gpiozero import Device
from gpiozero.pins.lgpio import LGPIOFactory
import sys
import termios
import argparse
from time import sleep
from ultralytics import YOLO
import cv2

Device.pin_factory = LGPIOFactory()
from gpiozero import LED, Buzzer

green_led = LED(18)
red_led = LED(17)
buzzer = Buzzer(15)

def reset_leds():
    """Turn off both LEDs"""
    green_led.off()
    red_led.off()

def process_detections(results, confidence_threshold=0.70):
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

def livefeed_mode(model, confidence_threshold):
    """Live camera feed mode"""
    cap = cv2.VideoCapture(0)
    
    try:
        print("Live feed mode - Press 'q' in the video window to quit")
        print("Classes 14-15: Green LED | Classes 0-13 & 16-22: Red LED")
        print(f"Confidence threshold: {confidence_threshold}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Run YOLO detection
            results = model(frame, verbose=False)
            
            # Process detections and control LEDs
            process_detections(results, confidence_threshold)
            
            # Annotate frame with detections
            annotated_frame = results[0].plot()
            
            # Display the frame
            cv2.imshow('YOLO Detection - Live Feed', annotated_frame)
            
            # Check for 'q' key in OpenCV window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

def loadimg_mode(model, image_path, confidence_threshold):
    """Load and process a single image"""
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    try:
        print(f"Image mode - Loading: {image_path}")
        print("Classes 14-15: Green LED | Classes 0-13 & 16-22: Red LED")
        print(f"Confidence threshold: {confidence_threshold}")
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        
        # Process detections and control LEDs
        process_detections(results, confidence_threshold)
        
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
    parser = argparse.ArgumentParser(description='YOLO Object Detection with LED indicators')
    parser.add_argument('--mode', type=str, required=True, choices=['livefeed', 'loadimg'],
                        help='Detection mode: livefeed or loadimg')
    parser.add_argument('--path', type=str, default=None,
                        help='Path to image file (required when mode=loadimg)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Path to YOLO model file (default: yolov8n.pt)')
    parser.add_argument('--confidence', type=float, default=0.70,
                        help='Confidence threshold for detections (default: 0.70)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'loadimg' and args.path is None:
        parser.error("--path is required when --mode=loadimg")
    
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0.0 and 1.0")
    
    # Load YOLO model
    print(f"Loading YOLO model: {args.model}")
    print(f"Confidence threshold: {args.confidence}")
    model = YOLO(args.model)
    
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
