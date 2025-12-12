from gpiozero import Device
from gpiozero.pins.lgpio import LGPIOFactory
import sys
import termios
import argparse
from time import sleep
from ultralytics import YOLO
import cv2

Device.pin_factory = LGPIOFactory()
from gpiozero import LED

green_led = LED(18)
red_led = LED(17)

def reset_leds():
    """Turn off both LEDs"""
    green_led.off()
    red_led.off()

def process_detections(results):
    """Process YOLO results and control LEDs"""
    detected_classes = []
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            detected_classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
    
    # Control LEDs based on detected classes
    if detected_classes:
        if any(cls in [0, 1] for cls in detected_classes):
            green_led.on()
            red_led.off()
            print(f"Green LED ON - Detected classes: {detected_classes}")
        else:
            green_led.off()
            red_led.on()
            print(f"Red LED ON - Detected classes: {detected_classes}")
    else:
        reset_leds()
        print("No objects detected - Both LEDs OFF")
    
    return detected_classes

def livefeed_mode(model):
    """Live camera feed mode"""
    cap = cv2.VideoCapture(0)
    
    try:
        print("Live feed mode - Press 'q' in the video window to quit")
        print("Classes 0-1: Green LED | Classes 2-21: Red LED")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Run YOLO detection
            results = model(frame, verbose=False)
            
            # Process detections and control LEDs
            process_detections(results)
            
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

def loadimg_mode(model, image_path):
    """Load and process a single image"""
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    try:
        print(f"Image mode - Loading: {image_path}")
        print("Classes 0-1: Green LED | Classes 2-21: Red LED")
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        
        # Process detections and control LEDs
        process_detections(results)
        
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
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'loadimg' and args.path is None:
        parser.error("--path is required when --mode=loadimg")
    
    # Load YOLO model
    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    
    # Set terminal settings
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    try:
        if args.mode == 'livefeed':
            livefeed_mode(model)
        elif args.mode == 'loadimg':
            loadimg_mode(model, args.path)
            
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