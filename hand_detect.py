#!/usr/bin/env python3
"""
YOLOv11n Pose Hand Detection using NCNN
For Raspberry Pi 5 with camera input
"""

import cv2
import numpy as np
import argparse
import sys
import time

try:
    import ncnn
except ImportError:
    print("Error: ncnn module not found. Install with: pip install ncnn")
    sys.exit(1)


class HandPoseDetector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45):
        """
        Initialize NCNN hand pose detector
        
        Args:
            model_path: Path to NCNN model (without .param/.bin extension)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize NCNN network
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = True  # Enable GPU acceleration on RPi5
        
        # Load model
        param_path = f"{model_path}.param"
        bin_path = f"{model_path}.bin"
        
        if self.net.load_param(param_path) != 0:
            raise FileNotFoundError(f"Failed to load param file: {param_path}")
        if self.net.load_model(bin_path) != 0:
            raise FileNotFoundError(f"Failed to load model file: {bin_path}")
        
        print(f"Model loaded successfully from {model_path}")
        
        # YOLOv11 input size (typically 640x640)
        self.input_size = 640
        
        # Hand keypoint connections for visualization (21 keypoints)
        self.hand_connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
    
    def preprocess(self, img):
        """Preprocess image for NCNN inference"""
        # Resize while maintaining aspect ratio
        h, w = img.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h))
        
        # Create padded image
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert to ncnn Mat
        mat_in = ncnn.Mat.from_pixels(
            padded,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            self.input_size,
            self.input_size
        )
        
        # Normalize
        mean_vals = [0.0, 0.0, 0.0]
        norm_vals = [1/255.0, 1/255.0, 1/255.0]
        mat_in.substract_mean_normalize(mean_vals, norm_vals)
        
        return mat_in, scale
    
    def postprocess(self, output, scale, img_shape):
        """Process NCNN output to get hand keypoints"""
        detections = []
        
        # Parse output (assuming YOLOv11 pose output format)
        # Output shape: [1, num_predictions, 56+num_keypoints*3]
        # 56 = 4 (bbox) + 1 (conf) + num_classes (usually 1 for hand) + 21*3 (keypoints x,y,conf)
        
        for i in range(output.h):
            # Get detection data
            detection = output.row(i)
            
            # Parse bbox
            x_center = detection[0] / scale
            y_center = detection[1] / scale
            width = detection[2] / scale
            height = detection[3] / scale
            
            # Get confidence
            conf = detection[4]
            
            if conf < self.conf_threshold:
                continue
            
            # Calculate bbox coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # Parse keypoints (21 keypoints, each with x, y, confidence)
            keypoints = []
            kpt_start = 5  # After bbox and conf
            
            for j in range(21):
                kpt_x = detection[kpt_start + j * 3] / scale
                kpt_y = detection[kpt_start + j * 3 + 1] / scale
                kpt_conf = detection[kpt_start + j * 3 + 2]
                keypoints.append([kpt_x, kpt_y, kpt_conf])
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'keypoints': keypoints
            })
        
        return detections
    
    def detect(self, img):
        """Run hand pose detection on image"""
        h, w = img.shape[:2]
        
        # Preprocess
        mat_in, scale = self.preprocess(img)
        
        # Create extractor and input
        ex = self.net.create_extractor()
        ex.input("in0", mat_in)  # Adjust input name if needed
        
        # Run inference
        ret, mat_out = ex.extract("out0")  # Adjust output name if needed
        
        if ret != 0:
            print("Inference failed")
            return []
        
        # Postprocess
        detections = self.postprocess(mat_out, scale, (h, w))
        
        return detections
    
    def draw_results(self, img, detections):
        """Draw hand keypoints and connections on image"""
        for det in detections:
            # Draw bounding box
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            conf_text = f"Hand: {det['confidence']:.2f}"
            cv2.putText(img, conf_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw keypoints
            keypoints = det['keypoints']
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.5:  # Only draw visible keypoints
                    x, y = int(x), int(y)
                    cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
                    # Optionally label keypoints
                    # cv2.putText(img, str(i), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
            
            # Draw connections
            for conn in self.hand_connections:
                pt1_idx, pt2_idx = conn
                pt1 = keypoints[pt1_idx]
                pt2 = keypoints[pt2_idx]
                
                if pt1[2] > 0.5 and pt2[2] > 0.5:  # Both points visible
                    x1, y1 = int(pt1[0]), int(pt1[1])
                    x2, y2 = int(pt2[0]), int(pt2[1])
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        return img


def parse_resolution(res_str):
    """Parse resolution string like '1280x720' to tuple"""
    try:
        w, h = res_str.lower().split('x')
        return int(w), int(h)
    except:
        raise ValueError(f"Invalid resolution format: {res_str}. Use WIDTHxHEIGHT (e.g., 1280x720)")


def open_camera(source, width, height):
    """Open camera with specified source and resolution"""
    # Parse source
    if source.startswith('usb'):
        # USB camera (e.g., usb0 -> /dev/video0)
        cam_id = int(source[3:]) if len(source) > 3 else 0
        cap = cv2.VideoCapture(cam_id)
    elif source.isdigit():
        # Direct camera ID
        cap = cv2.VideoCapture(int(source))
    else:
        # File path or stream URL
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera source: {source}")
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Verify actual resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {actual_w}x{actual_h}")
    
    return cap


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv11n Hand Pose Detection using NCNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hand_detect.py --model=hand_ncnn_model --source=usb0 --resolution=1280x720
  python hand_detect.py --model=models/yolo11n-pose-hand --source=0
  python hand_detect.py --model=hand_model --source=video.mp4
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to NCNN model (without .param/.bin extension)')
    parser.add_argument('--source', type=str, default='0',
                       help='Camera source: usb0, 0, 1, or video file path')
    parser.add_argument('--resolution', type=str, default='640x480',
                       help='Camera resolution (e.g., 1280x720)')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--show-fps', action='store_true',
                       help='Display FPS on screen')
    
    args = parser.parse_args()
    
    # Parse resolution
    width, height = parse_resolution(args.resolution)
    
    # Initialize detector
    print("Initializing hand pose detector...")
    detector = HandPoseDetector(args.model, conf_threshold=args.conf_threshold)
    
    # Open camera
    print(f"Opening camera source: {args.source}")
    cap = open_camera(args.source, width, height)
    
    print("\n=== Hand Pose Detection Started ===")
    print("Press 'q' to quit, 's' to save screenshot")
    print("=" * 40)
    
    fps_counter = 0
    fps_time = time.time()
    fps_display = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Run detection
            detections = detector.detect(frame)
            
            # Draw results
            frame = detector.draw_results(frame, detections)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Display info
            info_text = f"Hands: {len(detections)}"
            if args.show_fps:
                info_text += f" | FPS: {fps_display}"
            
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Hand Pose Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"hand_detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")


if __name__ == '__main__':
    main()
