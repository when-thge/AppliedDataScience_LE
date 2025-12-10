import cv2
import time
import argparse
import sys
import os
import numpy as np
import ncnn

class YoloPoseNCNN:
    def __init__(self, model_path, target_size=640, prob_threshold=0.25, nms_threshold=0.45):
        self.target_size = target_size
        self.prob_threshold = prob_threshold
        self.nms_threshold = nms_threshold
        
        # Initialize NCNN
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = True # Use GPU if available
        self.net.opt.num_threads = 4

        # 1. HANDLE FOLDER PATH (Fix for your screenshot structure)
        # If 'model_path' is a directory, look for model.ncnn.bin inside it
        if os.path.isdir(model_path):
            param_path = os.path.join(model_path, "model.ncnn.param")
            bin_path = os.path.join(model_path, "model.ncnn.bin")
        else:
            # Fallback if user provides "hand_model" (generic prefix)
            param_path = model_path + ".param"
            bin_path = model_path + ".bin"

        print(f"Loading Param: {param_path}")
        print(f"Loading Bin:   {bin_path}")

        if self.net.load_param(param_path) != 0:
            print(f"Error: Could not load param file at {param_path}")
            sys.exit(1)
        if self.net.load_model(bin_path) != 0:
            print(f"Error: Could not load bin file at {bin_path}")
            sys.exit(1)

        # Standard YOLOv8 Input/Output names
        self.input_name = "images"  # standard ultralytics input
        self.output_name = "output0" # standard ultralytics output

        # Normalization (0-255 -> 0-1)
        self.mean_vals = []
        self.norm_vals = [1/255.0, 1/255.0, 1/255.0]

    def infer(self, bgr_image):
        img_h, img_w = bgr_image.shape[:2]

        # Resize and Pad logic for YOLO (Letterbox)
        w, h = img_w, img_h
        scale = min(self.target_size / w, self.target_size / h)
        nw, nh = int(w * scale), int(h * scale)
        
        # NCNN Resize
        mat_in = ncnn.Mat.from_pixels_resize(
            bgr_image, ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, nw, nh
        )
        
        # Pad to target_size (640x640)
        # YOLO requires gray padding (114), but 0 is usually fine for inference
        w_pad = self.target_size - nw
        h_pad = self.target_size - nh
        mat_in_pad = ncnn.copy_make_border(
            mat_in, h_pad // 2, h_pad - h_pad // 2, 
            w_pad // 2, w_pad - w_pad // 2, 
            ncnn.BorderType.BORDER_CONSTANT, 114.0
        )

        mat_in_pad.substract_mean_normalize(self.mean_vals, self.norm_vals)

        # Run Inference
        ex = self.net.create_extractor()
        ex.input(self.input_name, mat_in_pad)
        ret, mat_out = ex.extract(self.output_name)

        if ret != 0:
            return []

        # 2. DECODE YOLO POSE OUTPUT
        # Output shape is usually [1, 56, 8400] for single class pose
        # 56 channels = 4 box + 1 score + 17*3 keypoints (or similar)
        out = np.array(mat_out)
        
        # Shape handling: [56, 8400] -> transpose to [8400, 56]
        if len(out.shape) == 2:
            out = out.T 
        elif len(out.shape) == 3:
            out = out[0].T
        
        # Filter by confidence
        scores = out[:, 4]
        mask = scores > self.prob_threshold
        detections = out[mask]
        
        if len(detections) == 0:
            return []

        # Extract Boxes and Keypoints
        # box: x, y, w, h (center coordinates)
        boxes_cxcy = detections[:, 0:4]
        scores = detections[:, 4]
        kpts = detections[:, 5:] # The rest are keypoints (x, y, vis)

        # Convert box center -> top-left for NMS
        boxes_xywh = boxes_cxcy.copy()
        boxes_xywh[:, 0] = boxes_cxcy[:, 0] - boxes_cxcy[:, 2] / 2
        boxes_xywh[:, 1] = boxes_cxcy[:, 1] - boxes_cxcy[:, 3] / 2

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), scores.tolist(), 
            self.prob_threshold, self.nms_threshold
        )

        results = []
        for i in indices:
            # Rescale boxes back to original image
            box = boxes_xywh[i]
            x = (box[0] - (w_pad // 2)) / scale
            y = (box[1] - (h_pad // 2)) / scale
            w_box = box[2] / scale
            h_box = box[3] / scale
            
            # Rescale Keypoints
            # kpts are in format [x1, y1, conf1, x2, y2, conf2 ...]
            curr_kpts = kpts[i].copy()
            # Loop over x,y coordinates (every 3rd element is conf)
            for k in range(0, len(curr_kpts), 3):
                curr_kpts[k] = (curr_kpts[k] - (w_pad // 2)) / scale   # x
                curr_kpts[k+1] = (curr_kpts[k+1] - (h_pad // 2)) / scale # y

            results.append({
                "box": [int(x), int(y), int(w_box), int(h_box)],
                "score": float(scores[i]),
                "kpts": curr_kpts
            })

        return results

def parse_args():
    parser = argparse.ArgumentParser(description='NCNN YOLO Hand Pose')
    parser.add_argument('--model', type=str, required=True, help='Folder path containing model.ncnn.bin/.param')
    parser.add_argument('--source', type=str, default='0', help='Camera index (0) or video file')
    parser.add_argument('--resolution', type=str, default='640x480', help='WxH')
    return parser.parse_args()

def main():
    args = parse_args()

    # Parse Source
    if args.source.isdigit():
        src = int(args.source)
    elif args.source.startswith("usb"):
        src = int(args.source.replace("usb", ""))
    else:
        src = args.source

    # Parse Resolution
    w_target, h_target = map(int, args.resolution.lower().split('x'))

    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_target)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_target)

    if not cap.isOpened():
        print(f"Error: Cannot open source {src}")
        return

    # Initialize Detector
    detector = YoloPoseNCNN(args.model)

    print("Running... Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret: break

        start = time.time()
        results = detector.infer(frame)
        end = time.time()

        # Draw Results
        for res in results:
            x, y, w, h = res['box']
            score = res['score']
            kpts = res['kpts']

            # Draw Box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw Keypoints
            # Assumes format: x, y, conf, x, y, conf...
            for k in range(0, len(kpts), 3):
                kx, ky, kconf = kpts[k], kpts[k+1], kpts[k+2]
                if kconf > 0.5: # Visibility threshold
                    cv2.circle(frame, (int(kx), int(ky)), 4, (0, 0, 255), -1)

        fps = 1.0 / (end - start)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("NCNN Hand Pose", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()