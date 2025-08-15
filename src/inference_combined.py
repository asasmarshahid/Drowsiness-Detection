import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms, models
import torch.nn as nn
import mediapipe as mp
import argparse
import os
import time
from PIL import Image

from Custom_CNN.model import DrowsinessClassifier

class DrowsinessDetector:
    def __init__(self, model_type: str, yolo_weights_path: str = None, mobilenet_weights_path: str = None, custom_cnn_weights_path: str = None):
        self.model_type = model_type.lower()

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        )

        # Load selected model
        self.yolo_model = None
        self.mobilenet_model = None
        self.custom_cnn_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.model_type == 'yolo':
            if not yolo_weights_path:
                raise ValueError("yolo_weights_path is required for model_type='yolo'")
            self.yolo_model = YOLO(yolo_weights_path)
        elif self.model_type == 'mobilenet':
            if not mobilenet_weights_path:
                raise ValueError("mobilenet_weights_path is required for model_type='mobilenet'")
            self.mobilenet_model = self._load_mobilenet(mobilenet_weights_path)
            self.mobilenet_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        elif self.model_type == 'custom_cnn':
            if not custom_cnn_weights_path:
                raise ValueError("custom_cnn_weights_path is required for model_type='custom_cnn'")
            self.custom_cnn_model = self._load_custom_cnn(custom_cnn_weights_path)
            self.custom_cnn_transform = transforms.Compose([
                 transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
             ])
        else:
            raise ValueError("model_type must be 'yolo', 'mobilenet', or 'custom_cnn'")

        # Class names and thresholds
        self.classes = ['DROWSY', 'NON-DROWSY']
        self.conf_threshold = 0.5
        
        # Optimization: Pre-define colors to avoid repeated tuple creation
        self.colors = {
            'GREEN': (0, 255, 0),
            'RED': (0, 0, 255),
            'YELLOW': (0, 255, 255),
            'GRAY': (128, 128, 128),
            'CYAN': (255, 255, 0),
            'ORANGE': (0, 165, 255)
        }

    def _load_mobilenet(self, weights_path: str):
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        state = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state)
        if self.device == 'cuda':
            model = model.cuda()
        model.eval()
        return model

    def _detect_faces(self, frame):
        # Convert BGR to RGB once
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)
        
        if not results.detections:
            return []
            
        faces = []
        h, w = frame.shape[:2]  # More efficient unpacking
        
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            bw = min(int(bbox.width * w), w - x)
            bh = min(int(bbox.height * h), h - y)
            if bw > 0 and bh > 0:
                faces.append((x, y, bw, bh))
        return faces

    def _crop_face(self, frame, bbox, pad_ratio=0.1):
        x, y, bw, bh = bbox
        pad = int(min(bw, bh) * pad_ratio)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + bw + pad)
        y2 = min(frame.shape[0], y + bh + pad)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _predict_yolo(self, face_img):
        # Optimize: Direct resize without intermediate copies
        if face_img.shape[:2] != (224, 224):
            resized = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_LINEAR)
        else:
            resized = face_img
            
        start = time.perf_counter()  # More precise timing
        results = self.yolo_model(resized, verbose=False)
        infer_ms = (time.perf_counter() - start) * 1000.0
        
        if len(results) == 0:
            return None, 0.0, infer_ms
        r0 = results[0]
        if getattr(r0, 'probs', None) is None:
            return None, 0.0, infer_ms
            
        probs = r0.probs.data.detach().cpu().numpy()
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        return pred, conf, infer_ms

    def _predict_mobilenet(self, face_img):
        tensor = self.mobilenet_transform(face_img).unsqueeze(0)
        if self.device == 'cuda':
            tensor = tensor.cuda()
            
        start = time.perf_counter()  # More precise timing
        with torch.no_grad():
            out = self.mobilenet_model(tensor)
            probs = torch.softmax(out, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            conf = float(probs[0, pred].item())
        infer_ms = (time.perf_counter() - start) * 1000.0
        return pred, conf, infer_ms

    def _load_custom_cnn(self, weights_path: str):
        model = DrowsinessClassifier().to(self.device)
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.eval()
        return model

    def _predict_custom_cnn(self, face_img):
        try:
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            if rgb_img.shape[:2] != (224, 224):
                rgb_img = cv2.resize(rgb_img, (224, 224))
            
            pil_img = Image.fromarray(rgb_img)
            
            tensor = self.custom_cnn_transform(pil_img).unsqueeze(0).to(self.device)
            
            start = time.perf_counter()
            with torch.no_grad():
                pred_class, confidence = self.custom_cnn_model.predict(tensor)
            infer_ms = (time.perf_counter() - start) * 1000.0
            
            return pred_class.item(), confidence.item(), infer_ms
        except Exception as e:
            print(f"Error in Custom CNN prediction: {str(e)}")
            print(f"Input shape: {rgb_img.shape}")
            print(f"Input type: {type(rgb_img)}")
            return None, 0.0, 0.0

    def _predict(self, face_img):
        if self.model_type == 'yolo':
            return self._predict_yolo(face_img)
        elif self.model_type == 'mobilenet':
            return self._predict_mobilenet(face_img)
        else:
            return self._predict_custom_cnn(face_img)

    def annotate_image(self, frame, bbox, label_text, conf, fps=None, infer_ms=None):
        # Use pre-defined colors for efficiency
        if label_text == 'NON-DROWSY':
            label_color = self.colors['GREEN']
        elif label_text == 'DROWSY':
            label_color = self.colors['RED']
        elif label_text == 'LOW-CONFIDENCE':
            label_color = self.colors['YELLOW']
        else:  # 'NO FACE' or others
            label_color = self.colors['GRAY']

        # Draw bbox
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)

        # Pre-format strings to avoid repeated string operations
        label_str = f"{label_text} ({conf:.2f})"
        cv2.putText(frame, label_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

        # Bottom-left stats
        y = frame.shape[0] - 20
        if fps is not None:
            fps_str = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_str, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['CYAN'], 2)
            y -= 25
        if infer_ms is not None:
            latency_str = f"Latency: {infer_ms:.1f} ms"
            cv2.putText(frame, latency_str, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['ORANGE'], 2)
        return frame

    def run_image(self, image_path: str, output_path: str = None):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: could not read image: {image_path}")
            return
            
        faces = self._detect_faces(frame)
        if not faces:
            print("No face detected.")
            cv2.putText(frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['GRAY'], 2)
            if output_path:
                cv2.imwrite(output_path, frame)
            else:
                cv2.imshow('Drowsiness Detection', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return
            
        # Largest face
        largest_face = max(faces, key=lambda b: b[2] * b[3])
        face_img, bbox_xyxy = self._crop_face(frame, largest_face)
        if face_img.size == 0:
            print("Invalid face region.")
            return
            
        pred, conf, infer_ms = self._predict(face_img)
        label = self.classes[pred] if pred is not None and conf >= self.conf_threshold else 'LOW-CONFIDENCE'
        
        frame = self.annotate_image(frame, bbox_xyxy, label, conf, fps=None, infer_ms=infer_ms)
        if output_path:
            cv2.imwrite(output_path, frame)
            print(f"Saved: {output_path}")
        else:
            cv2.imshow('Drowsiness Detection', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def run_video(self, video_path: str, output_path: str = None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: could not open video: {video_path}")
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Cap playback FPS to reasonable range (24-30 FPS)
        target_fps = min(30, max(24, src_fps))
        frame_interval = 1.0 / target_fps
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
            
        # Optimization: Process every N frames for face detection to reduce computational load
        face_detection_interval = max(1, int(src_fps // 30))  # Detect faces ~30 times per second
        frame_count = 0
        last_faces = []
        last_bbox = None
        
        prev_time = time.perf_counter()
        last_display_time = time.perf_counter()
        
        while True:
            loop_start = time.perf_counter()
            
            ret, frame = cap.read()
            if not ret:
                break
                
            # Optimize: Only detect faces periodically
            if frame_count % face_detection_interval == 0:
                last_faces = self._detect_faces(frame)
                
            faces = last_faces
            label = 'NO FACE'
            conf = 0.0
            infer_ms = None
            bbox_xyxy = last_bbox
            
            if faces:
                largest_face = max(faces, key=lambda b: b[2] * b[3])
                face_img, bbox_xyxy = self._crop_face(frame, largest_face)
                last_bbox = bbox_xyxy
                
                if face_img.size > 0:
                    pred, conf, infer_ms = self._predict(face_img)
                    label = self.classes[pred] if pred is not None and conf >= self.conf_threshold else 'LOW-CONFIDENCE'
            else:
                last_bbox = None
                
            # FPS calculation for display (actual processing FPS)
            now = time.perf_counter()
            inst_fps = 1.0 / max(1e-6, now - prev_time)
            prev_time = now
            
            frame = self.annotate_image(frame, bbox_xyxy, label, conf, fps=inst_fps, infer_ms=infer_ms)
            
            if writer:
                writer.write(frame)
                
            # Display frame at controlled rate
            cv2.imshow('Drowsiness Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
            
            # Control playback speed to target FPS
            elapsed = time.perf_counter() - loop_start
            remaining = frame_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
            
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='MediaPipe + (YOLO|MobileNet|Custom CNN) Drowsiness Inference')
    parser.add_argument('--mode', choices=['image', 'video'], default='image', help='image or video input')
    parser.add_argument('--input', type=str, required=True, help='path to input image/video')
    parser.add_argument('--model-type', choices=['yolo', 'mobilenet', 'custom_cnn'], default='yolo', help='which model to use')
    parser.add_argument('--yolo-weights', type=str, default='src/yolo/runs/classify/drowsiness_detection/weights/best.pt')
    parser.add_argument('--mobilenet-weights', type=str, default='src/MobileNetv2/outputs/best_mobilenetv2.pth')
    parser.add_argument('--custom-cnn-weights', type=str, default='src/Custom_CNN/outputs/best_model.pth')
    parser.add_argument('--output', type=str, help='optional output path for annotated result')
    args = parser.parse_args()

    if args.model_type == 'yolo' and not os.path.exists(args.yolo_weights):
        print(f"Error: YOLO weights not found at {args.yolo_weights}")
        return
    if args.model_type == 'mobilenet' and not os.path.exists(args.mobilenet_weights):
        print(f"Error: MobileNetV2 weights not found at {args.mobilenet_weights}")
        return
    if not os.path.exists(args.input):
        print(f"Error: input not found at {args.input}")
        return

    detector = DrowsinessDetector(
        model_type=args.model_type,
        yolo_weights_path=args.yolo_weights,
        mobilenet_weights_path=args.mobilenet_weights,
        custom_cnn_weights_path=args.custom_cnn_weights,
    )
    if args.mode == 'image':
        detector.run_image(args.input, args.output)
    else:
        detector.run_video(args.input, args.output)


if __name__ == '__main__':
    main() 