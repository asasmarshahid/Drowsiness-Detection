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
import matplotlib.pyplot as plt

class CombinedDrowsinessDetector:
    def __init__(self, yolo_weights_path, mobilenet_weights_path):
        """
        Initialize combined drowsiness detector with MediaPipe, YOLO, and MobileNet
        
        Args:
            yolo_weights_path: Path to YOLO classification model
            mobilenet_weights_path: Path to MobileNetV2 model
        """
        # Initialize MediaPipe Face Detection
        print("Initializing MediaPipe Face Detection...")
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.5
        )
        
        # Load YOLO model
        print("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_weights_path)
        
        # Load MobileNetV2 model
        print("Loading MobileNetV2 model...")
        self.mobilenet_model = self.load_mobilenet_model(mobilenet_weights_path)
        
        # MobileNet preprocessing (same as training)
        self.mobilenet_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Class names (same as training)
        self.classes = ['Drowsy', 'Non Drowsy']
        
        # Confidence thresholds
        self.yolo_conf_threshold = 0.7
        self.mobilenet_conf_threshold = 0.7
        
        print("All models loaded successfully!")
        print(f"Classes: {self.classes}")
    
    def load_mobilenet_model(self, weights_path):
        """Load MobileNetV2 model with custom classifier (same as training)"""
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 2)  # 2 classes
        
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(weights_path))
            model = model.cuda()
        else:
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        
        model.eval()
        return model
    
    def detect_faces_mediapipe(self, frame):
        """Detect faces using MediaPipe"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                faces.append((x, y, width, height))
        
        return faces
    
    def extract_face_region(self, frame, face_bbox, padding_ratio=0.1):
        """Extract face region with padding"""
        x, y, w, h = face_bbox
        
        # Add padding around the face
        padding = int(min(w, h) * padding_ratio)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        face_region = frame[y1:y2, x1:x2]
        return face_region, (x1, y1, x2, y2)
    
    def preprocess_for_yolo(self, face_region):
        """Preprocess face region for YOLO (resize to 224x224)"""
        resized_face = cv2.resize(face_region, (224, 224))
        return resized_face
    
    def predict_yolo(self, face_region):
        """Predict using YOLO model"""
        # Preprocess for YOLO
        processed_face = self.preprocess_for_yolo(face_region)
        
        # Run inference
        results = self.yolo_model(processed_face, verbose=False)
        
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'probs') and result.probs is not None:
                # Get class probabilities
                probs = result.probs.data.cpu().numpy()
                predicted_class = np.argmax(probs)
                confidence = np.max(probs)
                
                return predicted_class, confidence, probs
            else:
                return None, 0.0, None
        else:
            return None, 0.0, None
    
    def predict_mobilenet(self, face_region):
        """Predict using MobileNetV2 model"""
        try:
            # Preprocess for MobileNet (same as training)
            face_tensor = self.mobilenet_transform(face_region).unsqueeze(0)
            
            if torch.cuda.is_available():
                face_tensor = face_tensor.cuda()
            
            with torch.no_grad():
                outputs = self.mobilenet_model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            return predicted_class, confidence, probabilities[0].cpu().numpy()
        
        except Exception as e:
            print(f"Error in MobileNet prediction: {e}")
            return None, 0.0, None
    
    def combine_predictions(self, yolo_pred, yolo_conf, mobilenet_pred, mobilenet_conf):
        """Combine predictions from both models"""
        if yolo_pred is None and mobilenet_pred is None:
            return None, 0.0, "NO PREDICTION"
        
        if yolo_pred is None:
            return mobilenet_pred, mobilenet_conf, "MOBILENET_ONLY"
        
        if mobilenet_pred is None:
            return yolo_pred, yolo_conf, "YOLO_ONLY"
        
        # Both models made predictions
        if yolo_pred == mobilenet_pred:
            # Models agree - use weighted average confidence
            avg_conf = (yolo_conf + mobilenet_conf) / 2
            return yolo_pred, avg_conf, "BOTH_AGREE"
        else:
            # Models disagree - use the one with higher confidence
            if yolo_conf > mobilenet_conf:
                return yolo_pred, yolo_conf, "YOLO_HIGHER"
            else:
                return mobilenet_pred, mobilenet_conf, "MOBILENET_HIGHER"
    
    def detect_drowsiness(self, image_path):
        """Detect drowsiness in a single image using both models"""
        print(f"\n=== Analyzing: {os.path.basename(image_path)} ===")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image {image_path}")
            return None
        
        # Detect faces using MediaPipe
        faces = self.detect_faces_mediapipe(frame)
        
        if not faces:
            print("No faces detected in the image")
            return frame
        
        print(f"Detected {len(faces)} face(s)")
        
        # Process the largest face (main subject)
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        face_region, (x1, y1, x2, y2) = self.extract_face_region(frame, largest_face)
        
        if face_region.size == 0:
            print("Invalid face region extracted")
            return frame
        
        # Get predictions from both models
        print("Running YOLO prediction...")
        yolo_pred, yolo_conf, yolo_probs = self.predict_yolo(face_region)
        
        print("Running MobileNet prediction...")
        mobilenet_pred, mobilenet_conf, mobilenet_probs = self.predict_mobilenet(face_region)
        
        # Print individual model results
        print(f"\n--- Individual Model Results ---")
        if yolo_pred is not None:
            print(f"YOLO: {self.classes[yolo_pred]} (confidence: {yolo_conf:.3f})")
        else:
            print("YOLO: No prediction")
        
        if mobilenet_pred is not None:
            print(f"MobileNet: {self.classes[mobilenet_pred]} (confidence: {mobilenet_conf:.3f})")
        else:
            print("MobileNet: No prediction")
        
        # Combine predictions
        final_pred, final_conf, combination_method = self.combine_predictions(
            yolo_pred, yolo_conf, mobilenet_pred, mobilenet_conf
        )
        
        print(f"\n--- Combined Result ---")
        print(f"Final Prediction: {self.classes[final_pred] if final_pred is not None else 'None'}")
        print(f"Confidence: {final_conf:.3f}")
        print(f"Combination Method: {combination_method}")
        
        # Draw results on image
        self.draw_results(frame, (x1, y1, x2, y2), yolo_pred, yolo_conf, 
                         mobilenet_pred, mobilenet_conf, final_pred, final_conf, combination_method)
        
        return frame
    
    def draw_results(self, frame, bbox, yolo_pred, yolo_conf, mobilenet_pred, 
                    mobilenet_conf, final_pred, final_conf, combination_method):
        """Draw all results on the image"""
        x1, y1, x2, y2 = bbox
        
        # Draw face bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Determine colors
        def get_color(pred):
            if pred is None:
                return (128, 128, 128)  # Gray
            return (0, 255, 0) if pred == 1 else (0, 0, 255)  # Green for Non-Drowsy, Red for Drowsy
        
        # Draw YOLO result
        yolo_color = get_color(yolo_pred)
        yolo_text = f"YOLO: {self.classes[yolo_pred] if yolo_pred is not None else 'None'} ({yolo_conf:.3f})"
        cv2.putText(frame, yolo_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, yolo_color, 2)
        
        # Draw MobileNet result
        mobilenet_color = get_color(mobilenet_pred)
        mobilenet_text = f"MobileNet: {self.classes[mobilenet_pred] if mobilenet_pred is not None else 'None'} ({mobilenet_conf:.3f})"
        cv2.putText(frame, mobilenet_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mobilenet_color, 2)
        
        # Draw final result
        final_color = get_color(final_pred)
        final_text = f"FINAL: {self.classes[final_pred] if final_pred is not None else 'None'} ({final_conf:.3f})"
        cv2.putText(frame, final_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, final_color, 2)
        
        # Draw combination method
        cv2.putText(frame, f"Method: {combination_method}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw face region indicator
        cv2.putText(frame, "Face Region", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def test_single_image(self, image_path, output_path=None):
        """Test a single image and optionally save result"""
        result_frame = self.detect_drowsiness(image_path)
        
        if result_frame is not None:
            if output_path:
                cv2.imwrite(output_path, result_frame)
                print(f"Result saved to: {output_path}")
            else:
                cv2.imshow('Combined Drowsiness Detection', result_frame)
                print("Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    def test_video(self, video_path, output_path=None):
        """Test video with combined models"""
        print(f"\n=== Video Analysis: {os.path.basename(video_path)} ===")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces
            faces = self.detect_faces_mediapipe(frame)
            
            if faces:
                # Process largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                face_region, (x1, y1, x2, y2) = self.extract_face_region(frame, largest_face)
                
                if face_region.size > 0:
                    # Get predictions
                    yolo_pred, yolo_conf, _ = self.predict_yolo(face_region)
                    mobilenet_pred, mobilenet_conf, _ = self.predict_mobilenet(face_region)
                    
                    # Combine predictions
                    final_pred, final_conf, combination_method = self.combine_predictions(
                        yolo_pred, yolo_conf, mobilenet_pred, mobilenet_conf
                    )
                    
                    # Draw results
                    self.draw_results(frame, (x1, y1, x2, y2), yolo_pred, yolo_conf,
                                    mobilenet_pred, mobilenet_conf, final_pred, final_conf, combination_method)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write or display frame
            if out:
                out.write(frame)
            
            cv2.imshow('Video Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        elapsed_time = time.time() - start_time
        print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
        print(f"Average FPS: {frame_count/elapsed_time:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Combined Drowsiness Detection with MediaPipe, YOLO, and MobileNet')
    parser.add_argument('--input', type=str, required=True, help='Input image or video path')
    parser.add_argument('--yolo-weights', type=str, 
                       default='src/yolo/runs/classify/drowsiness_detection/weights/best.pt',
                       help='Path to YOLO weights')
    parser.add_argument('--mobilenet-weights', type=str,
                       default='src/MobileNetv2/outputs/best_mobilenetv2.pth',
                       help='Path to MobileNetV2 weights')
    parser.add_argument('--output', type=str, help='Output path for result')
    parser.add_argument('--mode', type=str, choices=['image', 'video'], default='image',
                       help='Input mode (image or video)')
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return
    
    # Check if model weights exist
    if not os.path.exists(args.yolo_weights):
        print(f"Error: YOLO weights not found at {args.yolo_weights}")
        return
    
    if not os.path.exists(args.mobilenet_weights):
        print(f"Error: MobileNetV2 weights not found at {args.mobilenet_weights}")
        return
    
    try:
        # Initialize detector
        detector = CombinedDrowsinessDetector(args.yolo_weights, args.mobilenet_weights)
        
        # Run detection
        if args.mode == 'image':
            detector.test_single_image(args.input, args.output)
        else:  # video
            detector.test_video(args.input, args.output)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 