import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms, models
import torch.nn as nn
import time
import argparse
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

class DrowsinessTester:
    def __init__(self, yolo_weights_path, mobilenet_weights_path=None, use_yolo=True):
        """
        Initialize the drowsiness tester with YOLOv8 and/or MobileNetV2 models.
        
        Args:
            yolo_weights_path: Path to YOLOv8 weights
            mobilenet_weights_path: Path to MobileNetV2 weights (optional)
            use_yolo: Whether to use YOLOv8 model (default: True)
        """
        self.use_yolo = use_yolo
        
        if use_yolo:
            print("Loading YOLOv8 model...")
            self.yolo_model = YOLO(yolo_weights_path)
        
        if mobilenet_weights_path and not use_yolo:
            print("Loading MobileNetV2 model...")
            self.mobilenet_model = self.load_mobilenet_model(mobilenet_weights_path)
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.classes = ['Drowsy', 'Non Drowsy']
        print("Models loaded successfully!")
    
    def load_mobilenet_model(self, weights_path):
        """Load MobileNetV2 model with custom classifier."""
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(weights_path))
            model = model.cuda()
        else:
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        
        model.eval()
        return model
    
    def classify_image_yolo(self, image):
        """Classify single image using YOLOv8."""
        results = self.yolo_model(image, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            probs = result.probs
            if probs is not None:
                probs_tensor = probs.data
                predicted_class = int(torch.argmax(probs_tensor).item())
                confidence = float(torch.max(probs_tensor).item())
                return predicted_class, confidence
        
        return 0, 0.0
    
    def classify_image_mobilenet(self, image):
        """Classify single image using MobileNetV2."""
        try:
            frame_tensor = self.transform(image).unsqueeze(0)
            
            if torch.cuda.is_available():
                frame_tensor = frame_tensor.cuda()
            
            with torch.no_grad():
                outputs = self.mobilenet_model(frame_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            return predicted_class, confidence
        
        except Exception as e:
            print(f"Error in classification: {e}")
            return 0, 0.0
    
    def test_single_image(self, image_path):
        """Test a single image with detailed analysis."""
        print(f"\n=== Testing Single Image: {image_path} ===")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        # Multiple predictions for accuracy
        predictions = []
        confidences = []
        
        for i in range(5):  # Take 5 predictions
            if self.use_yolo:
                pred_class, conf = self.classify_image_yolo(image)
            else:
                pred_class, conf = self.classify_image_mobilenet(image)
            
            predictions.append(pred_class)
            confidences.append(conf)
            time.sleep(0.1)  # Small delay between predictions
        
        # Calculate statistics
        avg_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        most_common_pred = max(set(predictions), key=predictions.count)
        prediction_consistency = predictions.count(most_common_pred) / len(predictions)
        
        print(f"Predictions: {predictions}")
        print(f"Confidences: {[f'{c:.3f}' for c in confidences]}")
        print(f"Final Classification: {self.classes[most_common_pred]}")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"Confidence Std Dev: {std_confidence:.3f}")
        print(f"Prediction Consistency: {prediction_consistency:.2f}")
        
        # Display image with results
        cv2.putText(image, f"{self.classes[most_common_pred]} ({avg_confidence:.3f})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if most_common_pred == 1 else (0, 0, 255), 2)
        cv2.putText(image, f"Consistency: {prediction_consistency:.2f}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Test Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return most_common_pred, avg_confidence, prediction_consistency
    
    def test_batch_images(self, image_folder, confidence_threshold=0.7):
        """Test multiple images in a folder with confidence filtering."""
        print(f"\n=== Batch Testing Images from: {image_folder} ===")
        
        # Get all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_folder, ext)))
            image_files.extend(glob.glob(os.path.join(image_folder, ext.upper())))
        
        if not image_files:
            print(f"No images found in {image_folder}")
            return
        
        results = []
        high_confidence_results = []
        
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Load and classify image
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Multiple predictions for accuracy
            predictions = []
            confidences = []
            
            for _ in range(3):  # 3 predictions per image
                if self.use_yolo:
                    pred_class, conf = self.classify_image_yolo(image)
                else:
                    pred_class, conf = self.classify_image_mobilenet(image)
                
                predictions.append(pred_class)
                confidences.append(conf)
            
            # Calculate statistics
            avg_confidence = np.mean(confidences)
            most_common_pred = max(set(predictions), key=predictions.count)
            consistency = predictions.count(most_common_pred) / len(predictions)
            
            result = {
                'image': os.path.basename(image_path),
                'prediction': most_common_pred,
                'confidence': avg_confidence,
                'consistency': consistency,
                'class_name': self.classes[most_common_pred]
            }
            
            results.append(result)
            
            # Filter high confidence results
            if avg_confidence >= confidence_threshold and consistency >= 0.8:
                high_confidence_results.append(result)
        
        # Print summary
        print(f"\n=== Batch Test Results ===")
        print(f"Total images processed: {len(results)}")
        print(f"High confidence results: {len(high_confidence_results)}")
        
        drowsy_count = sum(1 for r in results if r['prediction'] == 0)
        non_drowsy_count = sum(1 for r in results if r['prediction'] == 1)
        
        print(f"Drowsy: {drowsy_count} ({drowsy_count/len(results)*100:.1f}%)")
        print(f"Non Drowsy: {non_drowsy_count} ({non_drowsy_count/len(results)*100:.1f}%)")
        
        # High confidence results
        if high_confidence_results:
            print(f"\n=== High Confidence Results (≥{confidence_threshold}) ===")
            for result in high_confidence_results:
                print(f"{result['image']}: {result['class_name']} (conf: {result['confidence']:.3f}, consistency: {result['consistency']:.2f})")
        
        return results, high_confidence_results
    
    def test_video_with_sliding_window(self, video_path, window_size=10, confidence_threshold=0.6):
        """Test video with sliding window analysis for more accurate results."""
        print(f"\n=== Video Analysis with Sliding Window ===")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Sliding window for predictions
        prediction_window = deque(maxlen=window_size)
        confidence_window = deque(maxlen=window_size)
        
        frame_results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Classify frame
            if self.use_yolo:
                pred_class, conf = self.classify_image_yolo(frame)
            else:
                pred_class, conf = self.classify_image_mobilenet(frame)
            
            # Add to sliding window
            prediction_window.append(pred_class)
            confidence_window.append(conf)
            
            # Calculate window statistics
            if len(prediction_window) == window_size:
                window_avg_confidence = np.mean(confidence_window)
                most_common_pred = max(set(prediction_window), key=list(prediction_window).count)
                prediction_consistency = list(prediction_window).count(most_common_pred) / window_size
                
                # Only make decision if confidence and consistency are high enough
                if window_avg_confidence >= confidence_threshold and prediction_consistency >= 0.7:
                    final_prediction = most_common_pred
                    is_confident = True
                else:
                    final_prediction = pred_class  # Use current frame prediction
                    is_confident = False
                
                result = {
                    'frame': frame_count,
                    'prediction': final_prediction,
                    'confidence': window_avg_confidence,
                    'consistency': prediction_consistency,
                    'is_confident': is_confident,
                    'class_name': self.classes[final_prediction]
                }
                frame_results.append(result)
                
                # Display frame with results
                color = (0, 255, 0) if final_prediction == 1 else (0, 0, 255)
                cv2.putText(frame, f"{self.classes[final_prediction]} ({window_avg_confidence:.3f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Consistency: {prediction_consistency:.2f}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Confident: {'Yes' if is_confident else 'No'}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0) if is_confident else (0, 255, 255), 2)
                
                cv2.imshow('Video Analysis', frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print video analysis results
        print(f"\n=== Video Analysis Results ===")
        print(f"Total frames analyzed: {len(frame_results)}")
        
        confident_results = [r for r in frame_results if r['is_confident']]
        drowsy_frames = sum(1 for r in confident_results if r['prediction'] == 0)
        non_drowsy_frames = sum(1 for r in confident_results if r['prediction'] == 1)
        
        print(f"Confident predictions: {len(confident_results)}")
        print(f"Drowsy frames: {drowsy_frames} ({drowsy_frames/len(confident_results)*100:.1f}%)" if confident_results else "Drowsy frames: 0")
        print(f"Non Drowsy frames: {non_drowsy_frames} ({non_drowsy_frames/len(confident_results)*100:.1f}%)" if confident_results else "Non Drowsy frames: 0")
        
        return frame_results
    
    def test_webcam_with_analysis(self, confidence_threshold=0.7, analysis_interval=30):
        """Test webcam with periodic detailed analysis."""
        print(f"\n=== Webcam Analysis Mode ===")
        print("Press 'q' to quit, 'a' for instant analysis")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        frame_count = 0
        analysis_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Basic classification
            if self.use_yolo:
                pred_class, conf = self.classify_image_yolo(frame)
            else:
                pred_class, conf = self.classify_image_mobilenet(frame)
            
            # Display basic result
            color = (0, 255, 0) if pred_class == 1 else (0, 0, 255)
            cv2.putText(frame, f"{self.classes[pred_class]} ({conf:.3f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Periodic detailed analysis
            if frame_count % analysis_interval == 0:
                analysis_count += 1
                print(f"\n--- Analysis #{analysis_count} (Frame {frame_count}) ---")
                
                # Multiple predictions for accuracy
                predictions = []
                confidences = []
                
                for _ in range(5):
                    if self.use_yolo:
                        p, c = self.classify_image_yolo(frame)
                    else:
                        p, c = self.classify_image_mobilenet(frame)
                    predictions.append(p)
                    confidences.append(c)
                    time.sleep(0.1)
                
                avg_conf = np.mean(confidences)
                consistency = predictions.count(max(set(predictions), key=predictions.count)) / len(predictions)
                
                print(f"Predictions: {predictions}")
                print(f"Average Confidence: {avg_conf:.3f}")
                print(f"Consistency: {consistency:.2f}")
                
                if avg_conf >= confidence_threshold and consistency >= 0.8:
                    print("✅ HIGH CONFIDENCE RESULT")
                else:
                    print("⚠️  LOW CONFIDENCE - Need more analysis")
            
            cv2.imshow('Webcam Analysis', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                # Instant analysis
                print(f"\n--- Instant Analysis (Frame {frame_count}) ---")
                predictions = []
                confidences = []
                
                for _ in range(3):
                    if self.use_yolo:
                        p, c = self.classify_image_yolo(frame)
                    else:
                        p, c = self.classify_image_mobilenet(frame)
                    predictions.append(p)
                    confidences.append(c)
                
                avg_conf = np.mean(confidences)
                consistency = predictions.count(max(set(predictions), key=predictions.count)) / len(predictions)
                print(f"Predictions: {predictions}")
                print(f"Average Confidence: {avg_conf:.3f}")
                print(f"Consistency: {consistency:.2f}")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal frames processed: {frame_count}")
        print(f"Detailed analyses performed: {analysis_count}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Drowsiness Model Testing')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['single', 'batch', 'video', 'webcam'],
                       help='Testing mode')
    parser.add_argument('--input', type=str, required=True,
                       help='Input path (image, folder, or video)')
    parser.add_argument('--yolo-weights', type=str, 
                       default='src/yolo/runs/classify/drowsiness_detection/weights/best.pt',
                       help='Path to YOLOv8 weights')
    parser.add_argument('--mobilenet-weights', type=str,
                       default='src/MobileNetv2/outputs/best_mobilenetv2.pth',
                       help='Path to MobileNetV2 weights')
    parser.add_argument('--use-yolo', action='store_true', default=True,
                       help='Use YOLOv8 model (default: True)')
    parser.add_argument('--use-mobilenet', action='store_true', default=False,
                       help='Use MobileNetV2 model instead of YOLOv8')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                       help='Confidence threshold for filtering results')
    parser.add_argument('--window-size', type=int, default=10,
                       help='Sliding window size for video analysis')
    
    args = parser.parse_args()
    
    # Determine which model to use
    use_yolo = args.use_yolo and not args.use_mobilenet
    
    # Check if model weights exist
    if use_yolo:
        if not os.path.exists(args.yolo_weights):
            print(f"Error: YOLOv8 weights not found at {args.yolo_weights}")
            return
    else:
        if not os.path.exists(args.mobilenet_weights):
            print(f"Error: MobileNetV2 weights not found at {args.mobilenet_weights}")
            return
    
    # Initialize tester
    tester = DrowsinessTester(
        yolo_weights_path=args.yolo_weights,
        mobilenet_weights_path=args.mobilenet_weights if not use_yolo else None,
        use_yolo=use_yolo
    )
    
    # Run appropriate test
    if args.mode == 'single':
        tester.test_single_image(args.input)
    elif args.mode == 'batch':
        tester.test_batch_images(args.input, args.confidence_threshold)
    elif args.mode == 'video':
        tester.test_video_with_sliding_window(args.input, args.window_size, args.confidence_threshold)
    elif args.mode == 'webcam':
        tester.test_webcam_with_analysis(args.confidence_threshold)

if __name__ == "__main__":
    main() 