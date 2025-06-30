from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
import os

def train_model(
    model_name="yolov8n-cls.pt",  # Using nano model for faster training
    epochs=20,
    imgsz=224,
    batch=64,  # Increased batch size
    device="cuda" if torch.cuda.is_available() else "cpu",
    project="runs/classify",
    name="drowsiness_detection",
):
    # Print training configuration
    print(f"Training configuration:")
    print(f"- Model: {model_name}")
    print(f"- Device: {device}")
    print(f"- Epochs: {epochs}")
    print(f"- Image size: {imgsz}")
    print(f"- Batch size: {batch}")
    
    try:
        model = YOLO(model_name)
        
        results = model.train(
            data="./datasets/dataset_yolo",
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            pretrained=True,
            optimizer="AdamW",
            lr0=0.001,  # Initial learning rate
            lrf=0.01,   # Final learning rate (fraction of lr0)
            warmup_epochs=3.0,
            patience=5,  # Early stopping patience
            save=True,   # Save checkpoints
            plots=True,  # Save training plots
            amp=True,    # Automatic mixed precision
            augment=True,  # Enable default augmentations
            mosaic=1.0,  # Enable mosaic augmentation
            mixup=0.1,   # Enable mixup augmentation
            copy_paste=0.1,  # Enable copy-paste augmentation
            degrees=10.0,  # Random rotation augmentation
            translate=0.1,  # Random translation augmentation
            scale=0.5,   # Random scaling augmentation
            shear=0.5,   # Random shear augmentation
            fliplr=0.5,  # Horizontal flip probability
            flipud=0.0,  # No vertical flips for face images
            hsv_h=0.015, # HSV hue augmentation
            hsv_s=0.7,   # HSV saturation augmentation
            hsv_v=0.4,   # HSV value augmentation
            auto_augment="randaugment",  # Use RandAugment policy
            erasing=0.1,  # Random erasing probability
            cache=True,  # Cache images for faster training
            workers=8,   # Number of worker threads for data loading
        )
        print("Training completed successfully!")
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print("\nFinal metrics:")
            train_acc = metrics.get('metrics/accuracy_top1_top1', None)
            val_acc = metrics.get('metrics/val/accuracy_top1_top1', None)
            
            if train_acc is not None:
                print(f"- Training accuracy: {float(train_acc):.3f}")
            else:
                print("- Training accuracy: Not available")
                
            if val_acc is not None:
                print(f"- Validation accuracy: {float(val_acc):.3f}")
            else:
                print("- Validation accuracy: Not available")
        else:
            print("\nNo metrics available in results")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_model() 