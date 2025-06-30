from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

def prepare_dataset(
    dataset_path="dataset",
    output_path="dataset_yolo",
    train_size=0.8,
    random_state=42
):

    # Create output directories
    output_base = Path(output_path)
    train_dir = output_base / "train"
    val_dir = output_base / "val"
    
    # Create class directories
    classes = ["Non Drowsy", "Drowsy"]
    for split_dir in [train_dir, val_dir]:
        for class_name in classes:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_name in classes:
        print(f"\nProcessing {class_name} class:")
        class_path = Path(dataset_path) / class_name
        
        if not class_path.exists():
            print(f"Warning: {class_path} not found!")
            continue
            
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(class_path.glob(f'**/*{ext}')))
        
        if not image_files:
            print(f"Warning: No images found in {class_path}")
            continue
            
        print(f"Found {len(image_files)} images")
        
        # Split into train and validation
        train_files, val_files = train_test_split(
            image_files,
            train_size=train_size,
            random_state=random_state,
            shuffle=True
        )
        
        # Copy and process training images
        print("Processing training images...")
        for img_path in tqdm(train_files):
            process_and_save_image(img_path, train_dir / class_name)
            
        # Copy and process validation images
        print("Processing validation images...")
        for img_path in tqdm(val_files):
            process_and_save_image(img_path, val_dir / class_name)
    
    print("\nDataset preparation completed!")
    print(f"Dataset structure:")
    print(f"- Training images:")
    for class_name in classes:
        n_train = len(list((train_dir / class_name).glob('*')))
        print(f"  - {class_name}: {n_train} images")
    print(f"- Validation images:")
    for class_name in classes:
        n_val = len(list((val_dir / class_name).glob('*')))
        print(f"  - {class_name}: {n_val} images")

def process_and_save_image(src_path, dst_dir):
    try:
        # Read image
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"Warning: Could not read {src_path}")
            return
            
        # Ensure RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Save processed image
        dst_path = dst_dir / src_path.name
        cv2.imwrite(str(dst_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
    except Exception as e:
        print(f"Error processing {src_path}: {str(e)}")

if __name__ == "__main__":
    prepare_dataset() 