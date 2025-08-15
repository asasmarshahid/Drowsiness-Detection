# Driver Drowsiness Detection 🚗💤

Real-time detection using **YOLOv8**, **MobileNetV2**, and **Custom CNN** for image classification.  
Data is versioned with **DVC**, stored in **Google Drive** using Google Drive API via GCP (Google Cloud Platform).  
Environment is cleanly managed via **virtualenv**.

---

## ⚙️ Quick Setup

```bash
git clone https://github.com/asasmarshahid/Drowsiness-Detection.git
cd Drowsiness-Detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

Pull data/models:

```bash
dvc pull
```

---

# Project Structure

```
Driver_Drowsiness_Detection/
├── dataset.dvc
├── outputs/
│   └── plots/                # EDA plots and reports
├── src/
│   ├── analysis/
│   │   └── eda.py            # Main EDA script
│   ├── yolo/
│   │   ├── datasets/         # YOLO dataset and config
│   │   ├── prepare_dataset.py
│   │   ├── train.py          # YOLOv8 training script
│   │   └── runs/             # YOLOv8 training outputs
│   ├── MobileNetv2/
│   │   ├── train_mobilenetv2.py # MobileNetV2 training script
│   │   └── outputs/             # MobileNetV2 training outputs
│   └── Custom_CNN/
│       ├── model.py          # Custom CNN architecture
│       ├── train.py          # Training script
│       └── outputs/          # Training outputs
├── requirements.txt
└── README.md
```

---

## 🏆 Model Training & Evaluation

### 1. YOLOv8 (Classification)

- **Train YOLOv8 classifier:**

  ```bash
  cd src/yolo
  python train.py
  ```

- **Outputs:**
  - Model weights: `src/yolo/runs/classify/drowsiness_detection/weights/`
  - Metrics & plots: `src/yolo/runs/classify/drowsiness_detection/`

### 2. MobileNetV2 (Image Classification)

- **Train MobileNetV2 classifier:**

  ```bash
  cd src/MobileNetv2
  python train_mobilenetv2.py
  ```

- **Outputs:**
  - Model weights: `src/MobileNetv2/outputs/best_mobilenetv2.pth`
  - Metrics & plots: `src/MobileNetv2/outputs/`

### 3. Custom CNN (Image Classification)

- **Train Custom CNN:**

  ```bash
  cd src/Custom_CNN
  python train.py
  ```

- **Architecture:**

  - 4 convolutional blocks with batch normalization
  - Dropout for regularization
  - Fully connected layers: 512→128→2
  - Input size: 224x224x3

- **Outputs:**
  - Model weights: `src/Custom_CNN/outputs/best_model.pth`
  - Training curves, confusion matrix
  - Classification report

---

## 📊 Exploratory Data Analysis (EDA)

To run the full EDA:

```bash
python src/analysis/eda.py
```

This will:

- Analyze class distribution
- Examine image properties (dimensions, channels, formats)
- Analyze pixel intensities
- Check for quality issues (corrupted images, duplicates)
- Generate visualizations
- Create a comprehensive report

**EDA outputs:**

- `outputs/plots/` directory (class distribution, image properties, pixel intensities, sample images, augmentations, eda_report.txt)

## 🔎 Inference (MediaPipe + Model Selection)

Run fast inference on images or videos using MediaPipe for face detection and your choice of classifier:

- YOLO (YOLOv8 classifier)
- MobileNetV2
- Custom CNN (4-layer CNN with batch normalization)

### Features:

- MediaPipe face detection and ROI extraction
- Real-time FPS display and latency metrics
- Controlled video playback (24-30 FPS)
- Face detection optimization (10 detections/sec)
- Clear visual feedback with color-coded results

### Image Inference

```bash
# YOLO
python src/inference_combined.py --mode image \
  --input path/to/image.jpg \
  --model-type yolo \
  --yolo-weights src/yolo/runs/classify/drowsiness_detection/weights/best.pt

# MobileNetV2
python src/inference_combined.py --mode image \
  --input path/to/image.jpg \
  --model-type mobilenet \
  --mobilenet-weights src/MobileNetv2/outputs/best_mobilenetv2.pth

# Custom CNN
python src/inference_combined.py --mode image \
  --input path/to/image.jpg \
  --model-type custom_cnn \
  --custom-cnn-weights src/Custom_CNN/outputs/best_model.pth
```

### Video Inference

```bash
# YOLO
python src/inference_combined.py --mode video \
  --input path/to/video.mp4 \
  --model-type yolo \
  --yolo-weights src/yolo/runs/classify/drowsiness_detection/weights/best.pt \
  --output result.mp4   # optional

# MobileNetV2
python src/inference_combined.py --mode video \
  --input path/to/video.mp4 \
  --model-type mobilenet \
  --mobilenet-weights src/MobileNetv2/outputs/best_mobilenetv2.pth \
  --output result.mp4   # optional

# Custom CNN
python src/inference_combined.py --mode video \
  --input path/to/video.mp4 \
  --model-type custom_cnn \
  --custom-cnn-weights src/Custom_CNN/outputs/best_model.pth \
  --output result.mp4   # optional
```

### Model Details:

#### YOLOv8 Classifier

- Pre-trained backbone
- Classification head fine-tuned on drowsiness data
- Fast inference with CUDA support

#### MobileNetV2

- Efficient mobile-optimized architecture
- ImageNet pre-trained weights
- Custom classification head

#### Custom CNN

- 4 convolutional blocks with batch normalization
- Dropout for regularization
- Fully connected layers: 512→128→2
- Trained from scratch on drowsiness data

### Notes:

- All models use 224x224 input size
- Class mapping: index 0 → DROWSY, index 1 → NON-DROWSY
- Preprocessing: Resize + ImageNet normalization
- Face detection runs at ~10 FPS for efficiency
- Video playback capped at 24-30 FPS

---

## Contributing

Feel free to submit issues and enhancement requests!
