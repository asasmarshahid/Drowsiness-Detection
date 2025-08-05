# Driver Drowsiness Detection 🚗💤

Real-time detection using **YOLOv8**, **MobileNetV2** (image classification).  
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
│   │   └── runs/             # YOLOv8 training outputs (metrics, weights, plots)
│   └── MobileNetv2/
│       ├── train_mobilenetv2.py # MobileNetV2 training script
│       └── outputs/             # MobileNetV2 training outputs (metrics, weights, plots)
├── requirements.txt
└── README.md
```

---

## 🏆 Model Training & Evaluation

### 1. YOLOv8 (Object Detection)

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
  - Model weights: `src/MobileNetv2/outputs/best_mobilenetv2.pth`, `last_mobilenetv2.pth`
  - Metrics & plots: `src/MobileNetv2/outputs/` (loss/accuracy curves, confusion matrix, classification report)

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

## 🧪 **Accurate Model Testing**

For more accurate results, use the comprehensive testing script:

### **1. Single Image Testing**

```bash
python src/test_models.py --mode single --input path/to/image.jpg
```

### **2. Batch Image Testing**

```bash
python src/test_models.py --mode batch --input path/to/image/folder --confidence-threshold 0.7
```

### **3. Video Analysis with Sliding Window**

```bash
python src/test_models.py --mode video --input path/to/video.mp4 --window-size 15 --confidence-threshold 0.6
```

### **4. Webcam Analysis Mode**

```bash
python src/test_models.py --mode webcam --input dummy
```

### **Testing Features:**

#### **🔍 Single Image Analysis:**

- Multiple predictions per image (5 runs)
- Confidence statistics (mean, std dev)
- Prediction consistency checking
- Visual result display

#### **📁 Batch Testing:**

- Process entire folders of images
- Confidence filtering (≥0.7 default)
- Consistency checking (≥80% agreement)
- Summary statistics and high-confidence results

#### **🎬 Video Analysis:**

- Sliding window approach (10-15 frames)
- Temporal consistency checking
- Confidence-based decision making
- Real-time display with confidence indicators

#### **📹 Webcam Analysis:**

- Periodic detailed analysis (every 30 frames)
- Instant analysis on demand (press 'a')
- Multiple prediction averaging
- Confidence threshold filtering

#### **📊 Accuracy Improvements:**

- **Face Detection**: Uses dlib to detect and extract faces from images/videos
- **Face Extraction**: Automatically crops and processes detected faces
- **Multiple Predictions**: Each face image classified multiple times
- **Consistency Checking**: Only accept results with high agreement
- **Confidence Filtering**: Filter out low-confidence predictions
- **Temporal Smoothing**: Use sliding windows for video analysis
- **Statistical Validation**: Calculate confidence intervals and consistency scores

#### **🔍 Face Detection Features:**

- **Automatic Face Detection**: Uses dlib's frontal face detector
- **Face Cropping**: Extracts face regions with padding for better classification
- **Multiple Face Support**: Can detect and process multiple faces in an image
- **Largest Face Selection**: Automatically selects the largest face as the main subject
- **Visual Feedback**: Shows face bounding boxes and detection results

---

## Contributing

Feel free to submit issues and enhancement requests!
