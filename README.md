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

---

## Contributing

Feel free to submit issues and enhancement requests!
