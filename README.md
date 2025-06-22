# Driver Drowsiness Detection 🚗💤

Real-time detection using **YOLOv8** (face/eye detection) + **MobileNetV2**.  
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

# Driver Drowsiness Detection - Exploratory Data Analysis

This project performs comprehensive Exploratory Data Analysis (EDA) on a driver drowsiness detection dataset containing images classified as "drowsy" and "non-drowsy".

## Project Structure

```
Driver Drowsiness Detection/
├── dataset/               # Dataset directory (DVC tracked)
│   ├── drowsy/           # Drowsy class images
│   └── non-drowsy/       # Non-drowsy class images
├── src/                  # Source code
│   └── analysis/         # Analysis scripts
│       └── eda.py        # Main EDA script
├── data/                 # Generated data
│   └── plots/           # Generated plots and visualizations
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Setup and Installation

1. Create a virtual environment (recommended):

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Pull the dataset (if using DVC):
   ```bash
   dvc pull
   ```

## Running the Analysis

To run the full EDA:

```bash
python src/analysis/eda.py
```

This will:

1. Analyze class distribution
2. Examine image properties (dimensions, channels, formats)
3. Analyze pixel intensities
4. Check for quality issues (corrupted images, duplicates)
5. Generate visualizations
6. Create a comprehensive report

## Generated Outputs

The script generates the following outputs in the `data/plots` directory:

- `class_distribution.png`: Distribution of images across classes
- `image_properties.png`: Image dimension and aspect ratio distributions
- `pixel_intensities.png`: RGB channel distributions
- `sample_images.png`: Sample images from each class
- `augmentations.png`: Example augmentation techniques
- `eda_report.txt`: Comprehensive analysis report

## Analysis Features

The EDA script performs the following analyses:

1. **Class Distribution Analysis**

   - Counts images in each class
   - Visualizes class balance
   - Calculates class balance ratio

2. **Image Property Analysis**

   - Image dimensions
   - Color channels
   - File formats
   - Aspect ratios

3. **Pixel Intensity Analysis**

   - RGB channel distributions
   - Comparison between classes

4. **Quality Checks**

   - Identifies corrupted images
   - Detects duplicate images
   - Verifies file integrity

5. **Visualization Examples**
   - Sample images from each class
   - Common augmentation techniques

## Contributing

Feel free to submit issues and enhancement requests!
