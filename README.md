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
