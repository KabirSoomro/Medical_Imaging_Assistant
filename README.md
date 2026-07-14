# 🩺 Medical Imaging Assistant

An AI-powered diagnostic support system for automatic detection of abnormalities in radiological images (X-rays, MRI, and CT scans).

---

## 🎯 Problem Statement
Medical imaging analysis is critical for early disease detection, but radiologists face overwhelming workloads and subtle abnormalities can be missed.

---

## 💡 Solution
A Streamlit-based web application that:
1. Accepts medical images (X-ray, MRI, CT scans)
2. Processes them using trained AI models (CNN & VGG16)
3. Detects abnormalities with high accuracy
4. Generates structured diagnostic reports
5. Provides visual explanations for decisions

---

## 🤖 Models
- **CNN:** 74.30% accuracy (custom architecture)
- **VGG16:** 98.18% accuracy (transfer learning)

---

## 📥 Download Trained Models

⚠️ The model files are **not included** in this repository due to GitHub's file size limits.

### Download from Google Drive:
[📁 Download Models from Google Drive](https://drive.google.com/drive/folders/1QmEp0Q2xFP3kmLSP64NBhmuFIG5MxYTI?usp=drive_link)

### Files Included:
| File | Size | Description |
|------|------|-------------|
| `cnn_model.keras` | 6 MB | CNN model (74.30% accuracy) |
| `vgg16_finetuned.keras` | 159.8 MB | VGG16 model (98.18% accuracy) |

### After Download:
Place both files in the `models/` folder:
```text
models/
├── cnn_model.keras
└── vgg16_finetuned.keras
```

---

## 🛠️ Installation

```bash
# 1. Clone repository
git clone https://github.com/KabirSoomro/Medical_Imaging_Assistant.git
cd Medical_Imaging_Assistant

# 2. Create virtual environment (Python 3.11 recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download models from Google Drive and place in models/ folder

# 5. Run application
streamlit run app.py
```

---

## 📱 Features
- Image upload (JPG, PNG, DICOM)
- Model selection (CNN vs VGG16)
- Real-time prediction with confidence scores
- Interactive visualizations (charts, gauges)
- Natural language explanations
- Model comparison
- Analysis history tracking

---

## 📊 Results

### Model Performance Comparison
| Model | Accuracy | Sensitivity | Specificity |
|-------|----------|-------------|-------------|
| CNN   | 74.30%   | 72.15%      | 76.45%      |
| VGG16 | 98.18%   | 97.89%      | 98.47%      |

### Test Results on Sample Images
| Image | Prediction | Confidence | Status |
|-------|------------|------------|--------|
| Normal X-ray | Normal | 99.8% | ✅ Healthy |
| Pneumonia X-ray | Pneumonia | 94.3% | ⚠️ Abnormal |

---

## 🖥️ UI Screenshots
| Feature | Screenshot |
|---------|------------|
| Home Page | `screenshots/home.png` |
| Report Tab | `screenshots/report.png` |
| Visualizations | `screenshots/visuals.png` |
| Explainability | `screenshots/explain.png` |
| Comparison | `screenshots/compare.png` |
| History | `screenshots/history.png` |

---

## ⚠️ Disclaimer
This AI system provides supportive analysis only. Final diagnosis must be made by a qualified medical professional. The information provided is for educational and decision-support purposes only.

---

## 👥 Team
- **Student:** Ghulam Kabir
- **Roll Number:** 2K23/CSM/40
- **Course:** Introduction to Artificial Intelligence
- **Option:** D — Medical Imaging Assistant

---

## 📚 References
- Russell & Norvig — Artificial Intelligence: A Modern Approach, Chapter 2
- TensorFlow/Keras documentation
- Streamlit documentation

---

## 🔮 Future Work
- Add more disease classes
- Support for DICOM files natively
- Deploy on cloud (AWS/GCP)
- Mobile app development
- Real-time video analysis

---

## 📄 License
This project is for educational purposes only.
