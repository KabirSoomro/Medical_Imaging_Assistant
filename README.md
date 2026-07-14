# 🩺 Medical Imaging Assistant

An AI-powered diagnostic support system for automatic detection of abnormalities in radiological images (X-rays, MRI, and CT scans).

## 🎯 Problem Statement
Medical imaging analysis is critical for early disease detection, but radiologists face overwhelming workloads and subtle abnormalities can be missed.

## 💡 Solution
A Streamlit-based web application that:
1. Accepts medical images (X-ray, MRI, CT scans)
2. Processes them using trained AI models (CNN & VGG16)
3. Detects abnormalities with high accuracy
4. Generates structured diagnostic reports
5. Provides visual explanations for decisions

## 🤖 Models
- **CNN:** 74.30% accuracy (custom architecture)
- **VGG16:** 98.18% accuracy (transfer learning)

## 🛠️ Installation

```bash
# 1. Clone repository
git clone https://github.com/KabirSoomro/Medical_Imaging_Assistant
cd Medical_Imaging_Assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place trained models in models/ folder
# models/cnn_model.h5
# models/vgg16_model.h5

# 4. Run application
streamlit run app.py
```

## 📱 Features
- Image upload (JPG, PNG, DICOM)
- Model selection (CNN vs VGG16)
- Real-time prediction with confidence scores
- Interactive visualizations (charts, gauges)
- Natural language explanations
- Model comparison
- Analysis history tracking

## 📊 Results

| Model | Accuracy | Sensitivity | Specificity |
|-------|----------|-------------|-------------|
| CNN | 74.30% | 72.15% | 76.45% |
| VGG16 | 98.18% | 97.89% | 98.47% |

## ⚠️ Disclaimer
This AI system provides supportive analysis only. Final diagnosis must be made by a qualified medical professional.

## 👥 Team
- **Student:** Ghulam Kabir
- **Roll Number:** 2K23/CSM/40
- **Course:** Introduction to Artificial Intelligence
- **Option:** D — Medical Imaging Assistant
