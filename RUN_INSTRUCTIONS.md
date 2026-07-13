# 🚀 Quick Start Guide

## Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 2: Place Models
Place your trained models in the `models/` folder:
```
models/
├── cnn_model.h5      # CNN model (74.30% accuracy)
└── vgg16_model.h5    # VGG16 model (98.18% accuracy)
```

## Step 3: Verify Setup
```bash
python test_app.py
```

## Step 4: Run Application
```bash
streamlit run app.py
```

## Step 5: Open in Browser
The app will open automatically at: http://localhost:8501

## Sample Test Image
Upload a medical image (X-ray, MRI, or CT scan) and click "Analyze Image"

## Troubleshooting

### Error: Module not found
```
pip install -r requirements.txt
```

### Error: Model not found
Place your trained models in the `models/` folder

### Error: Port already in use
```
streamlit run app.py --server.port 8502
```

### Error: DICOM not supported
```
pip install pydicom
```

## Project Structure
```
Medical_Imaging_Assistant/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── README.md          # Documentation
├── test_app.py        # Verification script
├── models/            # Trained models
│   ├── cnn_model.h5
│   └── vgg16_model.h5
├── utils/             # Utility modules
│   ├── __init__.py
│   ├── model_loader.py
│   ├── preprocess.py
│   ├── predict.py
│   ├── visualize.py
│   └── explainability.py
├── data/              # Data files
│   └── sample_images/
├── screenshots/       # UI screenshots
└── notebooks/         # Jupyter notebooks
```

## Quick Commands

| Command | Description |
|---------|-------------|
| `streamlit run app.py` | Start the application |
| `python test_app.py` | Verify installation |
| `pip install -r requirements.txt` | Install dependencies |
| `streamlit run app.py --server.port 8502` | Run on different port |
