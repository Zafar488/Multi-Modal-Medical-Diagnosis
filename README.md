# 🏥 Multi-Modal Medical Diagnosis Assistant

An advanced AI-powered medical diagnosis system combining Computer Vision, Deep Learning, and NLP for comprehensive healthcare analysis.

## 🌟 Features

### 🩻 X-Ray Analysis
- Detect pneumonia from chest X-rays
- CNN-based classification
- Confidence scoring and visualization

### 🩺 Skin Cancer Detection
- Benign vs Malignant classification
- Transfer learning with MobileNetV2
- High accuracy skin lesion analysis

### 🤒 Symptom Checker
- 10+ symptoms supported
- Random Forest classification
- 6+ disease predictions
- Probability distribution visualization

### 💬 AI Chatbot
- Context-aware medical consultation
- Symptom extraction and analysis
- Emergency keyword detection
- Natural conversation flow

### 📄 Report Generation
- Professional PDF medical reports
- Comprehensive analysis summary
- Risk assessment and recommendations
- Downloadable reports

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow, Keras, MobileNetV2
- **Machine Learning**: Scikit-learn, Random Forest
- **Computer Vision**: OpenCV, PIL
- **Frontend**: Streamlit
- **Reporting**: ReportLab
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib

## 📁 Project Structure
medical-diagnosis-assistant/
├── datasets/
│ ├── chest_xray/ # Chest X-Ray dataset
│ ├── skin_cancer/ # Skin cancer dataset
│ └── symptoms/ # Symptom dataset
├── models/ # Trained models
├── training/ # Training scripts
│ ├── train_xray.py
│ ├── train_skin.py
│ └── train_symptom.py
├── app/ # Application code
│ ├── streamlit_app.py # Main UI
│ ├── predictor.py # Prediction engine
│ ├── report_generator.py # PDF generation
│ ├── chatbot.py # AI chatbot
│ └── utils.py # Utilities
├── reports/ # Generated reports
├── config/ # Configuration
├── deploy.py # Deployment script
└── requirements.txt # Dependencies

text

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/medical-diagnosis-assistant.git
cd medical-diagnosis-assistant