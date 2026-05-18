# AI Image Detection and Feature Extraction

This project focuses on detecting and analyzing features in AI-generated and real artwork images using machine learning techniques.

## Dataset

The dataset used for this project is available at:
[Dataset Link](https://drive.google.com/file/d/1MOARwVUpO2J_QhEN_ubPCU5f5OwJdIMJ/view?usp=sharing)

## Project Structure

```
├── App/                          # Application and model files
│   ├── app.py                   # Main application
│   ├── extract_features.py      # Feature extraction script
│   ├── model_rf.joblib          # Trained random forest model
│   ├── scaler.joblib            # Feature scaler
│   ├── features_extracted.csv   # Extracted features
│   └── asset/                   # Application assets
├── Module/                       # Core modules
│   ├── color_features.py        # Color feature extraction
│   ├── hog_feature.py           # HOG (Histogram of Oriented Gradients) features
│   ├── dataprocessing.py        # Data processing utilities
│   └── dataset.csv              # Dataset configuration
├── Data/                         # Data directory
│   ├── ai_images/               # AI-generated images
│   ├── real_art/                # Real artwork images
│   └── features_extracted.csv   # Extracted features
└── run.bat                       # Batch script to run the application
```

## Features

- Color-based feature extraction
- HOG (Histogram of Oriented Gradients) feature extraction
- Machine learning model for classification
- Data preprocessing and normalization

## Usage

Run the application using:
```bash
run.bat
```

Or execute the main application directly:
```bash
cd app
streamlit run app.py
```

## Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
- **numpy** - Numerical computing
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning library
- **opencv-python** - Computer vision (HOG features, image processing)
- **joblib** - Model serialization
- **streamlit** - Web application framework (optional, for UI)
- **Pillow** - Image processing

### Install Dependencies

Install all required packages using pip:

```bash
pip install numpy pandas scikit-learn opencv-python joblib streamlit pillow
```

Or if a requirements.txt file is provided:

```bash
pip install -r requirements.txt
```
