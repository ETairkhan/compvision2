# Computer Vision Assignment 2 — Household Object Recognition

This repository contains our Computer Vision Assignment 2 project, focused on household object recognition using deep learning and synthetic video augmentation.

The project uses a fine-tuned **ResNet18** model to classify four household object categories: **Books, Cups, Pens, and Spoons**.

We also explored synthetic data augmentation using **Wan2.1-T2V-1.3B** to evaluate its impact on model accuracy and robustness.

## Project Structure

```text
compvision2/
├── colab/
│   └── Original notebooks, experiments, datasets,
│       training scripts, and research materials
│
├── web_showcase/
│   ├── backend/
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── models/
│   ├── frontend/
│   │   ├── src/
│   │   ├── package.json
│   │   └── vite.config.ts
│   ├── README.md
│   └── start_windows.bat
│
├── .gitignore
└── README.md
```

## Technologies

**Machine Learning**
- Python
- PyTorch
- Torchvision
- ResNet18
- OpenCV
- Grad-CAM
- Wan2.1-T2V-1.3B

**Web Application**
- React
- TypeScript
- Vite
- FastAPI
- Tailwind CSS

## Dataset

The dataset consists of 405 real images and 64 synthetic frames.

| Category | Real Images | Synthetic Images |
|---|---:|---:|
| Books | 100 | 16 |
| Cups | 105 | 16 |
| Pens | 100 | 16 |
| Spoons | 100 | 16 |
| **Total** | **405** | **64** |

The real dataset was divided into training, validation, and testing subsets.

## Experimental Results

Three ResNet18 configurations were evaluated:

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Baseline | 100% | 100% |
| +32 Synthetic Frames | 100% | 100% |
| +64 Synthetic Frames | 98.39% | 98.33% |

Although synthetic augmentation did not improve clean-test accuracy, it demonstrated potential benefits for robustness under challenging lighting conditions.

## Web Showcase

The `web_showcase` directory contains an interactive full-stack web application for demonstrating the trained Computer Vision model.

Main features:

- Image upload and webcam capture
- Household object classification
- Prediction confidence scores
- Grad-CAM visual explanations
- Interactive dataset statistics
- Experimental results and model comparisons
- Research methodology dashboard

### Requirements

Install the following software:

- Python 3.11
- Node.js and npm
- Git

### Run the Backend

Open a terminal in the repository directory:

```powershell
cd web_showcase/backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

The backend API will run at:

http://localhost:8000

### Run the Frontend

Open a second terminal in
