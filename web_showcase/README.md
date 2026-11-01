# Object Atlas — Assignment 2 Computer Vision Showcase

A full-width React + TypeScript research dashboard with a real FastAPI/PyTorch backend, trained ResNet18 weights, image upload, webcam capture, Grad-CAM, dataset exploration and experiment charts.

## Requirements
- Windows 10/11
- Python **3.11** (avoid Python 3.15 due to wheel compatibility)
- Node.js LTS (includes npm)

## Quick start
1. Extract the ZIP, open `web_showcase` in VS Code.
2. Double-click `start_windows.bat`. Two terminal windows will open. On first run, installation can take a few minutes.
3. Visit **http://127.0.0.1:5173** (not port 7860; this is a real React website).
4. Wait until the top bar shows **Inference ready**.

## Manual start in VS Code PowerShell
Open **two separate terminals**.

**Terminal 1 (backend)**
```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 (frontend)**
```powershell
cd frontend
npm install
npm run dev
```

Visit http://127.0.0.1:5173. The frontend proxies `/api` to http://127.0.0.1:8000.

## Trained model
`backend/models/resnet18.pth` is the user-supplied 4-class ResNet18 weights file. This app does not substitute fake predictions. Backend `/api/health` reports whether the weights loaded. Inference runs on CPU or CUDA if available.

## Method and results
- Classes: Books, Cups, Pens, Spoons
- Real dataset: 405 photos, with 283 train / 60 validation / 62 test
- Synthetic augmentation: 16 Wan2.1-T2V-1.3B clips → 64 extracted frames
- Models compared: 0%, 10.2%, 18.4% synthetic share
- Clean test accuracy: 100%, 100%, 98.39% respectively
- Low-light stress test: approximately 89% baseline, 100% with synthetic frames
- Grad-CAM: last ResNet18 convolutional block (`layer4`)

The training convergence chart is **explicitly labeled illustrative**, not raw logged values. Other exact stress-condition values are omitted because raw `stress.csv` was not supplied. Reported GPU latency metrics are **not** confused with live local inference latency.

## Privacy and limitations
Uploads are processed in RAM and not persisted. The app is designed for object images only. The test set is small, some photos may be near duplicates, and one training run per variant limits conclusions. Grad-CAM is an approximate visual explanation.

## API
- `GET /api/health` model status, runtime info
- `POST /api/predict` image multipart + crop flag; prediction probabilities, Grad-CAM and latency

Original project: https://github.com/ETairkhan/compvision2
