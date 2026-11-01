import os
import time
import platform
import numpy as np
import cv2
import torch
import torch.nn as nn
import gradio as gr
from torchvision import models, transforms

CLASSES = ["Books", "Cups", "Pens", "Spoons"]
MODEL_PATH = os.environ.get("MODEL_PATH", "resnet18_sota.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
ENV = f"{platform.system()} | {gpu} | torch {torch.__version__}"

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

times = []


def center_crop(img, ratio=0.6):
    w, h = img.size
    nw, nh = int(w * ratio), int(h * ratio)
    left, top = (w - nw) // 2, (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh))


def cam(x):
    store = {}

    def fh(m, i, o):
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))

    h = model.layer4.register_forward_hook(fh)
    out = model(x)
    c = out.argmax(1).item()
    model.zero_grad()
    out[0, c].backward()
    h.remove()
    w = store["g"].mean(dim=(2, 3), keepdim=True)
    m = torch.relu((w * store["a"]).sum(1)).squeeze(0).detach().cpu().numpy()
    return m / (m.max() + 1e-8)


def overlay(img, m):
    img = np.array(img.resize((448, 448))) / 255.0
    h = cv2.resize(m, (448, 448))
    h = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB) / 255.0
    return ((0.5 * img + 0.5 * h).clip(0, 1) * 255).astype(np.uint8)


def run(img, crop):
    if img is None:
        return None, None, "Upload or capture an image first."
    img = img.convert("RGB")
    if crop:
        img = center_crop(img)
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        p = model(x).softmax(1)[0].cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
    times.append((time.time() - t) * 1000)
    heat = overlay(img, cam(x))
    info = f"Latency this run: {times[-1]:.1f} ms | average over {len(times)} runs: {np.mean(times):.1f} ms\n\nEnvironment: {ENV}"
    return {c: float(v) for c, v in zip(CLASSES, p)}, heat, info


NOTE = """**Synthetic media notice.** The model was trained on my own photos and on AI-generated video frames
(Step-Video-T2V, offline). Generated frames are synthetic and contain no people or faces.
Do not upload photos of people. Images are processed in memory and are not stored."""

with gr.Blocks() as demo:
    gr.Markdown("# Household object recognition: books, cups, pens, spoons")
    gr.Markdown(NOTE)
    with gr.Row():
        with gr.Column():
            inp = gr.Image(sources=["upload", "webcam"], type="pil", label="Image")
            crop = gr.Checkbox(value=True, label="Center-crop to 60% (same as training)")
            btn = gr.Button("Predict")
        with gr.Column():
            lab = gr.Label(num_top_classes=4, label="Prediction")
            heat = gr.Image(label="Grad-CAM (where the model looks)")
            info = gr.Markdown()
    btn.click(run, [inp, crop], [lab, heat, info])

if __name__ == "__main__":
    demo.launch()
