import io, os, time, platform, threading, base64
from pathlib import Path
from contextlib import asynccontextmanager
import numpy as np
import cv2
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'models' / 'resnet18.pth'
CLASSES = ['Books', 'Cups', 'Pens', 'Spoons']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = None
ERROR = None
LOCK = threading.Lock()
TIMES = []
PREPROCESS = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def load_model():
    global MODEL, ERROR
    try:
        net = models.resnet18(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features,4)
        state = torch.load(WEIGHTS,map_location='cpu',weights_only=True)
        if isinstance(state,dict) and 'state_dict' in state: state = state['state_dict']
        if isinstance(state,dict) and all(k.startswith('module.') for k in state): state={k[7:]:v for k,v in state.items()}
        net.load_state_dict(state,strict=True)
        MODEL=net.to(DEVICE).eval()
    except Exception as e: ERROR=f'{type(e).__name__}: {e}'

@asynccontextmanager
async def lifespan(app):
    load_model()
    yield

app=FastAPI(title='Object Atlas API', lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_credentials=False,allow_methods=['GET','POST'],allow_headers=['*'])

@app.get('/api/health')
def health():
    return {'ready':MODEL is not None,'error':ERROR,'device':str(DEVICE),'hardware':torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else platform.processor() or 'CPU','torch':torch.__version__,'weights':WEIGHTS.name,'average_latency_ms':round(float(np.mean(TIMES)),2) if TIMES else None,'runs':len(TIMES)}

def image_data_uri(img):
    b=io.BytesIO(); img.save(b,format='PNG'); return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode('ascii')

def predict_sync(image, crop):
    if MODEL is None: raise HTTPException(503, detail=ERROR or 'Model not loaded')
    image=ImageOps.exif_transpose(image).convert('RGB')
    if crop:
        w,h=image.size; nw,nh=int(w*.6),int(h*.6); image=image.crop(((w-nw)//2,(h-nh)//2,(w+nw)//2,(h+nh)//2))
    x=PREPROCESS(image).unsqueeze(0).to(DEVICE)
    with LOCK:
        with torch.no_grad():
            MODEL(x) # warm-up
            if DEVICE.type=='cuda':torch.cuda.synchronize()
            start=time.perf_counter(); logits=MODEL(x)
            if DEVICE.type=='cuda':torch.cuda.synchronize()
            ms=(time.perf_counter()-start)*1000
            probs=torch.softmax(logits,dim=1)[0].cpu().numpy().tolist()
        target=int(np.argmax(probs))
        saved={}
        def hook(module, inputs, output):
            saved['activations']=output
            output.register_hook(lambda grad:saved.__setitem__('gradients',grad))
        handle=MODEL.layer4.register_forward_hook(hook)
        try:
            with torch.enable_grad():
                logits=MODEL(x); MODEL.zero_grad(set_to_none=True); logits[0,target].backward()
            weights=saved['gradients'].mean(dim=(2,3),keepdim=True)
            cam=torch.relu((weights*saved['activations']).sum(dim=1))[0].detach().cpu().numpy()
        finally:handle.remove()
        cam=cam/(cam.max()+1e-8)
        TIMES.append(ms)
        if len(TIMES)>200:TIMES.pop(0)
    size=(448,448)
    rgb=np.asarray(image.resize(size),dtype=np.uint8)
    heat=cv2.applyColorMap(np.uint8(cv2.resize(cam,size)*255),cv2.COLORMAP_TURBO)
    heat=cv2.cvtColor(heat,cv2.COLOR_BGR2RGB)
    blend=np.uint8(np.clip(rgb*.52+heat*.48,0,255))
    return {'label':CLASSES[target],'confidence':round(probs[target]*100,2),'probabilities':[{'name':n,'value':round(float(p)*100,2)} for n,p in zip(CLASSES,probs)],'latency_ms':round(ms,2),'average_latency_ms':round(float(np.mean(TIMES)),2),'device':str(DEVICE),'original':image_data_uri(Image.fromarray(rgb)),'heatmap':image_data_uri(Image.fromarray(heat)),'overlay':image_data_uri(Image.fromarray(blend))}

@app.post('/api/predict')
async def predict(file:UploadFile=File(...),crop:bool=Form(True)):
    data=await file.read()
    if len(data)>12*1024*1024:raise HTTPException(413,detail='Image must be under 12 MB')
    try:
        im=Image.open(io.BytesIO(data)); im.verify(); im=Image.open(io.BytesIO(data))
        if im.width*im.height>25_000_000:raise HTTPException(413,detail='Image dimensions too large')
    except (UnidentifiedImageError,OSError,ValueError):raise HTTPException(400,detail='Please upload a valid image')
    return predict_sync(im,crop)
