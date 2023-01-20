from fastapi import FastAPI, HTTPException
import os
import torch
from enum import Enum
from fastapi import FastAPI
from src.predict import predict
from pathlib import Path


class ModelName(str, Enum):
    BACKBONE_FROZEN = "backbone_frozen"
    BACKBONE_UPDATED = 'backbone_updated'


class Device(str, Enum):
    CPU = 'cpu'
    GPU = 'cuda'


app = FastAPI()


@app.get("/infer/")
async def root(device: Device,
               model_name: ModelName,
               image_path: str):
    if device == Device.GPU and not torch.cuda.is_available():
        raise HTTPException(status_code=501, detail="Not possible to use CUDA on this device.")
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail='No file for given image.')
    if Path(image_path).suffix not in ['.png', '.jpg', '.jpeg', '.bmp']:
        raise HTTPException(status_code=501, detail='Invalid file extension. Only (png, jpg, jpeg, bmp) are accepted')
    return {predict(device=device,
                    image_path=image_path,
                    model_name=model_name)}