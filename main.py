from fastapi import FastAPI


from enum import Enum
from fastapi import FastAPI, File, UploadFile
from scripts.predict import predict


class ModelName(str, Enum):
    default = "default"

class Device(str, Enum):
    CPU = 'cpu'
    GPU = 'cuda'

app = FastAPI()


@app.get("/infer/")
async def root(device: Device,
               model_name: ModelName):
    return {predict(model_name=model_name, device=device, image_path='./data/interference/example_2.jpg')}