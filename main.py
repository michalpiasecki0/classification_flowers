"""Script for interference via FastAPI"""
import os
import torch
import pathlib

from typing import Union
from enum import Enum
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from src.predict import predict


class ModelName(str, Enum):
    BACKBONE_FROZEN = "backbone_frozen"
    BACKBONE_UPDATED = "backbone_updated"


class Device(str, Enum):
    CPU = "cpu"
    GPU = "cuda"


IMAGE_FOLDER = "./data/images_fast_api"
app = FastAPI()


@app.post("/images/upload/")
async def upload_file(
    file: UploadFile = File(
        ..., description="File is saved to folder with photos for interference."
    ),
    name: Union[None, str] = Query(
        default=None,
        description="Optional image name in interference directory. If not "
        "provided, same name as uploaded will be taken.",
    ),
):
    """
    Upload new image, which will be ready for interference.
    """

    contents = await file.read()
    suffix = pathlib.Path(file.filename).suffix

    if suffix not in [".png", ".jpg", ".jpeg", ".bmp"]:
        raise HTTPException(
            status_code=501,
            detail="Invalid file extension. Only (png, jpg, jpeg, bmp) are accepted",
        )
    if name and name.find(".") != -1:
        raise HTTPException(
            status_code=501, detail="Please do not provide extension for new image name"
        )
    save_name = (name + suffix) if name else file.filename
    with open(os.path.join(IMAGE_FOLDER, save_name), "wb") as f:
        f.write(contents)

    return {f"{save_name} was successfully saved in folder."}


@app.delete("/images/delete/")
async def delete_file(file_name: str):
    """
    Delete file from folder with images for interference.
    """
    if file_name not in os.listdir(IMAGE_FOLDER):
        return {"No such file in folder. Nothing was deleted"}

    os.remove(os.path.join(IMAGE_FOLDER, file_name))

    return {f"Successfully deleted {file_name} image from interference folder."}


@app.get("/images/list/")
async def get_all_images_uploaded():
    """
    List all filenames, which are ready for interference.
    """

    return {"filenames": os.listdir(IMAGE_FOLDER)}


# maybe q: Union[List[str], None] = Query(default=None) for many images?
@app.get("/infer/")
async def get_prediction(
    image_name: str = Query(
        default=None, description="Image name located in folder for interference."
    ),
    device: Device = Query(
        default=Device.CPU, description="Type on device for interference."
    ),
    model_name: ModelName = Query(
        default=ModelName.BACKBONE_FROZEN, description="Model to use for interference."
    ),
):
    """
    Get prediction on image.
    Class name and model confidence is returned.
    """

    image_path = os.path.join(IMAGE_FOLDER, image_name)

    if device == Device.GPU and not torch.cuda.is_available():
        raise HTTPException(
            status_code=501, detail="Not possible to use CUDA on this device."
        )
    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404,
            detail="Such image was not placed in folder for predictions.",
        )

    return {predict(device=device, image_path=image_path, model_name=model_name)}
