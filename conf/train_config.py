import torch
import os
from pathlib import Path

BASE_PATH = list(Path("/home").glob("**/classification_flowers"))

if len(BASE_PATH) != 1:
    raise Exception(
        "Ambiguous directory location. Please make sure there is only one `classification_flowers` folder"
    )
else:
    BASE_PATH = BASE_PATH[0]

# paths to train, val, test folders
TRAIN_PATH = os.path.join(BASE_PATH, "data", "dataset", "train")
VAL_PATH = os.path.join(BASE_PATH, "data", "dataset", "val")
TEST_PATH = os.path.join(BASE_PATH, "data", "dataset", "test")

SAVE_MODEL_PATH = os.path.join(BASE_PATH, "data", "models", "model_backbone_updated")

# parameters for models trained on imagenet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# data transformations

# model training parameters
NUM_CLASSES = 5
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.001

# id to flower name dictionary
LABELS_DICT = {0: "daisy", 1: "dandelion", 2: "rose", 3: "sunflower", 4: "tulip"}
