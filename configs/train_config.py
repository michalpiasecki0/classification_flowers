import torch
import os
from pathlib import Path
from torchvision import transforms

BASE_PATH = list(Path('/home').glob('**/classification_deeptale'))

if len(BASE_PATH) != 1:
    raise Exception('Ambiguous directory location. Please make sure there is only one `classification_deeptale` folder')
else:
    BASE_PATH = BASE_PATH[0]


TRAIN_PATH = os.path.join(BASE_PATH, 'data', 'dataset', 'train')
VAL_PATH = os.path.join(BASE_PATH, 'data','dataset', 'val')
TEST_PATH = os.path.join(BASE_PATH, 'data','dataset', 'test')

SAVE_MODEL_PATH = os.path.join(BASE_PATH, 'data', 'models')


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


NORMAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ]
)
AUGMENTED_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ]
)

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ]
    )
VALIDATION_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ]
)

NUM_CLASSES = 90
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001

LABELS_DICT = {0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'}

