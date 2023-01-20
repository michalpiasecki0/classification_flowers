import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import conf.train_config as config

from typing import Tuple
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose


def get_dataloader(
    root_dir: str,
    transforms: Compose,
    target_transform: Compose,
    batch_size: int,
    shuffle: bool = True,
) -> Tuple[ImageFolder, DataLoader]:
    """
    Get torch ImageFolder and Dataloader from given root directory. Folder must follow ImageFolder structure
    """
    ds = ImageFolder(
        root=root_dir, transform=transforms, target_transform=target_transform
    )
    dataloader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )
    return ds, dataloader


def save_plot(history: dict, key: str, output_path: str):
    """
    Plot and save history of parameter during model learning.
    """
    if key not in ["acc", "loss"]:
        print(
            "Invalid key type, no image will be saved. Please choose from [acc, loss]"
        )
    else:
        try:
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(history[f"train_{key}"], label=f"train_{key}")
            plt.plot(history[f"val_{key}"], label=f"val_{key}")
            plt.xlabel("Epoch #")
            plt.ylabel(key)
            plt.legend(loc="lower left")
            plt.savefig(os.path.join(config.SAVE_MODEL_PATH, output_path))
            print("Image correctly saved.")
        except FileNotFoundError:
            print("Path does not exist.")


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess image for model classification.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    image = image.astype("float32") / 255.0

    image -= config.MEAN
    image /= config.STD
    image = np.transpose(image, (2, 0, 1))

    return torch.from_numpy(image)[None]
