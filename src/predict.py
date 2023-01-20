"""
Script for interference with classification model.
"""
import argparse
import cv2
import os.path
import torch
import conf.train_config as config

from pathlib import Path
from src.utils import preprocess_image
from src.multiclass_model import MultiClassClassifier


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to the input image")
    parser.add_argument(
        "-m",
        "--model",
        choices=["backbone_frozen, backbone_updated"],
        default="backbone_frozen",
    )
    parser.add_argument("-d", "--device", choices=["cuda", "cpu"], default="cpu")
    return parser.parse_args()


def predict(image_path: str, device: str, model_name: str):

    if device not in ["cpu", "cuda"]:
        raise Exception("Invalid device type. Choose from cpu/cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise Exception(
            "Trying to use CUDA, with no active drivers. Make sure you have turned on GPU correctly."
        )
    if not os.path.exists(image_path):
        raise Exception("Image path does not exist")

    if Path(image_path).suffix not in [".jpg", ".jpeg", "png", "bmp"]:
        raise Exception(
            "Invalid image format. Please ensure that image is in one of following formats: "
            "[jpg, jpeg, png, bmp]"
        )
    if model_name not in ["backbone_frozen", "backbone_updated"]:
        raise Exception("Invalid model name")

    model = MultiClassClassifier(class_number=config.NUM_CLASSES, train_backbone=False)
    model.load_state_dict(
        torch.load(
            os.path.join(
                config.BASE_PATH, "data", "models", f"{model_name}", "model_dict.pt"
            ),
            map_location=device,
        )
    )

    model = model.to(device)
    model.eval()

    image = cv2.imread(image_path)
    orig = image.copy()

    image = preprocess_image(image)
    image = image.to(device)

    results = model(image)
    probabilities = torch.nn.Softmax(dim=-1)(results)
    best_index, best_prob = (
        torch.argmax(probabilities).item(),
        torch.max(probabilities).item(),
    )

    return config.LABELS_DICT[best_index], best_prob


if __name__ == "__main__":

    args = vars(parse_arguments())

    best_class, best_prob = predict(args["image"], args["device"], args["model"])

    orig = cv2.imread(args["image"])

    cv2.putText(
        orig,
        f"Label: {best_class}, prob: {best_prob}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.namedWindow("Classification", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        "Classification", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )
    cv2.imshow("Classification", orig)
    cv2.waitKey(0)
