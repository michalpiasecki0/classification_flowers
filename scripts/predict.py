"""
Script for interference with classification model.
"""
import argparse
import cv2
import os.path
import torch
import configs.train_config as config

from scripts.utils import preprocess_image
from scripts.multiclass_model import MultiClassClassifier


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to the input image")
    return parser.parse_args()


if __name__ == '__main__':

    args = vars(parse_arguments())

    model = torch.load(os.path.join(config.SAVE_MODEL_PATH, 'model.pth')).to(config.DEVICE)
    model.eval()

    image = cv2.imread(args["image"])
    orig = image.copy()
    image = preprocess_image(image)

    results = model(image)
    probabilities = torch.nn.Softmax(dim=-1)(results)

    best_index, best_prob = torch.argmax(probabilities).item(), torch.max(probabilities).item()

    cv2.putText(orig, f"Label: {config.LABELS_DICT[best_index]}, prob: {best_prob}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.namedWindow("Classification", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Classification", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Classification", orig)
    cv2.waitKey(0)
