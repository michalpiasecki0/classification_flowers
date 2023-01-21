import unittest
import torch
import cv2

from src.utils import preprocess_image
from conf.train_config import STD, MEAN


class PreprocessImageTest(unittest.TestCase):
    def setUp(self) -> None:
        image = cv2.imread("../data/test_image.jpg")
        self.processed = preprocess_image(image)

    def test_type(self):
        self.assertTrue(isinstance(self.processed, torch.Tensor))

    def test_shape(self):
        self.assertEqual(list(self.processed.shape), [1, 3, 224, 224])

    def test_values_range(self):
        before_std = self.processed[0, 0, :, :] * STD[0]

        self.assertTrue(torch.max(before_std) <= (1 - MEAN[0]))
        self.assertTrue(torch.min(before_std) >= (0 - MEAN[0]))

        before_mean = before_std + MEAN[0]
        self.assertTrue(torch.max(before_mean) <= 1)
        self.assertTrue(torch.min(before_mean) >= 0)


if __name__ == "__main__":
    unittest.main()
