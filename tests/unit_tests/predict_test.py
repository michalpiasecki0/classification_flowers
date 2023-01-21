import unittest
from src.predict import predict
from conf.train_config import LABELS_DICT


class PredictTest(unittest.TestCase):
    def setUp(self) -> None:
        image_path = "../data/test_image.jpg"
        self.label, self.probability = predict(image_path, "cpu", "backbone_frozen")

    def test_type(self):
        self.assertTrue(isinstance(self.label, str))
        self.assertTrue(isinstance(self.probability, float))

    def test_probability(self):
        self.assertTrue(0 < self.probability < 1)

    def test_label(self):
        self.assertTrue(self.label in LABELS_DICT.values())


if __name__ == "__main__":
    unittest.main()
