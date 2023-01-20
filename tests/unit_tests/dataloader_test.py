import unittest
import torch
from src.utils import get_dataloader
from torchvision import transforms



class DataLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset, self.dataloader = get_dataloader(root_dir='../data/dataset/train',
                                                       transforms=transforms.Compose([
                                                                    transforms.Resize((250, 250)),
                                                                    transforms.ToTensor()]),
                                                       target_transform=None,
                                                       batch_size=3,
                                                       shuffle=False)

    def test_length(self):
        self.assertEqual(len(self.dataset), 18)

    def test_batch_len(self):
        self.assertEqual(len(next(iter(self.dataloader))[0]), 3)

    def test_batch_shape(self):
        self.assertTrue(next(iter(self.dataloader))[0].shape == (3, 3, 250, 250))

    def test_batch_type(self):
        self.assertTrue(isinstance(next(iter(self.dataloader))[0], torch.Tensor))


if __name__ == '__main__':
    unittest.main()
