import unittest
import torch

from src.utils import get_dataloader
from torchvision import transforms
from src.multiclass_model import MultiClassClassifier
from src.train import train_loop


class TrainLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        _, self.dataloader = get_dataloader(
            root_dir="../data/dataset/train",
            transforms=transforms.Compose(
                [transforms.Resize((250, 250)), transforms.ToTensor()]
            ),
            target_transform=None,
            batch_size=3,
            shuffle=False,
        )
        self.model = MultiClassClassifier(class_number=5, train_backbone=False).to(
            "cpu"
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def test_types(self):
        acc, loss = train_loop(
            self.dataloader, self.model, self.loss_fn, self.opt, "cpu"
        )
        self.assertTrue(isinstance(acc, torch.Tensor))
        self.assertTrue(isinstance(loss, float))

    def test_values(self):
        acc, _ = train_loop(self.dataloader, self.model, self.loss_fn, self.opt, "cpu")
        self.assertTrue(0 < acc < 100)

    def test_wrong_device(self):
        self.assertRaises(
            RuntimeError,
            train_loop,
            self.dataloader,
            self.model,
            self.loss_fn,
            self.opt,
            "device",
        )


if __name__ == "__main__":
    unittest.main()
