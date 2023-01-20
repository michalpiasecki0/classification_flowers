import unittest
import torch

from src.utils import get_dataloader
from torchvision import transforms
from src.multiclass_model import MultiClassClassifier
from src.train import train_loop


class TrainLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        _, dataloader = get_dataloader(
            root_dir="../data/dataset/train",
            transforms=transforms.Compose(
                [transforms.Resize((250, 250)), transforms.ToTensor()]
            ),
            target_transform=None,
            batch_size=3,
            shuffle=False,
        )
        model = MultiClassClassifier(class_number=5, train_backbone=False).to("cpu")
        loss_fn = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        self.acc, self.loss = train_loop(dataloader, model, loss_fn, opt, "cpu")

    def test_types(self):
        self.assertTrue(isinstance(self.acc, torch.Tensor))
        self.assertTrue(isinstance(self.loss, float))

    def test_values(self):
        self.assertTrue(0 < self.acc < 100)


if __name__ == "__main__":
    unittest.main()
