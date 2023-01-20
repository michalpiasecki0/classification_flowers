
from torch import nn
from torchvision.models import resnet34, ResNet34_Weights


class MultiClassClassifier(nn.Module):
    """
    Model for multiclass classification, using fine-tuning of Resnet50 model.
    """
    def __init__(self, class_number: int, train_backbone: bool):
        """
        :param class_number: number of classes for classification problem
        :param train_backbone: if True, convolutional part will be trained as well, if False it will be frozen
        """
        super(MultiClassClassifier, self).__init__()
        self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        if train_backbone:
            self.backbone.requires_grad_(False)
        self.head = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

