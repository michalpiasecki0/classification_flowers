
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class MultiClassClassifier(nn.Module):
    """
    Model for multiclass classification, using fine-tuning of Resnet50 model.
    """
    def __init__(self, class_number):
        super(MultiClassClassifier, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.requires_grad_(False)
        self.head = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, class_number)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

