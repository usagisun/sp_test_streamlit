import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.models import resnet18

# ハイパーパラメータ
hyp_param = {
    'optimizer': 'SGD', # SGD, AdamW
    'lr': 0.001,
    'batch_size': 30,
    'max_epochs': 50,
    'transform' :   transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
}

class Net(pl.LightningModule):
    """モデル定義
    """
    def __init__(self):
        super().__init__()

        self.feature = resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 3)

    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h