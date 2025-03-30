import torch
import torchvision.transforms.v2 as tfs_v2

transforms = tfs_v2.Compose([
    tfs_v2.Resize((150, 150)),
    tfs_v2.ToImage(),
    tfs_v2.ToDtype(dtype=torch.float32)
])

class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 20, 3, 1, 1, bias=False),
            torch.nn.MaxPool2d(3),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(20),
            torch.nn.Conv2d(20, 50, 3, 1, 1, bias=True),
            torch.nn.Conv2d(50, 200, 3, 1, 1, bias=False),
            torch.nn.MaxPool2d(10),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(200),
            torch.nn.Flatten(),
            torch.nn.Linear(5000, 1000),
            torch.nn.ReLU(True),
            torch.nn.Linear(1000, 6),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


model = Model()
copy = torch.load('weights.pth', weights_only=True)
model.load_state_dict(copy)

description = [
    'Постройки',
    'Лес',
    'Айсберг',
    'Гора',
    'Море',
    'Улица'
]