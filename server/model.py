import torch
import torchvision.transforms.v2 as tfs_v2

class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return x


copy = torch.load('weights.txt', weights_only=True)