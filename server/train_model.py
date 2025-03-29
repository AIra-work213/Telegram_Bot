import torch
import torchvision.transforms.v2 as tfs_v2
import PIL.Image as img
import random as rd
from tqdm import tqdm

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
            torch.nn.BatchNorm2d(20),
            torch.nn.Conv2d(20, 50, 3, 1, 1, bias=True),
            torch.nn.Conv2d(50, 200, 3, 1, 1, bias=False),
            torch.nn.MaxPool2d(5),
            torch.nn.BatchNorm2d(200),
            torch.nn.Flatten(),
            torch.nn.Linear(20000, 10000),
            torch.nn.Linear(10000, 6)
        )

    def forward(self, x):
        return self.layers(x)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        if is_train:
            self.length = 14034
        else:
            self.length = 3000
        self.is_train = is_train
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        y = torch.eye(6, dtype=torch.float32)
        if self.is_train == False:
            if index<=436:
                return (transforms(img.open('archive/seg_test/seg_test/buildings')), y[0])
            elif index<=910:
                return (transforms(img.open('archive/seg_test/seg_test/forest')), y[1])
            elif index<=1463:
                return (transforms(img.open('archive/seg_test/seg_test/glacier')), y[2])
            elif index<=1988:
                return (transforms(img.open('archive/seg_test/seg_test/mountain')), y[3])
            elif index<=2498:
                return (transforms(img.open('archive/seg_test/seg_test/sea')), y[4])
            elif index<=2999:
                return (transforms(img.open('archive/seg_test/seg_test/street')), y[5])
        else:
            if index<=2190:
                return (transforms(img.open('archive/seg_train/seg_train/buildings')), y[0])
            elif index<=4461:
                return (transforms(img.open('archive/seg_train/seg_train/forest')), y[1])
            elif index<=6865:
                return (transforms(img.open('archive/seg_train/seg_train/glacier')), y[2])
            elif index<=9377:
                return (transforms(img.open('archive/seg_train/seg_train/mountain')), y[3])
            elif index<=11651:
                return (transforms(img.open('archive/seg_train/seg_train/sea')), y[4])
            elif index<=14033:
                return (transforms(img.open('archive/seg_train/seg_train/street')), y[5])



for_train = Dataset()
for_test = Dataset(False)

train_data = torch.utils.data.DataLoader(for_train, batch_size=10, shuffle=True)
test_data = torch.utils.data.DataLoader(for_test, batch_size=10, shuffle=True)

model = Model()

optimizator = torch.optim.Adam(model.parameters(), weight_decay=0.01)
loss_func = torch.nn.CrossEntropyLoss()

epochs = 10

for e in range(epochs):
    train_data_tqdm = tqdm(train_data, f'Эпоха: {e}/{epochs}')
    for x, y in train_data:
        y_pred = model(x)
        loss = loss_func(y, y_pred)

        optimizator.zero_grad()
        loss.backward()
        optimizator.step()
