import torch
import torchvision.transforms.v2 as tfs_v2
import PIL.Image as img
import random as rd
from tqdm import tqdm
import os


transforms = tfs_v2.Compose([
    tfs_v2.Resize((150, 150)),
    tfs_v2.ToImage(),
    tfs_v2.ToDtype(dtype=torch.float32)
])

class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 48, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(48),
            torch.nn.Conv2d(48, 64, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, padding=1)) # 64 38 38

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 96, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(96),
            torch.nn.Conv2d(96, 128, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 192, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(192),
            torch.nn.Conv2d(192, 256, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(4, padding=1) # 256 5 5
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 384, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(384),
            torch.nn.Conv2d(384, 512, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, padding=1), # 512 3 3
            torch.nn.Flatten()
        )
        self.skip = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 4, 4, 1, bias=False),
            torch.nn.BatchNorm2d(64)
        )
        self.skip2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 256, 8, 8, 1, bias=False),
            torch.nn.BatchNorm2d(256)
        )
        self.linears = torch.nn.Sequential(
            torch.nn.Linear(4608, 1000),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1000, 6)
        )

        self._initialize_weights()
    
    def _initialize_weights(self): # функция для инициализации весов
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                if m in [self.skip[0], self.skip2[0]]:  # Особый случай для skip-connection
                    torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
                else:
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        out = self.layer1(x)
        x = self.skip(x) + out
        out = self.layer2(x)
        x = self.skip2(x) + out
        x = self.layer3(x)
        return self.linears(x)


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
            classes = os.listdir('server/archive/seg_test/seg_test/buildings') + os.listdir('server/archive/seg_test/seg_test/forest') + os.listdir('server/archive/seg_test/seg_test/glacier') + os.listdir('server/archive/seg_test/seg_test/mountain') + os.listdir('server/archive/seg_test/seg_test/sea') + os.listdir('server/archive/seg_test/seg_test/street')
            if index<=436:
                return (transforms(img.open(f'server/archive/seg_test/seg_test/buildings/{classes[index]}')), y[0])
            elif index<=910:
                return (transforms(img.open(f'server/archive/seg_test/seg_test/forest/{classes[index]}')), y[1])
            elif index<=1463:
                return (transforms(img.open(f'server/archive/seg_test/seg_test/glacier/{classes[index]}')), y[2])
            elif index<=1988:
                return (transforms(img.open(f'server/archive/seg_test/seg_test/mountain/{classes[index]}')), y[3])
            elif index<=2498:
                return (transforms(img.open(f'server/archive/seg_test/seg_test/sea/{classes[index]}')), y[4])
            elif index<=2999:
                return (transforms(img.open(f'server/archive/seg_test/seg_test/street/{classes[index]}')), y[5])
        else:
            classes = os.listdir('server/archive/seg_train/seg_train/buildings') + os.listdir('server/archive/seg_train/seg_train/forest') + os.listdir('server/archive/seg_train/seg_train/glacier') + os.listdir('server/archive/seg_train/seg_train/mountain') + os.listdir('server/archive/seg_train/seg_train/sea') + os.listdir('server/archive/seg_train/seg_train/street')
            if index<=2190:
                return (transforms(img.open(f'server/archive/seg_train/seg_train/buildings/{classes[index]}')), y[0])
            elif index<=4461:
                return (transforms(img.open(f'server/archive/seg_train/seg_train/forest/{classes[index]}')), y[1])
            elif index<=6865:
                return (transforms(img.open(f'server/archive/seg_train/seg_train/glacier/{classes[index]}')), y[2])
            elif index<=9377:
                return (transforms(img.open(f'server/archive/seg_train/seg_train/mountain/{classes[index]}')), y[3])
            elif index<=11651:
                return (transforms(img.open(f'server/archive/seg_train/seg_train/sea/{classes[index]}')), y[4])
            elif index<=14033:
                return (transforms(img.open(f'server/archive/seg_train/seg_train/street/{classes[index]}')), y[5])


for_train = Dataset()
for_test = Dataset(False)

train_data = torch.utils.data.DataLoader(for_train, batch_size=32, shuffle=True)
test_data = torch.utils.data.DataLoader(for_test, batch_size=32, shuffle=True)

model = Model()
try:
  model.load_state_dict(torch.load('weights.pth', weights_only=True))
except:
  pass
model.train()

optimizator = torch.optim.Adam(model.parameters(), weight_decay=0.01)
loss_func = torch.nn.CrossEntropyLoss()

epochs = 30
acc = []

for e in range(epochs):
    model.eval()
    cnt = 0
    total = 0
    with torch.no_grad():
        test_data_tqdm = tqdm(test_data, 'Прогресс:')
        for x, y in test_data_tqdm:
            y_pred = model(x)
            cnt += torch.sum(torch.argmax(torch.softmax(y_pred, dim=1), dim=1) == torch.argmax(y, dim=1)).item()
            total += y.size(0)
        acc.append([e, cnt/total])
        print(f'Accuracy:{cnt/total}')

    model.train()
    train_data_tqdm = tqdm(train_data, f'Эпоха: {e+1}/{epochs}')
    for x, y in train_data_tqdm:
        y_pred = model(x)
        loss = loss_func(y, y_pred)

        optimizator.zero_grad()
        loss.backward()
        optimizator.step()
    torch.save(model.state_dict(), 'weights.pth') #сохранение весов каждую эпоху
