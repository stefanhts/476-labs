import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import datetime as dt
from tqdm import tqdm

f = h5py.File('Galaxy10.h5', 'r')
images = f['images'][()]
ans = f['ans'][()]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params ={
    'learning_rate': 0.001,
    'batch_size': 128,
    'epochs': 6,
    'num_workers': 6
}

split = int(len(images) * 0.8)
image_train = images[:split]
image_test = images[split:]
ans_train = ans[:split]
ans_test = ans[split:]

image_train = image_train.reshape(-1, 3, 69, 69)
image_train = image_train.astype('float')/255
image_test = image_test.reshape(-1, 3, 69, 69)
image_test = image_test.astype('float')/255

image_train = torch.from_numpy(np.float32(image_train))
image_test = torch.from_numpy(np.float32(image_test))
ans_train = torch.from_numpy(ans_train).long()
ans_test = torch.from_numpy(ans_test).long()

train_data = torch.utils.data.TensorDataset(image_train, ans_train)
train_load = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'])
test_data = torch.utils.data.TensorDataset(image_test, ans_test)
test_load = torch.utils.data.DataLoader(test_data, batch_size=100)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.d = None
        self.conv1 = nn.Conv2d(3, 69, kernel_size=3)
        self.conv2 = nn.Conv2d(69, 36, 5)
        self.conv3 = nn.Conv2d(36, 42, 5)
        self.pool = nn.MaxPool2d(2, 2)
        if self.d == None:
            x = image_train.view(-1, 3, 69, 69)[:2]
            x = self.pool(self.conv1(x))
            x = self.pool(self.conv2(x))
            x = self.pool(self.conv3(x))
            size = x.size()
            self.d = size[1] * size[2] * size[3]

        self.fc1 = nn.Linear(self.d, 256)
        self.fc2 = nn.Linear(256, 24)
        self.fc3 = nn.Linear(24, 10)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = x.view(-1, self.d)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def learn(loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def accuracy(out, pred):
    correct = 0
    for i,j in zip(out, pred):
        if i == j:
            correct += 1
    return (correct/len(pred))

def train(model, train_in, train_out):
    epochs = params['epochs']
    for epoch in range(epochs):
       for i, (images,ans) in enumerate(train_load):
           images = images.to(device)
           ans = ans.to(device)
           outputs = model(images)
           loss = criterion(outputs,ans)

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           if (i + 1) % 2000 == 0:
               print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}')



def validate(model, test_in, test_out):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        for i, (images, ans) in enumerate(test_load):
            images = images.to(device)
            ans = ans.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += ans.size(0)
            n_correct += (predicted == ans).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of network: {acc} %')


model = Model().to(device)
start = dt.datetime.now()
optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
criterion = nn.CrossEntropyLoss()

train(model, image_train, ans_train)
validate(model, image_test, ans_test)

