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

image_train = torch.from_numpy(np.float32(image_train))
image_test = torch.from_numpy(np.float32(image_test))
ans_train = torch.from_numpy(ans_train).long()
ans_test = torch.from_numpy(ans_test).long()

image_train = image_train.view(-1, 3, 69, 69)
image_test = image_test.view(-1, 3, 69, 69)


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
    train_in, train_out = train_in.to(device), train_out.to(device)
    N = train_in.size(0)
    for epoch in range(params['epochs']):
        for batch in tqdm(range(N // params['batch_size'])):
            i_start = batch*params['batch_size']
            i_end = i_start+params['batch_size']

            t_in = train_in[i_start:i_end]
            t_out = train_out[i_start:i_end]

            output = model(t_in)
            loss = criterion(output, t_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch+1) %16 == 0:
                print('Epoch: {0}/{1}, loss: {2}'.format(epoch+1,params['epochs'], loss.item()))
    print('Done training, training time: {0}'.format(dt.datetime.now() - start))

def validate(model, test_in, test_out):
    test_in, test_out = test_in.to(device), test_out.to(device)
    print('validating...')
    with torch.no_grad():
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        n_samples = len(test_out)
        n_correct = 0
        for batch in range(test_in.size(0) // params['batch_size']):
            i_start = batch*params['batch_size']
            i_end = i_start+params['batch_size']

            t_in = test_in[i_start:i_end]
            t_out = test_out[i_start:i_end]

            outputs = model(t_in)
            _, predicted = torch.max(outputs, 1)
            n_samples += t_out.size(0)
            n_correct += (predicted == t_out).sum().item()

            for i in range(params['batch_size']):
                label = t_out[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1
        acc = n_correct/n_samples
        print(f'Accuracy of network: {acc}%')


model = Model().to(device)
start = dt.datetime.now()
optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
criterion = nn.CrossEntropyLoss()

train(model, image_train, ans_train)
validate(model, image_test, ans_test)

