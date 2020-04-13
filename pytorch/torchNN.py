import numpy as np
import torch

N = 300
indim,outdim = 100,1
hiddim = 30

data_in = torch.randn(N,indim)
data_out = torch.randn(N,outdim)

nnet = torch.nn.Sequential(
    torch.nn.Linear(indim,hiddim),
    torch.nn.ReLU(),
    torch.nn.Linear(hiddim,outdim)
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learn = 1.E-4

epochs = 400
for i in range(epochs):
    y_pred = nnet(data_in)
    MSE = loss_fn(y_pred,data_out)
    if i % 10 == 9:
        print('%d : %.3e \n' % (i+1, MSE))
    nnet.zero_grad()
    MSE.backward()

    with torch.no_grad():
        for param in nnet.parameters():
            param -= learn * param.grad

print('Final MSE value: {0}'.format(MSE.item()))