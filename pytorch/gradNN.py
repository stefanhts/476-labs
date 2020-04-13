import pandas as pd
import numpy as np
import torch

N=300
indim,outdim = 100,1
hiddim=30
epochs = 400

data_in = torch.randn(N,indim)
data_out = torch.randn(N,outdim)

wt_in = torch.randn(indim, hiddim, requires_grad=True)
wt_out = torch.randn(hiddim, outdim, requires_grad=True)

learn = 5.E-4

for i in range(epochs):
    y_pred = data_in.mm(wt_in).clamp(min=0).mm(wt_out)

    MSE = torch.sum((y_pred - data_out)**2) / N
    if i % 10 == 9:
        print('%d : %.3e \n' % (i+1, MSE))
    MSE.backward()

    with torch.no_grad():
        wt_in -= learn * wt_in.grad
        wt_out -= learn * wt_out.grad

        wt_in.grad.zero_()
        wt_out.grad.zero_()
print('Final MSE: {0}'.format(MSE.item()))