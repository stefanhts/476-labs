import numpy as np
import pandas as pd

N=30 #number of samples
indim, outdim = 100, 1
hiddim=30 #dimension of hidden layer
epochs = 300

"create random data..."
data_in = np.random.randn(N, indim)
data_out = np.random.randn(N, outdim)
# print(data_out)

"input weights, no biases for simplicity"
wt_in = np.random.randn(indim, hiddim)
wt_out = np.random.randn(hiddim, outdim)

#Setting the learning rate (This is very important)
learn = 5.E-5

for i in range(epochs):
    hid = np.dot(data_in, wt_in)
    hid_act = np.maximum(hid,0) #relu
    y_pred = np.dot(hid_act, wt_out)
    diff2 = (y_pred - data_out)**2
    MSE = np.sum(diff2)
    if i % 10 == 9:
        print('%d : %.3e \n' % (i+1, MSE))
    grad_y_pred = 2*(y_pred-data_out)
    grad_weight_out = np.dot(hid_act.T, grad_y_pred)
    grad_hid_act = np.dot(grad_y_pred, wt_out.T)
    grad_hid = grad_hid_act.copy()
    grad_hid[hid < 0] = 0
    grad_weight_in = np.dot(data_in.T, grad_hid)
    wt_in -= learn * grad_weight_in
    wt_out -= learn * grad_weight_out

print('Final MSE value: {0}'.format(MSE))
