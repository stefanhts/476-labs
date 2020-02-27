import tensorflow
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History
from keras.optimizers import Adam
import matplotlib.pyplot as plt

data = pd.read_csv ('abalone.data', header=None, names=['sex','length','diameter','height','whole',
	'shucked','viscera','shell','rings'])

data = pd.concat ([pd.get_dummies (data['sex'], prefix='sex'), data], axis=1)
del data['sex']
train_data = data.sample (frac = 9/10, axis = 0)
test_data = data.drop (train_data.index)
train_out = train_data['rings'].values
del train_data['rings']
train_in = train_data.values
del train_data
test_out = test_data['rings'].values
del test_data['rings']
test_in = test_data.values
del test_data

model = Sequential()

model.add(Dense(40, activation = 'selu'))
model.add(Dense(40, activation = 'selu'))
model.add(Dense(3, activation = 'selu'))
model.add(Dense(40, activation = 'selu'))
model.add(Dense(40, activation = 'selu'))
model.add(Dense(1, activation = 'linear'))

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse'])
model.fit(train_in, train_out, batch_size = 32, epochs = 80)

eval = model.evaluate(test_in, test_out)

print('Mean_squared_error:',eval[1])
