# Stefan Heller hw2 #1
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import History
import matplotlib.pyplot as plt
import pandas as pd


names=[]
for i in range(33):
    names.append(i)
dat = pd.read_csv('3m-data.csv', header=None, names=names)

dat_train=dat.sample(frac = 9/10, axis = 0)
dat_test = dat.drop(dat_train.index)

dat_train = dat_train.values
dat_test = dat_test.values

model = Sequential()
model.add(Dense(len(names), activation='selu'))
model.add(Dense(30, activation='selu'))
model.add(Dense(1, activation='selu'))
model.add(Dense(30, activation='selu'))
model.add(Dense(len(names), activation='selu'))

model.compile(optimizer=Adam(lr=0.00001), loss='mse', metrics=['mse'])
model.fit(dat_train,dat_train, epochs=12, batch_size=32)
eval = model.evaluate(dat_test,dat_test)

print('Mean squaured error: {0}'.format(eval[0]))
plt.show()
