# Stefan Heller Problem 1 (Iris Classification)
import tensorflow
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
import pandas as pd
import numpy as py
from keras.callbacks import History

#read data and one hot encoding
data = pd.read_csv('iris.data', header = None, names = ['s_length', 's_width', 'p_length', 'p_width', 'Class' ])
data = pd.concat([pd.get_dummies (data['Class'], prefix='class'), data], axis=1)

#clean data
del data['Class']
print(data)
data_tr = data.sample (frac = 9/10, axis = 0)
data_tst = data.drop(data_tr.index)

a=['class_Iris-virginica', 'class_Iris-setosa' , 'class_Iris-versicolor']

data_tr_out = pd.concat([data_tr['class_Iris-setosa'], data_tr['class_Iris-versicolor'], data_tr['class_Iris-virginica']])

for i in a:
	del data_tr[i]

data_tr_in = data_tr
del data_tr

a=['class_Iris-virginica', 'class_Iris-setosa' , 'class_Iris-versicolor']
data_tst_out = pd.concat([data_tst['class_Iris-setosa'], data_tst['class_Iris-versicolor'], data_tst['class_Iris-virginica']])

for i in a:
	del data_tst[i]

data_tst_in = data_tst
del data_tst

#make model
model = Sequential()

model.add(Dense(4), 'relu')
model.add(Dense(40), 'sigmoid')
model.add(Dense(40), 'selu')
model.add(Dense(5), 'selu')
model.add(Dense(3))

#train model
model.complie(optimizer = 'adama', loss = 'msa', metrics = ['mse', 'accuracy'])

model.fit(data_tst_in, data_tst_out, batch_size = 32, epochs = 80)

eval = model.evaluate(data_tst_in, data_tst_out)

print(eval)