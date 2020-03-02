# Stefan Heller Problem 1 (Iris Classification)
import tensorflow
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
import pandas as pd
import numpy as py
from keras.callbacks import History


def read_data():
#read data and one hot encoding
	data = pd.read_csv('iris.data', header = None, names = ['s_length', 's_width', 'p_length', 'p_width', 'Class' ])
	return data

data = read_data()


def gen_model():
#make model
	model = Sequential()

	model.add(Dense(4, activation = 'relu'))
	model.add(Dense(40, activation = 'sigmoid'))
	model.add(Dense(40, activation = 'selu'))
	model.add(Dense(5, activation = 'selu'))
	model.add(Dense(1, activation = 'softmax', output_dim = 3))

	return model
data = pd.concat([pd.get_dummies(data['Class'], prefix='class'), data], axis=1)

#clean data
del data['Class']
print(data)
data_tr = data.sample (frac = 9/10, axis = 0)
data_tst = data.drop(data_tr.index)

a=['class_Iris-virginica', 'class_Iris-setosa' , 'class_Iris-versicolor']

data_tr_out = pd.concat([data_tr['class_Iris-setosa'], data_tr['class_Iris-versicolor'], data_tr['class_Iris-virginica']]).values

for i in a:
	del data_tr[i]

data_tr_in = data_tr.values
del data_tr

a=['class_Iris-virginica', 'class_Iris-setosa' , 'class_Iris-versicolor']
data_tst_out = pd.concat([data_tst['class_Iris-setosa'], data_tst['class_Iris-versicolor'], data_tst['class_Iris-virginica']]).values

for i in a:
	del data_tst[i]

data_tst_in = data_tst.values
del data_tst

model = gen_model()

#train model
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse', 'accuracy'])

model.fit(data_tst_in, data_tst_out, batch_size = 32, epochs = 80)

eval = model.evaluate(data_tst_in, data_tst_out)

print(eval)
