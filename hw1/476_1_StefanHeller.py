# Stefan Heller Problem 1 (Iris Classification)
import tensorflow
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.metrics import categorical_accuracy
import pandas as pd
import numpy as py
from keras.callbacks import History

#initialize dataframes
data_tr = pd.DataFrame()
data_tst = pd.DataFrame()
data_tr_in = pd.DataFrame()
data_tr_out = pd.DataFrame()
data_tst_in = pd.DataFrame()
data_tst_out = pd.DataFrame()

def read_data():
#read data and one hot encoding
	data = pd.read_csv('iris.data', header = None, names = ['s_length', 's_width', 'p_length', 'p_width', 'Class' ])
	return data


def gen_model():
	#make model
	model = Sequential()

	model.add(Dense(4, input_shape=(4,), activation = 'selu'))
	model.add(Dense(10, activation = 'selu'))
	model.add(Dense(10, activation = 'selu'))
	model.add(Dense(4, activation = 'selu'))
	model.add(Dense(3, activation = 'softmax'))

	return model


def clean_data(data):
	#one hot encoding
	data = pd.concat([pd.get_dummies(data['Class'], prefix='class'), data], axis=1)
	del data['Class']
	#clean data
	data_tr = data.sample (frac = 9/10, axis = 0)
	data_tst = data.drop(data_tr.index)

	a=['class_Iris-virginica', 'class_Iris-setosa' , 'class_Iris-versicolor']
	data_tr_out=pd.DataFrame()
	data_tr_out['setosa'] = data_tr['class_Iris-setosa']
	data_tr_out['versi'] = data_tr['class_Iris-versicolor']
	data_tr_out['virgi'] = data_tr['class_Iris-virginica']

	for i in a:
		del data_tr[i]

	data_tr_in = data_tr.values
	del data_tr
	data_tst_out=pd.DataFrame()
	data_tst_out['setosa'] = data_tst['class_Iris-setosa']
	data_tst_out['versi'] = data_tst['class_Iris-versicolor']
	data_tst_out['virgi'] = data_tst['class_Iris-virginica']

	for i in a:
		del data_tst[i]

	data_tst_in = data_tst.values
	del data_tst


#clean data, create, train, test model
cleandata(read_data())
model = gen_model()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [categorical_accuracy])
model.fit(data_tst_in, data_tst_out.values, batch_size = 32, epochs = 350)
eval = model.evaluate(data_tst_in, data_tst_out.values)

print(eval)
