# Stefan Heller Problem 1 (Iris Classification)
import tensorflow
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
import pandas as pd
import numpy as py
from keras.callbacks import History

data = pd.read_csv('iris.data', header = None, names = ['s_length', 's_width', 'p_length', 'p_width', 'Class' ])
data = pd.concat([pd.get_dummies (data['Class'], prefix='class'), data], axis=1)

del data['Class']
print(data)
data_tr = data.sample (frac = 9/10, axis = 0)
data_tst = data.drop(dat_tr.index)

data_tr_out = pd.concat(data_tr['class_Iris-setosa'], data_tr['class_Iris-versicolor'], data_tr['class_Iris-virginica'])
del data_tr[['class_Iris-virginica'], 'class_Iris-setosa' , 'class_Iris-versicolor']
data_tr_in = data_tr
del data_tr

data_tst_out = pd.concat(data_tst['class_Iris-setosa'], data_tst['class_Iris-versicolor'], data_tst['class_Iris-virginica'])
del data_tst[['class_Iris-virginica'], 'class_Iris-setosa' , 'class_Iris-versicolor']
data_tst_in = data_tst
del data_tst

del data_tr
del data_tst

print(data_tr)
print(data_tst)