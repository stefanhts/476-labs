#Homework 1 problem 2
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import to_categorical, np_utils

import matplotlib.pyplot as plt
from keras.callbacks import History
import pandas as pd

batch_size = 512
epochs = 9
loss = 'categorical_crossentropy'
optimizers = Adam(lr = 0.001)
metrics =['accuracy']

(tr_in, tr_out), (tst_in, tst_out) = mnist.load_data()

# reshape
tr_in = tr_in.reshape(tr_in.shape[0], 28, 28, 1)
tr_in = tr_in.astype('float32')
tst_in = tst_in.reshape(tst_in.shape[0], 28, 28, 1)
tst_in = tst_in.astype('float32')

# normalize
tr_in = tr_in / 255
tst_in = tst_in / 255

# one hot encoding
num_classes = 10
tr_out = np_utils.to_categorical(tr_out, num_classes)
tst_out = np_utils.to_categorical(tst_out, num_classes)


def create_model():
# create model
    model = Sequential()
    model.add(Conv2D(14, (3,3), input_shape=(28,28,1), activation = 'relu'))
    model.add(Conv2D(38, (3,3), activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(28, (3,3), activation = 'selu'))
    model.add(Conv2D(52, (3,3), activation = 'selu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(20, activation = 'selu'))
    model.add(Dense(10, activation = 'softmax'))
# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

model = create_model()

history = model.fit(tr_in, tr_out, batch_size = batch_size, epochs = epochs, validation_split = .1)

eval = model.evaluate(tst_in, tst_out)

print(model.summary())
print("categorical cross entropy loss of :", eval[0], "Accuracy:", eval[1])
