#Stefan Heller hw2 #2
import tensorflow
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense,Conv2D,Dropout, Flatten, MaxPooling2D
from keras.utils import to_categorical, np_utils
from keras.callbacks import History

import pandas as pd
import numpy as np


labels = np.load('flower_labels.npy')
images = np.load('flower_imgs.npy')

data = []

for i,l in zip(images,labels):
    data.append([i,l])

data = np.array(data)

data = pd.DataFrame(data=data, columns = ['pic','label'])

#shuffle data
data = data.sample(frac=1)

data_train = data.sample(frac=0.8, axis = 0)
data_test = data.drop(data_train.index)

data_in_train = data_train['pic'].values
data_in_test = data_test['pic'].values

train_in = np.concatenate(data_in_train)
train_in = train_in.reshape(data_train.shape[0], 32,32,3)
train_in = train_in.astype('float')
train_in = train_in/255

test_in = np.concatenate(data_in_test)
test_in = test_in.reshape(data_test.shape[0], 32,32,3)
test_in = test_in.astype('float')
test_in = test_in/255

train_out = data_train['label'].values
test_out = data_test['label'].values


model = Sequential()
model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(32,32,3),padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=3, activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=3, activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=3, activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
# model.add(Conv2D(16, input_shape=(32,32,3), activation='selu', kernel_size=3))
# model.add(Conv2D(32, kernel_size=3, activation='selu',padding='same'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.2))
# model.add(Conv2D(16, kernel_size=3, activation='relu',padding='same'))
# # model.add(MaxPooling2D(pool_size=(2,2)))
# # model.add(Conv2D(16, kernel_size=3, activation='relu',padding='same'))
#
# model.add(Flatten())
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(64, activation='relu', kernel_size=3))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(32, activation='relu', kernel_size=3))
# model.add(MaxPooling2D(2,2))
# model.add(Flatten())
model.add(Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
history = model.fit(train_in,train_out, batch_size=300, epochs=40)
eval = model.evaluate(test_in,test_out)
print(eval)
print(model.summary())



