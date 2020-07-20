#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 00:16:59 2020

@author: john
"""

import numpy as np
import random
from four_plane_encoder import FourPlaneEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.optimizers import SGD



# %% load datae
X1 = np.load('data.npy')
Y1 = np.load('labels.npy')

X2 = np.load('data_30.npy')
Y2 = np.load('labels_30.npy')

X = np.concatenate((X1, X2), axis=0)
Y = np.concatenate((Y1, Y2))

encoder = FourPlaneEncoder()

X, Y = encoder.expand_data(X, Y)

samples = X.shape[0]
size = 7
input_shape = (7, 7, 4)

random_indices = random.shuffle([i for i in range(samples)])

X = X[random_indices,:,:,:]
Y = Y[random_indices,:]

X = X.reshape(samples, 7, 7, 4)
Y = Y.reshape(samples, 96)

train_samples = int(0.9*samples)
X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]



# %% Define model
model = Sequential()

model.add(Conv2D(filters=60,
                kernel_size=(3,3),
                activation='relu',
                padding='same',
                input_shape=input_shape))

# model.add(Dropout(rate=0.1))

model.add(Conv2D(60, (3,3), padding='same', activation='relu'))

# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(rate=0.1))

model.add(Flatten())

model.add(Dense(120, activation='sigmoid'))

model.add(Dense(100, activation='sigmoid'))

# model.add(Dropout(rate=0.1))

model.add(Dense(96, activation='softmax'))
model.summary()



# %% compile model

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.5, momentum=0.9, nesterov=True),
              metrics=['accuracy'])



# %% Train network

model.fit(X_train, Y_train,
          batch_size=600,
          epochs=10,
          verbose=1,
          validation_data=(X_test,Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



# %% Test prediction
# X = np.load('data_10k.npy')

# test_board = np.transpose(X[0].reshape(1,4,7,7), (0,2,3,1))
test_board = X[0].reshape(1,7,7,4)

move_probs = model.predict(test_board)



# %% Save model
model.save('dl_model.h5')



# %% load model
from keras.models import load_model
model = load_model('dl_model.h5')