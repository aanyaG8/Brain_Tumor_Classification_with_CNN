import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\
import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score
import ipywidgets as widgets
import io
from PIL import Image
import tqdm
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import tensorflow as tf
import os
import pickle

# Load the variables
data = np.load('train_data.npz')
X_train = data['X_train']
y_train = data['y_train']



model = Sequential()


# layer 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# layer 2
model.add(Conv2D(64,(3,3), activation='relu'))
# MaxPooling layer
model.add(MaxPooling2D(2,2))
# Dropout layer (for futile features)
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

# For classification 
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))

# Output layer should be dense and should have the total number of outputs
# Softmax activation as output of prediction is y/n 
model.add(Dense(4, activation='softmax')) 

# Model summary
#model.summary()

# Compiling the model
model.compile(loss= 'categorical_crossentropy', optimizer='Adam', metrics = ['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs = 20, validation_split=0.1)

# Save the history variable
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Saving the model
model.save('Brain-Tumor-Classification.h5')


