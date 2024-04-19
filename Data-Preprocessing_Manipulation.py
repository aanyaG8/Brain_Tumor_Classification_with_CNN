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




#directory path to the location of your training and testing files
data_directory = r"C:\Users\AANNYA GUPTA\Downloads\archive (10)"

for dirname, _, filenames in os.walk(data_directory):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Folder paths
X_train = [] #images converted to numpy array
Y_train = [] #labels
image_size = 150
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
for i in labels:
    folderPath = os.path.join('C:/Users/AANNYA GUPTA/Downloads/archive (10)/Training',i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)
        
for i in labels:
    folderPath = os.path.join('C:/Users/AANNYA GUPTA/Downloads/archive (10)/Testing',i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_train,Y_train = shuffle(X_train,Y_train,random_state=101)
X_train.shape

# SPLITTING AND TRAINING DATA
X_train,X_test,y_train,y_test = train_test_split(X_train,Y_train,test_size=0.1,random_state=101)


# Dealing with y_train values
y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))

y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)


# Dealing with y_test values
y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))

y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)

#saving the X_train and X_test values 
import numpy as np
import os

# Save the variables
np.savez_compressed('train_data.npz', X_train=X_train, y_train=y_train)





