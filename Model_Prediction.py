import io
import cv2
import os
import tqdm
import keras
import pickle
import pandas as pd # data processing
import numpy as np # linear algebra
from PIL import Image
import streamlit as st
import tensorflow as tf
import ipywidgets as widgets
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import load_model
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout


def pred(uploaded_file):
    # Load the Model
    model = load_model('Brain-Tumor-Classification.h5')

    # Lables
    labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

    #img = cv2.imread(r'C:\Users\AANNYA GUPTA\Downloads\archive (10)\Training\pituitary_tumor\p (107).jpg')
    img = np.array(Image.open(uploaded_file))
    img= cv2.resize(img,(150,150))
    img_array= np.array(img)
    print(img_array.shape)

    img_array = img_array.reshape(1,150,150,3)
    print(img_array.shape)

    #visualizing image
    from tensorflow.keras.preprocessing import image
    img = image.load_img(r'C:\Users\AANNYA GUPTA\Downloads\archive (10)\Training\pituitary_tumor\p (107).jpg')
    plt.imshow(img,interpolation='nearest')
    print(plt.show())


    # To check if model is predicting the tumor or not.
    a = model.predict(img_array) #array of multiple probabilities
    indices = a.argmax() # highest probability = the type of tumor
    print(indices)
    return indices,a