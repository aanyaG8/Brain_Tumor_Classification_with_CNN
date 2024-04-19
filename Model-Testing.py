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
import matplotlib.pyplot as plt
import seaborn as sns


# Load the history variable
with open('history.pkl', 'rb') as f:
    history = pickle.load(f)



# Use the history variable as needed
print(history)

# Now you can access the 'accuracy', 'loss', 'val_accuracy', and 'val_loss' from the history dictionary
acc = history['accuracy']
val_acc = history['val_accuracy']

epochs= range(len(acc))

# Extracting data from history dictionary
accuracy = history['accuracy']
loss = history['loss']
val_accuracy = history['val_accuracy']
val_loss = history['val_loss']
epochs = range(1, len(accuracy) + 1)

# Plotting accuracy
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting loss
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Converting history to a dataframe
history_df = pd.DataFrame(history)

# Heatmap for accuracy
plt.figure(figsize=(10,6))
sns.heatmap(history_df[['accuracy','val_accuracy']], annot = True, fmt='.4f', cmap ='coolwarm')
plt.title('Accuracy Heatmap')
plt.xlabel('Metrices')
plt.ylabel('Epochs')
print(plt.show())

#The value on the y-axis (0) corresponds to the first epoch,
#  which means the heatmap cell at location (0,?) represents the 
# model's performance on the first training session.
#  The  value  in the blue portion of the heatmap at location (0, ?) is 0.3023,
#  which signifies that the model's accuracy (or the value on the other metric 
# represented on the x-axis) was relatively low during the first epoch.  
# The blue color in the heatmap corresponds to lower accuracy values (as you mentioned earlier).

# Heatmap for Loss
plt.figure(figsize=(10,6))
sns.heatmap(history_df[['loss','val_loss']], annot = True, fmt='.4f', cmap='coolwarm')
plt.title('Loss Heatmap')
plt.xlabel('Metrices')
plt.ylabel('Epochs')
print(plt.show())
