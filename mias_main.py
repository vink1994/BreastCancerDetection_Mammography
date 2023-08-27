import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
from keras import optimizers
from keras import losses
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import random  # for visualization
import pandas as pd
import numpy as np
from keras.utils import np_utils
from sklearn.metrics import roc_auc_score
from keras.models import load_model
from AFIM_My_model.afim_enh_mod import afim_deepnet

print('Libraries Imported')
path = '/AFIM/MIAS_Data/'
print("reading dataframe")
mias_DS_info = pd.read_csv("mias_DS_info.txt", sep=" ")
mias_DS_info = mias_DS_info.drop('Unnamed: 7', axis=1)
print(mias_DS_info)
mias_DS_info.dropna(subset=["SEVERITY"], inplace=True)
mias_DS_info.reset_index(inplace=True)
print(mias_DS_info)
mias_DS_info = mias_DS_info.drop([3], axis=0)
mias_DS_info.reset_index(inplace=True)
print(mias_DS_info)

# Turning our outputs B-M to 1-0
label = []
for i in range(len(mias_DS_info)):
    if mias_DS_info.CLASS[i] == 'CIRC':
        label.append(0)
    if mias_DS_info.CLASS[i] == 'MISC':
        label.append(1)
    if mias_DS_info.CLASS[i] == 'ASYM':
        label.append(2)
    if mias_DS_info.CLASS[i] == 'ARCH':
        label.append(3)
    if mias_DS_info.CLASS[i] == 'SPIC':
        label.append(4)
    if mias_DS_info.CLASS[i] == 'CALC':
        label.append(5)

label = np.array(label)
print(label.shape)

# define the every images filepaths in to list
img_name = []
for i in range(len(label)):
    img_name.append(path + mias_DS_info.REFNUM[i] + '.pgm')
img_name = np.array(img_name)
print(f'image addres amount {img_name.shape}')
print(mias_DS_info['CLASS'].unique())

img_path = []
last_label = []
print (img_path )
for i in range(len(img_name)):

    img = cv2.imread(img_name[i], 0)
    img = cv2.resize(img, (224,224))
    rows, cols= img.shape
    for angle in range(360):
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)    #Rotate 0 degree
            img_rotated = cv2.warpAffine(img, M, (224, 224))
            img_path.append(img_rotated)
            if label[i] == 0:
                last_label.append(0)
            if label[i] == 1:
                last_label.append(1)
            if label[i] == 2:
                last_label.append(2)
            if label[i] == 3:
                last_label.append(3)
            if label[i] == 4:
                last_label.append(4)
            if label[i] == 5:
                last_label.append(5)

# Load the training and test images
x_train, x_test, y_train, y_test = train_test_split(img_name, label, test_size=0.2, random_state=42)
len(x_train),len(x_test),len(y_train),len(y_test)
x_train = np.array(x_train)
x_test = np.array(x_test)
print(x_train.shape)
print(x_test.shape)
from keras.utils import np_utils
nClasses = 6
y_train=np_utils.to_categorical(y_train, nClasses)
y_test=np_utils.to_categorical(y_test, nClasses)

(mias_mod_param1,b,c)=x_train.shape # (35136, 224, 224)
x_train = np.reshape(x_train, (mias_mod_param1, mias_mod_param2, mias_mod_param3, 1)) # 1 for gray scale
(mias_mod_param1, mias_mod_param2, mias_mod_param3)=x_test.shape
x_test = np.reshape(x_test, (mias_mod_param1, mias_mod_param2, mias_mod_param3, 1))
model = afim_deepnet()
model.summary()
model = load_model('\mias_weights/mod_weights_mias.h5')
loss_value , accuracy = model.evaluate(x_test, y_test)
print('Test_loss_value = ' +str(loss_value))
print('test_AUC = ' + str(accuracy))
print(model.predict(x_test))
predicted_scores = model.predict(x_test)
auc = roc_auc_score(y_test, predicted_scores)
print('AUC = ' + str(auc))