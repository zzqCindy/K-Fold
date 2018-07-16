import numpy as np
import os
import cv2
import keras
from keras.models import Sequential
from keras import optimizers
import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l1, l2
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import to_categorical
from keras import initializers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import random
from KFold import *


def img_gray_collect(pathlist,path):
    data = []
    label = []
    list = []
    for p in pathlist:
        file_dir = path+p
        for root, dirs, files in os.walk(file_dir):
            count = 0
            for file in files:
                if file.find('.mat') >= 0:
                    continue
                if file[-5] == 'L':
                    count += 1
                    src=os.path.join(root,file)
                    img=cv2.imread(src)
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                    data.append(img)
                    label.append(int(p)-1)
            list.append(count)
    return data,label,list

def load_data(index=0):
    path="../IIITD_Contact_Lens_Iris_DB/Cogent Scanner/"
    pathlist=os.listdir(path)
    data1,label1,indexList = img_gray_collect(pathlist,path)
    data=np.array(data1)
    shape = data.shape+(1,)
    data=data.reshape(shape)
    # KFold
    kf = KFold(data,label1,4,2,indexList)
    train,label,vali_train,vali_label = kf.getItem(1)
    label=to_categorical(label)
    vali_label=to_categorical(vali_label)
    return (train,label),(vali_train,vali_label)

if __name__ == "__main__":
    (train,label),(vali_train,vali_label) = load_data()
    # print(train.shape)
    # print(vali_train.shape)

