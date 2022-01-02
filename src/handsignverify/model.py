'''
Handwritten Signature Verification with Keras framework.

# Reference:
- [Yash Gupta](Graduation Project Indian Institute of Information Technology, Nagpur)
- [Signatures Dataset](https://www.kaggle.com/divyanshrai/handwritten-signatures)


'''

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import glob
import requests
import json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import applications
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import backend as K
import gc
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def match_sign(img1 = 'NONE',
            img2 = 'NONE',
            features = 'VGG16',
            classification = 'SVM'
            ):

    SIZE = 224


    if img1 == 'NONE':
        print("Pass the path of image 1. Using Default for now.")
        img1 = './001_01.PNG'
    if img2 == 'NONE':
        print("Pass the path of image 2. Using Default for now.")
        img2 = './001_02.PNG'

    img1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (SIZE,SIZE))

    img2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, (SIZE,SIZE))

    train_data = [img1,img2]
    train_data = np.array(train_data)/255.0


    # VGG16 Model
    url = 'https://raw.github.com/erYash15/handsignverify/master/handsignverify/models/model_VGG_ADAM.json'

    model_json = requests.get(url).content
    model_keras = model_from_json(model_json)

    print("Created Model.........")

    # loading weights

    if not os.path.exists('VGG_ADAM.h5'):
        url = 'https://raw.github.com/erYash15/handsignverify/master/handsignverify/models/model_VGG_ADAM.h5'

        model_weights = requests.get(url)

        with open('VGG_ADAM.h5','wb') as f:
            f.write(model_weights.content)

    #print(model_weights)
    #print(type(model_weights))
    model_keras.load_weights('VGG_ADAM.h5')

    print("Loaded Weights.......")

    model_keras.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

    print("Model Compiled.........")

    intermediate_layer_model = Model(
                                inputs = model_keras.input,
                                outputs = model_keras.layers[-2].output)
    intermediate_output_train = intermediate_layer_model.predict(train_data)

    
    intermediate_output_train = intermediate_output_train.reshape(1, 512)

    #print(intermediate_output_train.shape)

    # classification
    
    if not os.path.exists('SVM_sklearn.sav'):
        url = 'https://raw.github.com/erYash15/handsignverify/master/handsignverify/models/finalized_model_class.sav'


        with open('SVM_sklearn.sav','wb') as f:
            f.write(requests.get(url).content)

    class_model = pickle.load(open('./SVM_sklearn.sav', 'rb'))
    label = class_model.predict(intermediate_output_train)
    return label


        
if __name__ == "__main__":
    match_sign()