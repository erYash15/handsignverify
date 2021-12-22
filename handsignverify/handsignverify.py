'''Handwritten Signature Verification with Keras framework.

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

#from handsignverify.models 


def match_sign(path_img1 = 'default',
    path_img2 = 'default_diff',
    features = 'VGG16',
    classification = 'SVM'
    ):

    SIZE = 224

    #image preprocessing
    if path_img1 == 'default':
        path_img1 = './images/001_01.PNG'
    
    if path_img2 == 'default_same':
        path_img2 = './images/001_02.PNG'
    
    if path_img2 == 'default_diff':
        path_img2 = './images/002_02.PNG'

    
    img1 = cv2.imread(path_img1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (SIZE,SIZE))

    img2 = cv2.imread(path_img2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, (SIZE,SIZE))

    train_data = [img1,img2]
    train_data = np.array(train_data)/255.0

    #feature extraction
    # load json and create model
    json_file = open('./models/model_VGG_ADAM.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./models/model_VGG_ADAM.h5")
    print("Loaded model from disk")

    loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

    print(loaded_model.summary())
    
    intermediate_layer_model = Model(inputs=loaded_model.input,
                                 outputs=loaded_model.layers[-2].output)
    intermediate_output_train = intermediate_layer_model.predict(train_data)

    print(intermediate_output_train.shape)
    intermediate_output_train = intermediate_output_train.reshape(1, 512)
    print(intermediate_output_train.shape)
    #classification
    filename = "./models/finalized_model_class.sav"
    class_model = pickle.load(open(filename, 'rb'))
    label = class_model.predict(intermediate_output_train)
    print(label)

if __name__ == '__main__':
    match_sign()