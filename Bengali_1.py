# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 20:37:17 2020

@author: Abhishek Mandal
"""
from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
import random 
import os
import cv2
import gc
from tqdm.auto import tqdm
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU
from keras.models import clone_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime as dt

train = pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
class_map_data = pd.read_csv('class_map.csv')
sample_sub_data = pd.read_csv('sample_submission.csv')

train.head(10)

def resize(df, size=64, need_progress_bar=True):
    resized = {}
    for i in range(df.shape[0]):
        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized

def res_net(input_data, filters):
  
    x1 = layers.Conv2D(filters, 3, activation='relu', padding='same')(input_data)
    x1 = tf.nn.leaky_relu(x1, alpha=0.01, name='Leaky_ReLU') 
    x2 = layers.BatchNormalization()(x1)
    x2 = layers.Dropout(0.1)(x2)
    
    x3 = layers.Conv2D(filters, 5, activation=None, padding='same')(input_data)
    x3 = tf.nn.leaky_relu(x3, alpha=0.01, name='Leaky_ReLU') 
    x4 = layers.BatchNormalization()(x3)
    x4 = layers.Dropout(0.1)(x4)
  
    x5 = layers.Conv2D(filters, 1, activation=None, padding='same')(input_data)
    x5 = tf.nn.leaky_relu(x5, alpha=0.01, name='Leaky_ReLU') 
    
    x6 = layers.Conv2D(filters, 5, activation='softmax', padding='same')(input_data)
    x6 = tf.nn.leaky_relu(x5, alpha=0.01, name='Leaky_ReLU') 

    x = layers.Add()([x2 , x4 , x5 , x6 ])
    x = layers.Activation('relu')(x)
    return x

def resnet(inputsize,outputsize,depth):
    inputs = keras.Input(shape=(inputsize,inputsize,1))
    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = tf.nn.leaky_relu(x, alpha=0.01, name='Leaky_ReLU') 
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = tf.nn.leaky_relu(x, alpha=0.01, name='Leaky_ReLU') 
    x = layers.MaxPooling2D(3)(x)
    x = layers.Dropout(0.1)(x)
    num_res_net_blocks = depth
    for i in range(num_res_net_blocks):
        x = res_net(x, 64)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.nn.leaky_relu(x, alpha=0.01, name='Leaky_ReLU') 
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(outputsize, activation='softmax')(x)
    model = keras.Model(inputs, output)
    return model

ResNet = True 
CNN = False
model_root = resnet(64, 168,10) 
model_vowel = resnet(64, 11,10)
model_consonant = resnet(64, 7,10)

model_root.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy']) 
model_vowel.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])
model_consonant.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])

batch_size = 128
epochs = 10

num_dataset = 1

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1,
                                            factor=0.5, 
                                            min_lr=0.00001)

model_dict = {
    'grapheme_root': model_root,
    'vowel_diacritic': model_vowel,
    'consonant_diacritic': model_consonant
}

history_list = []

for i in range(num_dataset): 
    train_data =  pd.merge(pd.read_parquet(f'train_image_data_{i}.parquet'), train, on='image_id').drop(['image_id'], axis=1)
    train_image = train_data.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic','grapheme'], axis=1)
    train_image = resize(train_image)/255
    train_image = train_image.values.reshape(-1, 64, 64, 1) # Image with 64x64x1 dimentions
    
    for target in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']:
        Y_train = train_data[target]
        Y_train = pd.get_dummies(Y_train).values
        x_train, x_test, y_train, y_test = train_test_split(train_image, Y_train, test_size=0.05, random_state=666)
        datagen = ImageDataGenerator(
            featurewise_center=False,  #Input mean set to 0 
            samplewise_center=False,  #Sample mean set to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range 0 to 10 degrees
            zoom_range = 0.20, # Randomly zoom image to a 20% 
            width_shift_range=0.20,  # randomly shift images horizontally to a  fraction value of height
            height_shift_range=0.20,  # randomly shift images vertically to a  fraction value of height
            horizontal_flip=False,  # randomly flip images horizontally, set to false
            vertical_flip=False)  # randomly flip images vertically, set to false


        #Calculate the parameters required to augment the given data
        datagen.fit(x_train)
        history = model_dict[target].fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                      epochs = epochs, validation_data = (x_test, y_test),
                                      steps_per_epoch=x_train.shape[0] // batch_size, 
                                      callbacks=[learning_rate_reduction])
        history_list.append(history)
        del x_train
        del x_test
        del y_train
        del y_test
        history_list.append(history)
        gc.collect()
        
    # Delete to reduce memory usage
    del train_image
    del train_data
    
del train
gc.collect()
model_save = {
    'grapheme_root': 'lir_model_root.h5',
    'vowel_diacritic': 'lir_model_vowel.h5',
    'consonant_diacritic': 'lir_model_consonant.h5'
}


model_root.save('model_root.h5')
model_vowel.save('model_vowel.h5')
model_consonant.save('model_consonant.h5')

preds_dict = {
    'grapheme_root': [],
    'vowel_diacritic': [],
    'consonant_diacritic': []
}

components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
target=[] # model predictions placeholder
row_id=[] # row_id place holder
for i in range(4):
    test_img = pd.read_parquet('test_image_data_{}.parquet'.format(i)) 
    test_img.set_index('image_id', inplace=True)

    X_test = resize(test_img)/255
    X_test = X_test.values.reshape(-1, 64, 64, 1)

    for pred in preds_dict:
        preds_dict[pred]=np.argmax(model_dict[pred].predict(X_test), axis=1)

    for k,id in enumerate(test_img.index.values):  
        for i,comp in enumerate(components):
            id_sample=id+'_'+comp
            row_id.append(id_sample)
            target.append(preds_dict[comp][k])
    del test_img
    del X_test
    gc.collect()

df_sample = pd.DataFrame(
    {
        'row_id': row_id,
        'target':target
    },
    columns = ['row_id','target'] 
)
df_sample.to_csv('submission.csv',index=False)
df_sample.head()