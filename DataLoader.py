import os
import pandas as pd
import numpy as np
import tensorflow as tf
import glob, warnings
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# DF = pd.read_csv('/home/kannika/codes_AI/Rheology2023/datasetMSDT_train_valid.csv')
# DF_TRAIN = DF[DF['subset']=='train'].reset_index(drop=True)
# print(DF_TRAIN.shape)
# DF_VAL = DF[DF['subset']=='valid'].reset_index(drop=True)
# print(DF_VAL.shape)
#IMAGE_SIZE = 224 
#BATCH_SIZE = 16

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      brightness_range=[0.5,1.5],
      shear_range=0.4,
      zoom_range=0.2,
      horizontal_flip=False,
      fill_mode='constant')

valid_datagen = ImageDataGenerator(rescale=1./255)

def Data_generator(IMAGE_SIZE, BATCH_SIZE, DF_TRAIN, DF_VAL):
    train_generator = train_datagen.flow_from_dataframe(
            dataframe = DF_TRAIN,
            directory = None,
            x_col = 'pathimg',
            y_col = 'MSDT',
            target_size = (IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            color_mode= 'rgb',
            class_mode='raw')

    val_generator = valid_datagen.flow_from_dataframe(
            dataframe = DF_VAL,
            directory = None,
            x_col = 'pathimg',
            y_col = 'MSDT',
            target_size = (IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            color_mode= 'rgb',
            class_mode='raw')
    return train_generator, val_generator 


