import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import glob, warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras import callbacks as callbacks_
from tensorflow.keras import layers
from keras import models
from DataLoader import Data_generator
from Vit_model import build_model, build_Functional_model
from tensorflow.keras import callbacks
from keras.callbacks import Callback
import imageio
from keras.optimizers import Adam
import argparse



def get_run_logdir(root_logdir):
        import time
        run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
        return os.path.join(root_logdir,run_id)

    
def avoid_error(gen):
     while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except:
            pass

        

def main():
     # construct the argument parser
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train our network for')
    my_parser.add_argument('--gpu', type=int, default=1, help='Number GPU 0,1')
    my_parser.add_argument('--data_path', type=str, default='/home/kannika/codes_AI/Rheology2023/datasetMSDT_train_valid.csv')
    my_parser.add_argument('--save_dir', type=str, help='Path to Save output training model')
    my_parser.add_argument('--name', type=str, help='Name to save output in save_dir')
    my_parser.add_argument('--lr', type=float, default=1e-4)
    my_parser.add_argument('--size', type=int, default=224)
    my_parser.add_argument('--batchsize', type=int, default=16)
    
    args = my_parser.parse_args()
    
    ## set gpu
    gpu = args.gpu
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}" 
    physical_devices = tf.config.list_physical_devices('GPU') 
    print("Num GPUs:", len(physical_devices))
    
    ## get my_parser
    save_dir = args.save_dir
    name = args.name
    root_base = f'{save_dir}/{name}/R1'
    os.makedirs(root_base, exist_ok=True)
    data_path = args.data_path
    IMAGE_SIZE = args.size
    BATCH_SIZE = args.batchsize
    ## train seting
    lr = args.lr
    EPOCHS = args.epochs
    
    ## import dataset
    DF = pd.read_csv(data_path)
    DF_TRAIN = DF[DF['subset']=='train'].reset_index(drop=True)
    DF_VAL = DF[DF['subset']=='valid'].reset_index(drop=True)
    ### Get data Loder
    train_generator, val_generator = Data_generator(IMAGE_SIZE, BATCH_SIZE, DF_TRAIN, DF_VAL)
    
    ## Set mkdir TensorBoard 
    ##root_logdir = f'/media/SSD/rheology2023/VitModel/Regression/tensorflow/ExpTest/R1/Mylogs_tensor/'
    root_logdir = f"{root_base}/Mylogs_tensor"
    os.makedirs(root_logdir, exist_ok=True)
    ### Run TensorBoard 
    run_logdir = get_run_logdir(root_logdir)
    tensorboard_cb = callbacks_.TensorBoard(log_dir=run_logdir)
    
    ### Create Model 
    model = build_Functional_model(fine_tune=False, image_size = IMAGE_SIZE)
    model.summary()
    print('='*100)
    
    ## Model Complier
    model.compile(optimizer = Adam(2e-6, decay=1e-6), 
              loss = 'mse', 
              metrics = ['mse'])
    
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = val_generator.n // val_generator.batch_size
    
    
    save_checkpoin = f"{root_base}/checkpointer"
    os.makedirs(save_checkpoin, exist_ok=True)
    ## reduce_lr, checkpointer ต้องเปลี่ยนใหม่
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',  
                                                     factor = 0.2,
                                                     patience = 5,
                                                     verbose = 1,
                                                     min_delta = 1e-5,
                                                     min_lr = 1e-6,
                                                     mode = 'auto')


    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = f'{save_checkpoin}/modelRegress_ViT_l32_Rheology.hdf5',
                                                      monitor = 'val_loss', 
                                                      verbose = 1, 
                                                      save_best_only = True,
                                                      save_weights_only = True,
                                                      mode = 'auto')

    callbacks = [reduce_lr, checkpointer]
    
    modelNamemkdir = f"{root_base}/models"
    os.makedirs(modelNamemkdir, exist_ok=True)
    Model2save = f'{modelNamemkdir}/modelRegress_ViT_l32_Rheology_R1.h5'
    
    ## Fit model
    model.fit(x = avoid_error(train_generator),
              steps_per_epoch = STEP_SIZE_TRAIN,
              validation_data = val_generator,
              validation_steps = STEP_SIZE_VALID,
              epochs = EPOCHS,
              callbacks = [tensorboard_cb, callbacks])
    model.save(Model2save)
    ### print
    print(f"Save Linear regression Vitsiontranformer as: {Model2save}")
    print(f"***"*100)
    

    
## Run Function 
if __name__ == '__main__':
    main()












