import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import glob, warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers
from keras import models
from DataLoader import Data_generator, split_train_valid ## New function 
from sklearn.model_selection import train_test_split
from Vit_model import build_Functional_model, build_Functional_ViTb32, loadresumemodel
from tensorflow.keras import callbacks
from keras.callbacks import Callback
import imageio
from keras.optimizers import Adam
#from keras.optimizers.legacy import Adam
import argparse
#load Check point
from tensorflow.keras.models import load_model



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
    my_parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train our network for')
    my_parser.add_argument('--gpu', type=int, default=1, help='Number GPU 0,1')
    my_parser.add_argument('--set4000sr', action='store_true')
    my_parser.add_argument('--data_path', type=str, default='/home/kannika/codes_AI/Rheology2023/MSDTGLY7Level10fold_datatrain_tSecond.csv')
    my_parser.add_argument('--save_dir', type=str, help='Main Path to Save output training model')
    my_parser.add_argument('--name', type=str, help='Name to save output in save_dir')
    my_parser.add_argument('--R', type=int, help='1 or 2 : 1=R1, 2=R2')
    my_parser.add_argument('--fold', type=int, help='1-10')
    my_parser.add_argument('--lr', type=float, default=1e-4)
    my_parser.add_argument('--size', type=int, default=384)
    my_parser.add_argument('--batchsize', type=int, default=16)
    my_parser.add_argument('--resume', action='store_true')
    my_parser.add_argument('--checkpoint_dir', type=str ,default=".")
    my_parser.add_argument('--tensorName', type=str ,default="Mylogs_tensor")
    #my_parser.add_argument('--checkpointerName', type=str ,default="checkpoin_callback")
    my_parser.add_argument('--epochendName', type=str ,default="on_epoch_end")
    my_parser.add_argument('--FmodelsName', type=str ,default="models")
    
    args = my_parser.parse_args()
    
    ## set gpu
    Fold = f"fold{args.fold}"
    gpu = args.gpu
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}" 
    physical_devices = tf.config.list_physical_devices('GPU') 
    print("Num GPUs:", len(physical_devices))
    
    ## get my_parser
    save_dir = args.save_dir
    name = args.name
    R = args.R
    _R = f'R{R}'
    root_base = f'{save_dir}/{name}/{Fold}/{_R}'
    os.makedirs(root_base, exist_ok=True)
    data_path = args.data_path
    IMAGE_SIZE = args.size
    BATCH_SIZE = args.batchsize
    ## train seting
    lr = args.lr
    EPOCHS = args.epochs
    
    ## import dataset  
    DF = pd.read_csv(data_path)
    #DF_Fold = DF[DF["fold"]!= args.fold].reset_index(drop=True)
    ## Split train valid set.
    DF_TRAIN = DF[DF["fold"]!= args.fold].reset_index(drop=True)
    DF_VAL = DF[DF["fold"]== args.fold].reset_index(drop=True)
    #DF_TRAIN, DF_VAL = split_train_valid(DF_Fold)
    ### Get data Loder
    train_generator, val_generator = Data_generator(IMAGE_SIZE, BATCH_SIZE, DF_TRAIN, DF_VAL)
    
    ## Set mkdir TensorBoard 
    ## root_logdir = f'/media/SSD/rheology2023/VitModel/Regression/tensorflow/ExpTest/R1/Mylogs_tensor/'
    root_logdir = f"{root_base}/{args.tensorName}"
    os.makedirs(root_logdir, exist_ok=True)
    ### Run TensorBoard 
    run_logdir = get_run_logdir(root_logdir)
    tensorboard_cb = callbacks.TensorBoard(log_dir=run_logdir)
    
    ### Create Model 
    if args.resume and args.R == 1:
        print("**********Load Model R1 Resume**********")
        input_shape, model = loadresumemodel(args.checkpoint_dir)
    elif args.resume and args.R == 2:
        print("**********Load Model R2 Resume**********")
        input_shape, model = loadresumemodel(args.checkpoint_dir)
    elif args.R == 1:
        print("**********Build VITl32 Model**********")
        model = build_Functional_model(fine_tune=False, image_size = IMAGE_SIZE)
    elif args.R == 2:
        print("**********Load Model R1 to Train R2 Model**********")
        input_shape, model = loadresumemodel(args.checkpoint_dir)
    model.summary()
    print('='*100)
    
    ## Set up model path
    modelNamemkdir = f"{root_base}/{args.FmodelsName}"
    os.makedirs(modelNamemkdir, exist_ok=True)
    
    ## Check GLY set
    if args.set4000sr :
        modelName = f'ViT_l32_RegressMSD_GLY7Level4000Serise_{Fold}_{_R}.h5'
    else: 
        modelName = f'ViT_l32_RegressMSD_GLY7Level_{Fold}_{_R}.h5'
        
    Model2save = f'{modelNamemkdir}/{modelName}'
    root_Metrics = f'{root_base}/{args.epochendName}/'
    os.makedirs(root_Metrics, exist_ok=True)
    class Metrics(Callback):
            def on_epoch_end(self, epochs, logs={}):
                self.model.save(f'{root_Metrics}{modelName}')
                return
    
    # For tracking Quadratic Weighted Kappa score and saving best weights
    metrics = Metrics()
    
   
    
    ## Model Complier
    model.compile(optimizer = Adam(lr, decay=lr), ##tf.keras.optimizers.legacy.Adam
              loss = 'mse', 
              metrics = ['mse'])
    
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = val_generator.n // val_generator.batch_size
    
    
#     save_checkpoin = f"{root_base}/checkpointer"
#     os.makedirs(save_checkpoin, exist_ok=True)
#     ## reduce_lr, checkpointer ต้องเปลี่ยนใหม่
#     reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',  
#                                                      factor = 0.2,
#                                                      patience = 5,
#                                                      verbose = 1,
#                                                      min_delta = 1e-4,
#                                                      min_lr = 1e-5,
#                                                      mode = 'auto')


#     checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = f'{save_checkpoin}/modelRegress_ViT_l32_Rheology.hdf5',
#                                                       monitor = 'val_loss', 
#                                                       verbose = 1, 
#                                                       save_best_only = True,
#                                                       save_weights_only = True,
#                                                       mode = 'auto')

#     callbacks = [reduce_lr, checkpointer]
    
    ## set up save Checkpoint every epoch
#     save_checkpoin_callback = f"{root_base}/{args.checkpointerName}"
#     os.makedirs(save_checkpoin_callback, exist_ok=True)
#     model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_checkpoin_callback, 
#                                                                    save_freq='epoch', ave_weights_only=False, monitor="val_mean_absolute_percentage_error")
    
    
   
    
    ## Fit model
    model.fit(x = avoid_error(train_generator),
              steps_per_epoch = STEP_SIZE_TRAIN,
              validation_data = val_generator,
              validation_steps = STEP_SIZE_VALID,
              epochs = EPOCHS,
              callbacks = [metrics, tensorboard_cb])
    
    model.save(Model2save)
    ### print
    print(f"Save Linear regression Vitsiontranformer at: {Model2save}")
    print(f"*"*100)
    

    
## Run Function 
if __name__ == '__main__':
    main()












