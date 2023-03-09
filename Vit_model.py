import tensorflow as tf
import tensorflow_addons as tfa
import glob, warnings
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import callbacks as callbacks_
from tensorflow.keras import layers
from keras import models
from vit_keras import vit, utils



def build_model(fine_tune, image_size):
    """
    :param fine_tune (bool): Whether to train the hidden layers or not.
    """
    
    vit_model = vit.vit_l32(image_size, activation = 'linear', pretrained = True, 
                            include_top = False, pretrained_top = False, classes = 1)
    print('[INFO]: Loading pre-trained weights')
    x = vit_model.get_layer('ExtractToken').output### add the tail layer ###  
    Flatten_layer1 = layers.Flatten()(x)
    BatchNormalization_layer1 = layers.BatchNormalization(name='BatchNormalization_1')(Flatten_layer1)
    Dense_layer1 = layers.Dense(64, activation='gelu',name='Dense_regress')(BatchNormalization_layer1)
    BatchNormalization_layer2 = layers.BatchNormalization(name='BatchNormalization_2')(Dense_layer1)
    Dense_layer2 = layers.Dense(1, activation='linear',name='linear_regress')(BatchNormalization_layer2)
    model = models.Model(inputs= vit_model.input, outputs=[Dense_layer2], name='Vision_transformer_regression') 
    #model.summary()

    print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.trainable_weights))

    if fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for layer in vit_model.layers:
            layer.trainable = False

    print('This is the number of trainable layers '
              'after freezing the conv base:', len(model.trainable_weights))
    print('-'*125)

    return model


def build_Functional_model(fine_tune, image_size):
    """
    :param fine_tune (bool): Whether to train the hidden layers or not.
    """
    
    vit_model = vit.vit_l32(image_size, activation = 'linear', pretrained = True, 
                            include_top = False, pretrained_top = False, classes = 1)
    print('[INFO]: Loading pre-trained weights')
    model = tf.keras.Sequential([
        vit_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation = tfa.activations.gelu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation = tfa.activations.gelu),
        tf.keras.layers.Dense(32, activation = tfa.activations.gelu),
        tf.keras.layers.Dense(1, 'linear')
    ],
    name = 'visionregress_transformer')

    print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.trainable_weights))

    if fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for layer in vit_model.layers:
            layer.trainable = False

    print('This is the number of trainable layers '
              'after freezing the conv base:', len(model.trainable_weights))
    print('-'*100)

    return model



            
            
            
            
            