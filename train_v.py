# fine-tune image model

import os
import time
import numpy as np
import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as plt
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[1],True)
tf.config.set_visible_devices(gpus[1],'GPU')

import data_v
from model import image_model
import config


BATCH_SIZE = 128
TOTAL_NUM = 134349
TRAIN_NUM = 114196
VAL_NUM = 13435
TEST_NUM = 6718
EPOCHS = 20
TRAIN_STEPS = tf.math.ceil(TRAIN_NUM / BATCH_SIZE).numpy()
VAL_STEPS = tf.math.floor(VAL_NUM / BATCH_SIZE).numpy() 
TEST_STEPS = tf.math.floor(TEST_NUM / BATCH_SIZE).numpy() 


def train():
    image_model_weights_path = 'model_weights/base_model_{}_image_task_{}_BS{}_Epoch{}.h5'.format(config.base_model,config.image_task,config.batch_size,config.epoch)
    print('base_model: {}, image_task: {}, image_scale: {}'.format(config.base_model,config.image_task,config.image_scale))
    print('epoch: {}, batch_size: {}'.format(config.epoch, config.batch_size))
    print('image_model_weights_path: {}'.format(image_model_weights_path))
    
    model_ckp = tf.keras.callbacks.ModelCheckpoint(filepath=image_model_weights_path,monitor='val_loss',save_best_only=True,save_weights_only=True)
    train_ds = data_v.get_image_ds('train',config.batch_size,CD=config.image_task, base_model=config.base_model) 
    val_ds = data_v.get_image_ds('val',config.batch_size,CD=config.image_task, base_model=config.base_model)
    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.001,patience=2)
    model = image_model.CDR_CD(base_model=config.base_model,task=config.image_task)
    model.build(input_shape=(None,224,224,3))
    model.summary()
    if config.image_task == 'CD':
    	model.compile(optimizer=tf.keras.optimizers.Adam(),loss=[tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.CategoricalCrossentropy()],loss_weights=[1.0,1.0],metrics=[tf.keras.metrics.CategoricalAccuracy()])
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.CategoricalCrossentropy(),metrics=[tf.keras.metrics.CategoricalAccuracy()])
    H = model.fit(train_ds,validation_data=val_ds,epochs=config.epoch, initial_epoch=0, verbose=2,callbacks=[model_ckp])  
    #model.save_weights(model_weights_path)

def evaluate():
    #model_weights_path = "model_weights/base_model_mobilenet_image_task_CD_BS64_Epoch20.h5"
    model_weights_path = "model_weights/base_model_xception_image_task_CD_BS128_Epoch20.h5"
    #model_weights_path = "model_weights/base_model_resnet_image_task_CD_BS64_Epoch20.h5"
    base_model = 'xception'
    val_ds = data_v.get_image_ds('val',128,CD='CD', base_model=base_model)
    
    model = image_model.CDR_CD(base_model=base_model, task='CD')
    model.build(input_shape=(None,224,224,3))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(),loss=[tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.CategoricalCrossentropy()],loss_weights=[1.0,1.0],metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.load_weights(model_weights_path)
    model.evaluate(val_ds)
    

if __name__ == '__main__':
    train()
    #evaluate()
