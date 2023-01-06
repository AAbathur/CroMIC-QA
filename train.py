#!/usr/bin/env python
# -*- coding:utf-8 -*-

# train answer re-ranking model
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_ranking as tfr 
from tensorflow.keras.utils import plot_model
import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as plt

import data
from model import rank_model 
import config



parser = argparse.ArgumentParser()
parser.add_argument('--batch-size',default=config.batch_size,type=int)
parser.add_argument('--model_path',default='',type=str,help='path to save model')
parser.add_argument('--num-epochs',default=config.epoch,type=int)
parser.add_argument('--data_type',default=config.data_type,type=int)
parser.add_argument('--plot_path',default='',help='path to save loss figure')
args = parser.parse_args()

if args.data_type == 1:
    TRAIN_STEPS = tf.math.ceil(17601 *4 / args.batch_size).numpy()
    VAL_STEPS = tf.math.floor(2071 *4 / args.batch_size).numpy()        
    TEST_STEPS = tf.math.floor(1036 *4 / args.batch_size).numpy() 
if args.data_type == 2:
    TRAIN_STEPS = tf.math.ceil(39314 *4 / args.batch_size).numpy()
    VAL_STEPS = tf.math.floor(4625 *4 / args.batch_size).numpy() 
    TEST_STEPS = tf.math.floor(2313 *4 / args.batch_size).numpy() 

def NDCG1(labels, predictions):
    return tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.NDCG, topn=1)(labels,predictions,features=None)
def NDCG2(labels, predictions):
    return tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.NDCG, topn=2)(labels,predictions,features=None)
def NDCG3(labels, predictions):
    return tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.NDCG, topn=3)(labels,predictions,features=None)
def NDCG4(labels, predictions):
    return tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.NDCG, topn=4)(labels,predictions,features=None)
def NDCG6(labels, predictions):
    return tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.NDCG, topn=6)(labels,predictions,features=None)
def MAP(labels, predictions):
    return tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.MAP)(labels,predictions,features=None)
def MRR(labels, predictions):
    return tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.MRR)(labels,predictions,features=None)
def ARP(labels, predictions):
    return tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.ARP)(labels,predictions,features=None)
def DCG(labels, predictions):
    return tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.DCG)(labels,predictions,features=None)
def PRECISION_1(labels, predictions):
    return tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.PRECISION,topn=1)(labels,predictions,features=None)
def PRECISION_2(labels, predictions):
    return tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.PRECISION,topn=2)(labels,predictions,features=None)
def PRECISION_3(labels, predictions):
    return tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.PRECISION,topn=3)(labels,predictions,features=None)
def PRECISION_4(labels, predictions):
    return tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.PRECISION,topn=4)(labels,predictions,features=None)
def PRECISION_5(labels, predictions):
    return tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.PRECISION,topn=5)(labels,predictions,features=None)
def ORDERED_PAIR_ACCURACY(labels, predictions):
    return tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY)(labels,predictions,features=None)

def Binary_acc(labels,logits):
    logits = tf.keras.activations.sigmoid(logits)
    return tf.keras.metrics.binary_accuracy(labels,logits)

def PairwiseLogisticLoss(labels, logits):
    loss_fn = tfr.losses.make_loss_fn('pairwise_logistic_loss')
    return loss_fn(labels,logits,features=None)

def train():
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.001,patience=2)
 
    if config.fusion == 'conv_fusion' or config.att_type == 'q_guided' or config.att_type == 'co-att-a' or config.att_type == 'co-att-p':
        train_ds = data.rank_ds_all_img_vec('train',args.batch_size,config.data_type,config.image_type)
        val_ds = data.rank_ds_all_img_vec('val',args.batch_size,config.data_type,config.image_type)
        test_ds = data.rank_ds_all_img_vec('test',args.batch_size,config.data_type,config.image_type)
    elif config.data_type in [1,2,3]:
        train_ds = data.rank_ds('train',args.batch_size,config.data_type,config.image_type, answer_file_index=config.answer_file_index)
        val_ds = data.rank_ds('val', args.batch_size,config.data_type,config.image_type, answer_file_index=config.answer_file_index)
        test_ds = data.rank_ds('test', args.batch_size,config.data_type,config.image_type, answer_file_index=config.answer_file_index)
    elif config.data_type == 6:
        train_ds = data.rank_ds_mix('train',args.batch_size, image_type=config.image_type)
        val_ds = data.rank_ds_mix('val',args.batch_size, image_type=config.image_type)
        test_ds = data.rank_ds_mix('test', args.batch_size, image_type=config.image_type)
    print('dataset done')
    model = rank_model.RANK_MODEL()
    img_vec_dim = 1280
    if config.image_type == 'RN-FT' or config.image_type == 'XC-FT':
        img_vec_dim = 2048
    if config.fusion == 'conv_fusion' or config.att_type == 'q_guided' or config.att_type == 'co-att-a' or config.att_type == 'co-att-p':
        model.build(input_shape=[(None,30),(None,6,50),(None,7,7,img_vec_dim)])
    else:
        model.build(input_shape=[(None,30),(None,6,50),(None,img_vec_dim)])
    print('model build done!')
    model.summary()
    print('-'*100)
    #model_arch_path = "tmp/Rank_Arch_Fusion_{}_Att_{}.png".format(config.fusion,config.att_type)
    model_weight_path = "model_weights/Rank_datatype6_normal_concat_MN_FT.h5"
    model_ckp = tf.keras.callbacks.ModelCheckpoint(filepath=model_weight_path,monitor='val_loss',save_best_only=True,save_weights_only=True)
    print('att_type: {}, att_id: {}, fusion: {}'.format(config.att_type, config.att_id, config.fusion))
    print("answer_file_index: {}".format(config.answer_file_index))
    print('reply_num: {}'.format(config.reply_num))
    print("data_type: {}, image_type: {}".format(config.data_type, config.image_type))
    print('hyper_parameter: ')
    print('q,a feature: ', config.q_feature)
    print('v_mid_feature: ', config.v_mid_feature)
    print('config.epoch: ',config.epoch)
    print('epochs: {}, lr: {}, batch_size: {}'.format(args.num_epochs, config.learning_rate, config.batch_size))
    print('with_q: {}'.format(config.with_q))
    print('fc1 initializer: {}, fc1_init_scale: {}, fc3 is same as fc1'.format(config.fc1_initializer, config.fc1_init_scale))
    print('with_v: {}, fc2 initializer: {}, fc2_init_scale: {}'.format(config.with_v, config.fc2_initializer, config.fc2_init_scale))
    print('lstm_initializer: {}, lstm_scale: {}, fc4_initializer: {}, fc4_scale: {}, fc5_initializer: {}, fc5_init_scale: {}'.format(config.lstm_initializer, config.lstm_init_scale, config.fc4_initializer, config.fc4_init_scale, config.fc5_initializer, config.fc5_init_scale))
    print('-'*100)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate), loss=PairwiseLogisticLoss, metrics=[MAP,MRR,ORDERED_PAIR_ACCURACY,PRECISION_1,PRECISION_2,PRECISION_3,PRECISION_4,PRECISION_5,ARP,Binary_acc])
    
    H = model.fit(train_ds, validation_data=val_ds, epochs=config.epoch,initial_epoch=0,callbacks=[model_ckp],verbose=2)
    #model.save(model_weight_path)

def evaluate():
    data_type = 6
    model_weight= "model_weights/Rank_datatype6_concat_MN_FT.h5"
    val_ds = data.rank_ds_mix('val',args.batch_size, image_type=config.image_type)
    model = rank_model.RANK_MODEL()
    model.build(input_shape=[(None,30),(None,6,50),(None,1280)])
    print('model build done!')
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),loss=PairwiseLogisticLoss,metrics=[MAP,MRR,ORDERED_PAIR_ACCURACY,PRECISION_1,PRECISION_2,PRECISION_3,PRECISION_4,PRECISION_5,ARP,Binary_acc])
    model.load_weights(model_weight)
    model.evaluate(val_ds)


if __name__ == "__main__":

    train()
  
   
