#!/usr/bin/env python
# -*- coding:utf-8 -*-

# embedding layer + GRU
import pickle
import os
import six
import array
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import time
import data
class TextProcessor(keras.layers.Layer):
    def __init__(self,sequence_len,embedding_features,lstm_features,drop=0.0,use_hidden=True,use_tanh=False,only_embed=False,kernel_initializer='glorot_uniform',kernel_init_scale=1.0):
        """ seqence_len: 输入的句子的长度
            embedding_feature: 输入的句子中每个token对应词向量维度
            lstm_feature: lstm中状态向量维度
            use_hidden: True表示返回last output, False表示返回整个output sequence
         """
        super(TextProcessor, self).__init__()
        self.use_hidden = use_hidden# return last layer hidden, else return all the outputs for each words
        self.use_tanh = use_tanh
        self.only_embed = only_embed
        self.sequence_len = sequence_len
        
        self.embedding_features = embedding_features
        ### complete embedding tensor
        self.masking = tf.keras.layers.Masking(mask_value=0.,input_shape=(self.sequence_len,self.embedding_features))
        self.embedding_table = tf.convert_to_tensor(np.load("data/WV/vocab_vec_200d.npy"),dtype=tf.float32)
        self.drop = drop
        
        if not self.only_embed:
            if kernel_initializer == 'glorot_uniform':
                initializer = tf.keras.initializers.VarianceScaling(scale=kernel_init_scale,mode='fan_avg',distribution='uniform')
            elif kernel_initializer == 'glorot_normal':
                initializer = tf.keras.initializers.VarianceScaling(scale=kernel_init_scale,mode='fan_avg',distribution='truncated_normal')
            else:
                raise KeyError('parameter kernel_initializer wrong')
            self.lstm = keras.layers.GRU(units=lstm_features,return_sequences=True,return_state=True,kernel_initializer=initializer)

    def call(self,x):
        
        embedded = tf.nn.embedding_lookup(self.embedding_table,x)
       
        embedded = keras.layers.Dropout(self.drop,name='textprocessor_dropout')(embedded)

        if self.use_tanh:
            embedded = tf.math.tanh(embedded)
        if self.only_embed:
            return embedded
        # masking 处理变长序列
        masked_embedding = self.masking(embedded)
        if len(masked_embedding.shape) == 4:
            # it means input x is replies list, with shape (batch, num_sequences, sequence_len, embedding_feature)
            # if len(masked_embedding.shape) == 3, it means input x is question, with shape (batch,sequence_len, embedding_feature)
            masked_embedding_list = tf.unstack(masked_embedding, axis=1)
           
            if self.use_hidden:
                hid_list = []
                for embedding in masked_embedding_list:
                    _, hid = self.lstm(embedding)
                    hid_list.append(hid)
                return tf.stack(hid_list, axis=1)
            else:
                out_list = []
                for embedding in masked_embedding_list:
                    out, _ = self.lstm(embedding)
                    out_list.append(out)
                return tf.stack(out_list, axis=1)
        else:
            if self.use_hidden:
                # return last state
                _, hid = self.lstm(masked_embedding)
                return hid
            else:
                # return whole seqence
                out, _ = self.lstm(masked_embedding)
                return out
class TextProcessor_BiLSTM(keras.Model):
    def __init__(self,sequence_len,embedding_features,lstm_features,drop=0.0,use_hidden=True,use_tanh=False,bilstm=True,only_embed=False):
        super(TextProcessor_BiLSTM, self).__init__()
        self.bilstm = bilstm
        self.use_hidden = use_hidden# return last layer hidden, else return all the outputs for each words
        self.use_tanh = use_tanh
        self.only_embed = only_embed
        self.sequence_len = sequence_len
        self.embedding_features = embedding_features
 
        self.masking = tf.keras.layers.Masking(mask_value=0.,input_shape=(self.sequence_len,self.embedding_features))
        self.embedding_table = tf.convert_to_tensor(np.load(r"data\WV\vocab_vec_200d_181431_set_0.npy"))
        self.drop = drop
        
        if not self.only_embed:
            # keras.layers.GRU默认只输出最后一个
            self.lstm = keras.layers.GRU(units=lstm_features,return_sequences=True,return_state=True)
            if bilstm:
                self.backward_lstm = keras.layers.GRU(units=lstm_features, return_sequences=True, return_state=True,go_backwards=True)
                self.bilstm = keras.layers.Bidirectional(self.lstm, backward_layer=self.backward_lstm, merge_mode='ave')
            
    def call(self,x):
        embedded = tf.nn.embedding_lookup(self.embedding_table,x)
        embedded = keras.layers.Dropout(self.drop)(embedded)
        if self.use_tanh:
            embedded = tf.math.tanh(embedded)
        if self.only_embed:
            return embedded
        # masking 处理变长序列
        masked_embedding = self.masking(embedded)
        if len(masked_embedding.shape) == 4:
            # it means input x is replies list, with shape (batch, num_sequences, sequence_len, embedding_feature)
            # if len(masked_embedding.shape) == 3, it means input x is question, with shape (batch,sequence_len, embedding_feature)
            masked_embedding_list = tf.unstack(masked_embedding, axis=1)
            if self.bilstm:
                out_list = []
                for embedding in masked_embedding_list:
                    out, _, _= self.bilstm(embedding)
                    out_list.append(out)
                return tf.stack(out_list, axis=1)

            elif not self.bilstm and self.use_hidden:
                hid_list = []
                for embedding in masked_embedding_list:
                    _, hid = self.lstm(embedding)
                    hid_list.append(hid)
                return tf.stack(hid_list, axis=1)
            else:
                out_list = []
                for embedding in masked_embedding_list:
                    out, _ = self.lstm(embedding)
                    out_list.append(out)
                return tf.stack(out_list, axis=1)
        else:
            if self.bilstm:
                out, _, _ = self.bilstm(masked_embedding) 
                return out
            elif not self.bilstm and self.use_hidden:
                _, hid = self.lstm(masked_embedding)
                return hid
            else:
                out, _ = self.lstm(masked_embedding)
                return out



 
    
    
