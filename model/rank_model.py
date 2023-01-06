import sys
import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow import keras
from reuse_modules import FCNet,Fusion
from fusion import *
import config
import word_embedding
import numpy as np

class CoAtt_P(tf.keras.layers.Layer):
    def __init__(self,dim):
        super(CoAtt_P,self).__init__()
        self.dim = dim
        self.w = self.add_weight(name='coatt-w',shape=(self.dim,self.dim),dtype="float32",
            initializer=tf.keras.initializers.GlorotUniform(),trainable=True)
        
    def call(self,q,v):
        qw = tf.matmul(q,self.w)
        qwv = tf.matmul(qw,v,transpose_b=True)
        qwv = tf.math.tanh(qwv) 

        dist1 = tf.reduce_max(qwv,axis=1,keepdims=True) 
        dist1 = tf.math.softmax(dist1)
        dist2 = tf.reduce_max(qwv,axis=2) 
        dist2 = tf.expand_dims(dist2,axis=1) 
        dist2 = tf.math.softmax(dist2)

        att_v = tf.squeeze(tf.matmul(dist1, v),1)
        att_q = tf.squeeze(tf.matmul(dist2, q),1)
    
        return att_q, att_v

class CoAtt_A(tf.keras.layers.Layer):
    def __init__(self):
        super(CoAtt_A,self).__init__()
        self.att_q_layer = tf.keras.layers.Attention()
        self.att_v_layer = tf.keras.layers.Attention()

    def call(self,q,v,q_mask,v_mask):
        tmp_v = tf.reduce_mean(v,1,keepdims=True) 
        v_mask = tf.ones_like(tmp_v)
        v_mask = tf.reduce_sum(v_mask,2)
        v_mask = tf.cast(v_mask,tf.bool)
     
        att_q = self.att_q_layer([tmp_v,q],[v_mask,q_mask])
        att_v = self.att_v_layer([att_q,v])

        att_q = tf.squeeze(att_q,1)
        att_v = tf.squeeze(att_v,1)
        return att_q,att_v

class Att_1(tf.keras.layers.Attention):
    # general attention
    def __init__(self,dim):
        super(Att_1, self).__init__()
        self.dim = dim
        self.w = self.add_weight(shape=(self.dim,self.dim),dtype="float32",
            initializer=tf.keras.initializers.GlorotUniform(),trainable=True)
    def _calculate_scores(self, query, key):
        qw = tf.matmul(query,self.w)
        qwv = tf.matmul(qw,key,transpose_b=True)
        return qwv

class Att_2(tf.keras.layers.Attention):
    # perceptron attention
    def __init__(self,dim):
        super(Att_2,self).__init__()
        self.dim = dim
        self.w1 = self.add_weight(shape=(self.dim,self.dim),dtype="float32",
            initializer=tf.keras.initializers.GlorotUniform(),trainable=True)
        self.w2 = self.add_weight(shape=(self.dim,self.dim),dtype="float32",
            initializer=tf.keras.initializers.GlorotUniform(),trainable=True)
        self.v = self.add_weight(shape=(1,self.dim),dtype="float32",
            initializer=tf.keras.initializers.GlorotUniform(),trainable=True)
    def _calculate_scores(self, query, key):
        qw1 = tf.matmul(query,self.w1)
        vw2 = tf.matmul(key,self.w2)
        x = qw1 + vw2
        x = tf.math.tanh(x)
        scores = tf.matmul(self.v,x,transpose_b=True) 
        return scores
class Att_3(tf.keras.layers.Attention):
    # concat attention
    def __init__(self,dim,tv=30):
        super(Att_3,self).__init__()
        self.tv = tv 
        self.dim = dim
        self.w = self.add_weight(shape=(1,self.dim*2),dtype="float32",
            initializer=tf.keras.initializers.GlorotUniform(),trainable=True)
    def _calculate_scores(self, query, key):
        query = tf.tile(query,tf.stack([1,self.tv,1]))
        qk = tf.concat([query,key],2) 
        scores = tf.matmul(self.w,qk,transpose_b=True)
        return scores

class RANK_MODEL(tf.keras.Model):
    def __init__(self):
        super(RANK_MODEL, self).__init__()
        
        question_len = config.q_length
        replies_len = config.a_length

        embedding_dim = config.embedding_dim
        self.att_type = config.att_type
        self.reply_features = config.a_feature
        self.vision_features = config.v_feature
        self.question_features = config.q_feature
        self.v_mid_feature = config.v_mid_feature
        self.fusion = config.fusion
        self.reply_num = config.reply_num
        self.with_v = config.with_v
        self.with_q = config.with_q
        lstm_use_hidden = True
        self.att_id = config.att_id
        
        assert self.with_v or self.with_q
        if self.att_type:
            assert self.with_v and self.with_q
            if self.att_type == 'v_guided':
                lstm_use_hidden = False
                if self.att_id == 0:
                    self.att_layer = tf.keras.layers.Attention()
                elif self.att_id == 1:
                    self.att_layer = Att_1(dim=self.v_mid_feature)
                elif self.att_id == 2:
                    self.att_layer = Att_2(dim=self.v_mid_feature)
                elif self.att_id == 3:
                    self.att_layer = Att_3(dim=self.v_mid_feature)
               
            elif self.att_type == 'q_guided':
                lstm_use_hidden = True
                if self.att_id == 0:
                    self.att_layer = tf.keras.layers.Attention()
                elif self.att_id == 1:
                    self.att_layer = Att_1(dim=self.v_mid_feature)
                elif self.att_id == 2:
                    self.att_layer = Att_2(dim=self.v_mid_feature)
                elif self.att_id == 3:
                    self.att_layer = Att_3(dim=self.v_mid_feature)
            elif self.att_type == 'co-att-p':
                self.co_att = CoAtt_P(self.v_mid_feature)
                lstm_use_hidden = False
            elif self.att_type == 'co-att-a':
                self.co_att = CoAtt_A()
                lstm_use_hidden = False
        if self.with_q:
            self.question = word_embedding.TextProcessor(
                sequence_len = question_len,
                embedding_features = embedding_dim,
                lstm_features = self.question_features,
                drop=0.0,
                use_hidden=lstm_use_hidden,
                only_embed = config.only_embed,
                kernel_initializer=config.lstm_initializer,
                kernel_init_scale=config.lstm_init_scale)

        self.replies = word_embedding.TextProcessor(
            sequence_len = replies_len,
            embedding_features = embedding_dim,
            lstm_features = self.reply_features,
            drop=0.0,
            only_embed = config.only_embed,
            kernel_initializer=config.lstm_initializer,
            kernel_init_scale=config.lstm_init_scale)

        if self.with_v:
            self.linl2 = FCNet(self.vision_features, self.v_mid_feature, activate='tanh',kernel_initializer=config.fc2_initializer, kernel_init_scale=config.fc2_init_scale)
        if self.with_q and self.with_v:
            if self.fusion == 'bilinear':
                self.fusion_layer = Bilinear(units=self.question_features+self.v_mid_feature,input_dims=[self.question_features, self.v_mid_feature])
            elif self.fusion == 'conv_fusion':
                self.fusion_layer = Conv_Fusion(units=self.question_features+self.v_mid_feature)
            elif self.fusion == 'mfb':
                self.fusion_layer = MFB(units=self.question_features+self.v_mid_feature, sum_k=1)
        self.linl4 = FCNet(256, 128, activate='tanh',kernel_initializer=config.fc4_initializer, kernel_init_scale=config.fc4_init_scale)
        self.linl5 = FCNet(128, 1, activate=None,kernel_initializer=config.fc5_initializer, kernel_init_scale=config.fc5_init_scale)
     
    def call(self,x):
        if self.with_q:
            q = x[0]
            q = tf.cast(q,tf.int32)
            q_mask = tf.less(q,181431)
            q = self.question(q)   

        if self.with_v:
            v = x[2]  
            v = self.linl2(v)

        if self.with_v and self.with_q:
            if self.att_type == 'v_guided':
                v1 = tf.expand_dims(v,1)
                v_mask = tf.ones_like(v1)
                v_mask = tf.reduce_sum(v_mask,axis=2)
                v_mask = tf.cast(v_mask,dtype=tf.bool)
                att_q = self.att_layer([v1,q],[v_mask,q_mask])
                q = tf.squeeze(att_q,1)
            elif self.att_type == 'q_guided':
                v = tf.reshape(v,(-1,49,self.v_mid_feature))
                q1 = tf.expand_dims(q,1)
                att_v = self.att_layer([q1,v])
                v = tf.squeeze(att_v,1)
            elif self.att_type == 'co-att-p':
                v = tf.reshape(v,(-1,49,self.v_mid_feature))
                q, v = self.co_att(q,v)
            elif self.att_type == 'co-att-a':
                v = tf.reshape(v,(-1,49,self.v_mid_feature))
                v_mask = tf.ones_like(v)
                v_mask = tf.reduce_sum(v_mask,axis=2)
                v_mask = tf.cast(v_mask,dtype=tf.bool)
                q, v = self.co_att(q,v,q_mask,v_mask)

            if self.fusion == 'concat':
                Q = tf.concat([q,v],1)       
            else:
                Q = self.fusion_layer([q,v])
        else:
            if self.with_q:
                Q = q
            if self.with_v:
                Q = v
        Q = tf.expand_dims(Q,axis=1)
        Q = tf.repeat(Q,repeats=[self.reply_num],axis=1)
       
        rs = x[1]
        rs = tf.cast(rs,tf.int32)
        rs = self.replies(rs)
        z = tf.concat([Q,rs],2)
        z = self.linl4(z)
        z = self.linl5(z)
     
        logit  = tf.squeeze(z,axis=2)
        return logit 



