import tensorflow as tf
from tensorflow import keras

class CDR_CD(tf.keras.Model):
    def __init__(self,base_model,task):
        super(CDR_CD, self).__init__()
        self.task = task
        self.base_model = base_model
        if self.base_model == 'mobilenet':
            self.base_layer = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False)
        if self.base_model == 'resnet':
            self.base_layer = tf.keras.applications.ResNet50V2(input_shape=(224,224,3),include_top=False)
        if self.base_model == 'efficientnet': # 24G Memory is not enough
            self.base_layer = tf.keras.applications.EfficientNetB4(input_shape=(224,224,3),include_top=False)
        if self.base_model == 'xception':
            self.base_layer = tf.keras.applications.Xception(input_shape=(224,224,3),include_top=False)
        self.pool_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.mid_layer = tf.keras.layers.Dense(512,activation='relu',name='mid_layer')
        self.dp_layer = tf.keras.layers.Dropout(0.5)
        if self.task == 'CD':
            self.crop_cls = tf.keras.layers.Dense(124,activation=None,name='crop_output')
            self.dise_cls = tf.keras.layers.Dense(91,activation=None,name='dise_output')
        elif self.task == 'C':
            self.crop_cls = tf.keras.layers.Dense(124,activation=None,name='crop_output')
        elif self.task == 'D':
            self.dise_cls = tf.keras.layers.Dense(91,activation=None,name='dise_output')

    def call(self,x):
        x = self.base_layer(x)
        x = self.pool_layer(x)
        x = self.mid_layer(x)
        x = self.dp_layer(x)
        if self.task == 'CD':
            c = self.crop_cls(x)
            c = tf.keras.activations.softmax(c)
            d = self.dise_cls(x)
            d = tf.keras.activations.softmax(d)
            return (c,d)
        elif self.task == 'C':
            c = self.crop_cls(x)
            c = tf.keras.activations.softmax(c)
            return c
        elif self.task == 'D':
            d = self.dise_cls(x)
            d = tf.keras.activations.softmax(d)
            return d



