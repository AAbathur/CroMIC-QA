import numpy as np
import tensorflow as tf


class Bilinear(tf.keras.layers.Layer):
    def __init__(self,units=1000,input_dims=[512,1024]):
        super(Bilinear, self).__init__()
        self.units = units
        self.input_dims = input_dims
        #w_init = tf.random_normal_initializer()
    def build(self,input_shape):
        self.w = self.add_weight(name='bilinear_w',shape=(self.units,self.input_dims[0],self.input_dims[1]),dtype="float32",initializer=tf.keras.initializers.GlorotUniform(),trainable=True)
    
        self.b = self.add_weight(name='bilinear_b',shape=(self.units,),dtype="float32",initializer=tf.keras.initializers.Zeros(),trainable=True)
    def call(self,inputs):
        x = inputs[0] # shape: batch_size, 512
        y = inputs[1] # shape: batch_size, 1024
        x = tf.expand_dims(x,1) # bs,1,512
        x = tf.expand_dims(x,3) # bs,1,512,1
        x = tf.tile(x, tf.stack([1,self.units,1,self.input_dims[1]])) # bs,1000,512,1024
        xw = tf.reduce_sum(tf.multiply(x,self.w),2) # bs,1000,1024
        y = tf.expand_dims(y,1) # bs,1,1024
        y = tf.tile(y,tf.stack([1,self.units,1])) # bs,1000,1024
        output = tf.reduce_sum(tf.multiply(xw,y),2) + self.b # bs,1000
        return output

class MFB(tf.keras.layers.Layer):
    def __init__(self, units, sum_k):
        super(MFB, self).__init__()
        assert units%sum_k == 0
        self.units = units
        self.sum_k = sum_k
        self.U_layer = tf.keras.layers.Dense(units)
        self.V_layer = tf.keras.layers.Dense(units)
        self.dp_layer = tf.keras.layers.Dropout(0.5)

    def call(self,inputs):
        x = inputs[0]
        y = inputs[1]
        ux = self.U_layer(x)
        vy = self.V_layer(y)
        z = tf.math.multiply(ux,vy)
        z = self.dp_layer(z)
        z = tf.reshape(z,shape=(-1,int(self.units/self.sum_k),self.sum_k))
        z = tf.reduce_sum(z,2)
        # power norm
        #z = tf.math.multiply(tf.math.sign(z),tf.math.sqrt(tf.math.abs(z)+1e-12))
        z = tf.math.l2_normalize(z)
        return z

class Conv_Fusion(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Conv_Fusion,self).__init__()
        self.units = units
        self.conv_2d = tf.keras.layers.Conv2D(filters=self.units,kernel_size=(3,3),activation='tanh',padding='same')
        self.pool_layer = tf.keras.layers.GlobalAveragePooling2D()
    def call(self,inputs):
        x = inputs[0] # bs,128
        y = inputs[1] # bs,7,7,1024
        x = tf.expand_dims(x,1)
        x = tf.expand_dims(x,2)
        x = tf.tile(x,tf.stack([1,7,7,1])) # bs,7,7.128
        z = tf.concat((x,y),3) # bs,7,7,1024+128
        z = self.conv_2d(z) # bs,7,7,256
        output = self.pool_layer(z)
        return output

if __name__ == '__main__':
    x = tf.random.normal((8,24))
    y = tf.random.normal((8,32))
    mfb = MFB(64,4)
    z = mfb([x,y])
    print(z.shape)
