import tensorflow as  tf

class FCNet(tf.keras.layers.Layer):
    def __init__(self, in_size, out_size, activate=None,norm_type=None,drop=0.0,kernel_initializer='glorot_uniform',kernel_init_scale=1.0):
        super(FCNet, self).__init__()
        self.norm_type = norm_type
        if kernel_initializer == 'glorot_uniform':
            initializer = tf.keras.initializers.VarianceScaling(scale=kernel_init_scale,mode='fan_avg',distribution='uniform')
        elif kernel_initializer == 'glorot_normal':
            initializer = tf.keras.initializers.VarianceScaling(scale=kernel_init_scale,mode='fan_avg',distribution='truncated_normal')
        else:
            raise KeyError('parameter kernel_initializer wrong')
        self.lin = tf.keras.layers.Dense(out_size, kernel_initializer=initializer, name='FCNET_dropout')
        if norm_type == 'LN':
            self.norm = tf.keras.layers.LayerNormalization()
        elif norm_type == 'BN':
            self.norm = tf.keras.layers.BatchNormalization()
        self.drop_value = drop
        self.drop = tf.keras.layers.Dropout(drop)

        # in case of using upper character by mistake
        self.activate = activate.lower() if (activate is not None) else None 
        if self.activate == 'relu':
            self.ac_fn = tf.keras.layers.Activation('relu')
        elif self.activate == 'sigmoid':
            self.ac_fn = tf.keras.layers.Activation('sigmoid')
        elif self.activate == 'tanh':
            self.ac_fn = tf.keras.layers.Activation('tanh')

    def call(self, x):
        if self.drop_value > 0:
            x = self.drop(x)
        x = self.lin(x)
        if self.norm_type:
            x = self.norm(x)
        if self.activate is not None:
            x = self.ac_fn(x)
        return x

