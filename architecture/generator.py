# Resource
# Figure4: arXiv:1609.04802v5

import tensorflow as tf




def preprocess_img(x):
    x = tf.cast(x, tf.float32)
    return x / 127.5 - 1


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, block_size=scale)



class ResidualConvLayer(tf.keras.layers.Layer):
    '''
    The axes along which to share learnable parameters for the activation function.
    For example, if the incoming feature maps are from a 2D convolution with output shape (batch, height, width, channels),
    and you wish to share parameters across space so that each filter only has one set of parameters,
    set shared_axes=[1, 2].
    '''
    def __init__(self, filters, kernel_size, strides, padding, **kwargs):
        super(ResidualConvLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           kernel_initializer=tf.keras.initializers.random_normal(mean=0.0, stddev=0.02))
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-9)
        self.act = tf.keras.layers.PReLU(shared_axes=[1,2])

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.conv_1 = ResidualConvLayer(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
        self.conv_2 = ResidualConvLayer(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)

    def call(self, inputs):
        res = inputs
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        return x + res



class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, scale, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.scale = scale
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           kernel_initializer=tf.keras.initializers.random_normal(mean=0.0, stddev=0.02))
        self.shuffler = tf.keras.layers.Lambda(pixel_shuffle(self.scale))
        self.act = tf.keras.layers.PReLU(shared_axes=[1,2])

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.shuffler(x)
        return self.act(x)




class ImgScaleLayer(tf.keras.layers.Layer):
    def __init_(self, **kwargs):
        super(ImgScaleLayer, self).__init__(**kwargs)
        self.normalize = tf.keras.layers.Lambda(lambda x: preprocess_img(x))

    def call(self, inputs):
        return self.normalize(inputs)


class SRGenerator(tf.keras.Model):
    def __init__(self, n_res_layers, **kwargs):
        super(SRGenerator, self).__init__(**kwargs)
        self.n_res_layers = n_res_layers

        self.normalize_layer = ImgScaleLayer()
        self.conv_1 = tf.keras.layers.Conv2D(filters=64,
                                             kernel_size=9,
                                             strides=1,
                                             padding='same',
                                             kernel_initializer=tf.keras.initializers.random_normal(mean=0.0, stddev=0.02))
        self.conv_1_act = tf.keras.layers.PReLU(shared_axes=[1,2])
        self.res_blocks = [ResidualBlock(filters=64, kernel_size=3, strides=1, padding='same') for _ in range(self.n_res_layers)]
        self.conv_2 = tf.keras.layers.Conv2D(filters=64,
                                             kernel_size=9,
                                             strides=1,
                                             padding='same',
                                             kernel_initializer=tf.keras.initializers.random_normal(mean=0.0,
                                                                                                    stddev=0.02))
        self.conv_2_bn = tf.keras.layers.BatchNormalization(epsilon=1e-9)
        self.add_layer = tf.keras.layers.Add()

        self.shufflers = [PixelShuffle(filters=256, kernel_size=3, strides=1, padding='same') for _ in range(2)]

        self.out_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=9, strides=1, padding='same', name='output')

    def call(self, inputs):
        pass

