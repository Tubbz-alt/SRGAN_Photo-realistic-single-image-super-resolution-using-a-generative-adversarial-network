# Resource
# Figure4: arXiv:1609.04802v5

import tensorflow as tf




def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, block_size=scale)


class ResidualConvLayer(tf.keras.layers.Layer):
    '''
    About PRelu
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


class LRImgScaleLayer(tf.keras.layers.Layer):
    '''
    arXiv:1609.04802v5
    3.2 Training details and parameters
    We scaled the range of the LR input images to [0, 1]
    and for the HR images to [−1, 1]. The
    '''
    def __init_(self, **kwargs):
        super(LRImgScaleLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs / 255.


class SRGenerator(tf.keras.Model):
    '''
    Generator takes
    input:  LR img
    Output: SR img
    '''
    def __init__(self, n_res_layers=16, **kwargs):
        super(SRGenerator, self).__init__(**kwargs)
        self.n_res_layers = n_res_layers

        self.normalize_layer = LRImgScaleLayer()
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

        self.shufflers = [PixelShuffle(filters=256, kernel_size=3, strides=1, padding='same', scale=2) for _ in range(2)]

        self.out_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=9, strides=1, padding='same', activation='tanh', name='output')

    def call(self, inputs):
        norm_result = self.normalize_layer(inputs)
        x = self.conv_1(norm_result)
        x = self.conv_1_act(x)

        res = x

        for res_layer in self.res_blocks:
            x = res_layer(x)
        x = self.conv_2(x)
        x = self.conv_2_bn(x)
        x = self.add_layer([x, res])

        for shuffle_layer in self.shufflers:
            x = shuffle_layer(x)

        return self.out_layer(x)



if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    test_generator = SRGenerator(n_res_layers=16)
    tmp_input = tf.random.normal(shape=(2, 64, 64, 3))
    test_output = test_generator(tmp_input)
    test_output[0].shape


